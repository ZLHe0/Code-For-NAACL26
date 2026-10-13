"""Stage 1 on the toy domain, offline: scripted creator, scripted policy, real serving path."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from harness_programs.compression import (NAIVE_NAME, PHASES, Campaign, CampaignConfig, Sizing, Target, by_task, emit, gate,
                                   refine, rollout)
from harness_programs.compression.__main__ import parse_args
from harness_programs.compression.episodes import OptEpisode
from harness_programs.compression.evidence import contrast_pairs, creator_prompt
from harness_programs.compression.measure import final_context, measure, served
from harness_programs.domains import load
from harness_programs.domains.toy import write_split
from harness_programs.domains.toy_agents import DIGEST_PROGRAM, LIVE_EDIT_PROGRAM, ToyCreator, toy_policy
from harness_programs.llm.mock import ScriptedLLM
from harness_programs.serving.compress import NOOP_PROGRAM
from harness_programs.serving.program import Program, load_program, write_program
from harness_programs.serving.steer import Trigger
from harness_programs.splits import HOLDOUT_SPLIT, HoldoutError

TIGHT = Sizing(4096, 350)
LOOSE = Sizing(4096, None)
TARGET = Target("toy", TIGHT)

DROP_LISTINGS = '''\
def compress(state):
    for turn in state[:-1]:
        if turn.action == "list":
            turn.digest = "[listing]"
    return state
'''


class NoPolicy:
    def chat(self, *_, **__):
        raise AssertionError("the policy was called")


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger("harness_programs.serving.budget").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def reservoir(tmp_path_factory) -> list[OptEpisode]:
    """Toy rollouts under the tight limit, two per task."""
    domain = load("toy")
    path = tmp_path_factory.mktemp("rollout") / "episodes.jsonl"
    return rollout.run(domain, domain.load_tasks("synthetic"), ScriptedLLM(toy_policy), TIGHT, path,
                       group_size=2, workers=1)


@pytest.fixture(scope="module")
def mixed(reservoir, tmp_path_factory) -> list[OptEpisode]:
    """The same tasks also rolled out with room to spare, so most tasks have a success and a failure."""
    domain = load("toy")
    path = tmp_path_factory.mktemp("loose") / "episodes.jsonl"
    return reservoir + rollout.run(domain, domain.load_tasks("synthetic"), ScriptedLLM(toy_policy), LOOSE, path,
                                   group_size=1, workers=1)


def _episode() -> OptEpisode:
    """List, open two files, answer; the second file name is readable only in the listing."""
    obs = lambda body: {"role": "user", "content": f"<observation>{body}</observation>"}  # noqa: E731
    act = lambda why, a: {"role": "assistant", "content": f"{why}\n<action>{a}</action>"}  # noqa: E731
    body = "\n".join(["alpha bravo cedar delta ember fjord harbor iris"] * 12)
    messages = [
        {"role": "system", "content": "Find the file."}, {"role": "user", "content": "Which file contains 'jade'?"},
        act("I need the file list.", "list"), obs("file_a.txt\nfile_b.txt"),
        act("Open the first file.", "open file_a.txt"), obs(body),
        act("Open the next file.", "open file_b.txt"), obs(body + " jade"),
        {"role": "assistant", "content": "Found it.\n<answer>file_b.txt</answer>"},
    ]
    return OptEpisode("hand", messages, outcome=1.0, status="answered", gold={"file": "file_b.txt", "word": "jade"})


# ---------------------------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------------------------


def test_rollout_groups_episodes_by_task(reservoir):
    groups = by_task(reservoir)
    assert len(groups) == 16 and all(len(g) == 2 for g in groups.values())
    assert 0 < sum(e.outcome for e in reservoir) / len(reservoir) < 0.6
    info = rollout.summary(reservoir, ["episodes.jsonl"])
    assert info["episodes"] == 32 and info["tasks"] == 16 and info["mixed_outcome_tasks"] == 0


# ---------------------------------------------------------------------------------------------
# ProposeCompress: evidence
# ---------------------------------------------------------------------------------------------


def test_contrast_shows_a_success_and_a_failure_on_the_same_task(mixed):
    pairs, note = contrast_pairs(mixed)
    assert pairs and "mixed outcomes" in note
    assert all(s.task_id == f.task_id and s.outcome > f.outcome for s, f in pairs)
    prompt, evidence = creator_prompt(TARGET, mixed, 3)
    head = prompt.split("## Contract")[0]
    assert "SUCCESSFUL attempt: reward=1.0" in head and "FAILED attempt: reward=0.0" in head
    assert "evidence the successful attempt used" in head and "3 DISTINCT" in head
    assert evidence["contrast"][0]["success_reward"] > evidence["contrast"][0]["failure_reward"]


def test_contrast_falls_back_across_tasks_without_mixed_groups(reservoir):
    pairs, note = contrast_pairs(reservoir)
    assert len(pairs) == 1 and pairs[0][0].outcome > pairs[0][1].outcome and "No task" in note


# ---------------------------------------------------------------------------------------------
# ProposeCompress: measurements
# ---------------------------------------------------------------------------------------------


def test_unfaithful_counts_needed_items_the_program_removes():
    ep = _episode()
    lossy = measure(TARGET, DROP_LISTINGS, [ep])
    assert lossy["compiles"] and not lossy["violations"]
    assert lossy["unfaithful"]["lost"] >= 1 and lossy["unfaithful"]["pct"] > 0
    assert any("file_b.txt" in e for e in lossy["unfaithful"]["examples"])
    assert measure(TARGET, NOOP_PROGRAM, [ep])["unfaithful"]["lost"] == 0


def test_inert_is_the_share_of_pressured_episodes_left_unchanged():
    ep = _episode()
    sizing = Sizing(4096, 250)
    assert served(None, sizing, final_context(ep.messages, TARGET.dsl))[1].evicted
    noop = measure(Target("toy", sizing), NOOP_PROGRAM, [ep])["inert"]
    assert noop["episodes"] == 1 and noop["pct"] == 100.0 and noop["token_saving_pct"] == 0.0
    digest = measure(Target("toy", sizing), DIGEST_PROGRAM, [ep])["inert"]
    assert digest["pct"] == 0.0 and digest["token_saving_pct"] > 5
    assert digest["turns_served"]["with"] > digest["turns_served"]["without"]


def test_measure_reports_contract_violations_as_errors(reservoir):
    assert not measure(TARGET, DIGEST_PROGRAM, reservoir)["violations"]
    assert "live turn" in measure(TARGET, LIVE_EDIT_PROGRAM, reservoir)["violations"][0]
    broken = measure(TARGET, "def compress(state):\n    return state +", reservoir)
    assert not broken["compiles"] and broken["violations"][0].startswith("does not compile")
    raising = "def compress(state):\n    if len(state) > 2:\n        raise ValueError('boom')\n    return state\n"
    assert "raised on" in measure(TARGET, raising, reservoir)["violations"][0]


def test_served_is_none_when_head_alone_overflows(reservoir):
    assert served(None, Sizing(8, None), reservoir[0].messages) is None


# ---------------------------------------------------------------------------------------------
# ProposeCompress: refine loop
# ---------------------------------------------------------------------------------------------


def _creator(review: dict, repair: str = "") -> ScriptedLLM:
    def reply(messages):
        if '"verdict"' in messages[0]["content"]:
            return json.dumps(review)
        return json.dumps({"code": repair})
    return ScriptedLLM(reply)


def test_refine_repairs_until_it_runs_then_accepts(reservoir):
    llm = _creator({"verdict": "accept", "diagnosis": "ok"}, repair=DIGEST_PROGRAM)
    record = refine.refine_one(llm, TARGET, {"name": "c", "code": LIVE_EDIT_PROGRAM}, reservoir)
    assert record["accepted"] and record["code"] == DIGEST_PROGRAM and record["repairs"] == 1
    assert record["measurement"]["inert"]["episodes"] > 0


def test_refine_drops_what_never_runs_or_is_never_accepted(reservoir):
    never_runs = refine.refine_one(_creator({}), TARGET, {"name": "c", "code": LIVE_EDIT_PROGRAM}, reservoir)
    assert not never_runs["accepted"] and never_runs["dropped"].startswith("does not run")
    revise = {"verdict": "revise", "diagnosis": "inert", "code": NOOP_PROGRAM}
    rejected = refine.refine_one(_creator(revise), TARGET, {"name": "c", "code": DIGEST_PROGRAM}, reservoir, rounds=2)
    assert not rejected["accepted"] and len(rejected["rounds"]) == 2 and "not accepted" in rejected["dropped"]


# ---------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------


def test_select_needs_strictly_more_than_no_compression():
    assert gate.select({NAIVE_NAME: 0.5}) == NAIVE_NAME
    assert gate.select({NAIVE_NAME: 0.5, "a": 0.5}) == NAIVE_NAME
    assert gate.select({NAIVE_NAME: 0.5, "a": 0.6, "b": 0.7}) == "b"


def test_gate_scores_every_program_once_on_the_holdout(tmp_path):
    domain = load("toy")
    _, holdout = write_split(tmp_path / "data", n_holdout=8)
    tasks = domain.load_tasks(str(holdout), split=HOLDOUT_SPLIT)
    proposals = [{"name": "digest", "code": DIGEST_PROGRAM}, {"name": "noop", "code": NOOP_PROGRAM}]
    result = gate.run(domain, proposals, tasks, ScriptedLLM(toy_policy), TIGHT, tmp_path)
    assert set(result["scores"]) == {NAIVE_NAME, "digest", "noop"} and result["winner"] == "digest"
    assert result["scores"]["noop"] == result["scores"][NAIVE_NAME] < result["scores"]["digest"] == 1.0
    assert result["compression"] == {"name": "digest", "code": DIGEST_PROGRAM}
    assert len((tmp_path / "digest" / "episodes.jsonl").read_text().splitlines()) == 8
    only_noop = gate.run(domain, proposals[1:], tasks, NoPolicy(), TIGHT, tmp_path)
    assert only_noop["winner"] == NAIVE_NAME and only_noop["compression"] is None


# ---------------------------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------------------------


def _config(out, **kw) -> CampaignConfig:
    """D_hold carved out of the toy's training pool; the evaluation pool is checked to be disjoint."""
    train, holdout = write_split(Path(out) / "data", n_holdout=8)
    fields = {"tasks_path": str(train), "holdout_path": str(holdout), "eval_tasks": ("synthetic",),
              "rollout_tasks": 16, "group_size": 2, "holdout_tasks": 16, "workers": 1, **kw}
    return CampaignConfig(domain="toy", out_dir=str(out), sizing=TIGHT, **fields)


def test_campaign_end_to_end_emits_a_servable_program(tmp_path):
    creator = ToyCreator()
    result = Campaign(_config(tmp_path), llm=creator, policy=ScriptedLLM(toy_policy)).run()

    assert PHASES == ("rollout", "propose", "gate")
    assert all((tmp_path / f"{phase}.json").exists() for phase in PHASES)
    assert result["winner"] == "digest_opened_files"
    assert result["scores"]["digest_opened_files"] > result["scores"][NAIVE_NAME]
    proposed = json.loads((tmp_path / "propose.json").read_text())
    assert [p["name"] for p in proposed["accepted"]] == ["digest_opened_files"]
    dropped = next(c for c in proposed["candidates"] if c["name"] == "truncate_everything")
    assert "live turn" in dropped["dropped"]

    program = load_program(tmp_path / "program.json")
    assert program.harness == "toy" and program.compression["name"] == "digest_opened_files"
    domain = load("toy")
    tasks = domain.load_tasks("synthetic", split="val")
    assert gate.trial(domain, program.compression, tasks, ScriptedLLM(toy_policy), TIGHT, tmp_path / "h0") == 1.0


def test_campaign_resumes_from_artifacts(tmp_path):
    first = Campaign(_config(tmp_path), llm=ToyCreator(), policy=ScriptedLLM(toy_policy)).run()
    again = ToyCreator()
    second = Campaign(_config(tmp_path), llm=again, policy=NoPolicy()).run()
    assert again.calls == [] and second == first


def test_campaign_takes_pre_collected_episodes(tmp_path, reservoir):
    path = tmp_path / "r0.jsonl"
    domain = load("toy")
    config = _config(tmp_path / "out", episodes=(str(path),), holdout_tasks=4)
    rollout.run(domain, domain.load_tasks(config.tasks_path, limit=4), ScriptedLLM(toy_policy), TIGHT, path,
                group_size=2, workers=1)
    Campaign(config, llm=ToyCreator(), policy=ScriptedLLM(toy_policy)).run()
    info = json.loads((tmp_path / "out" / "rollout.json").read_text())
    assert info["episode_files"] == [str(path)] and info["episodes"] == 8
    assert not (tmp_path / "out" / "rollout").exists()


def test_emit_keeps_existing_triggers(tmp_path):
    path = tmp_path / "program.json"
    trigger = Trigger(name="c", code="def check(ep):\n    return False\n", guidance="Check your last action.")
    write_program(path, Program(harness="toy", triggers=[trigger]))
    emit.run("toy", {"name": "digest", "code": DIGEST_PROGRAM}, path)
    program = load_program(path)
    assert program.compression["name"] == "digest" and len(program.triggers) == 1
    emit.run("toy", None, path)
    assert load_program(path).compression is None and len(load_program(path).triggers) == 1


def test_cli_flags():
    args = parse_args(["--domain", "toy", "--out", "o", "--tasks", "t", "--holdout", "h", "--serving-limit", "4096"])
    assert (args.group_size, args.candidates, args.rounds) == (4, 3, 3)
    for missing in ("--tasks", "--holdout"):
        flags = {"--tasks": "t", "--holdout": "h"}
        del flags[missing]
        with pytest.raises(SystemExit):
            parse_args(["--domain", "toy", "--out", "o", "--serving-limit", "4096",
                        *[x for kv in flags.items() for x in kv]])


@pytest.mark.parametrize("holdout", ["train", "eval", "synthetic"])
def test_campaign_refuses_a_holdout_that_is_not_held_out_training_data(tmp_path, holdout):
    config = _config(tmp_path)
    bad = {"train": config.tasks_path, "eval": config.holdout_path, "synthetic": "synthetic"}[holdout]
    evaluation = (config.holdout_path,) if holdout == "eval" else config.eval_tasks
    with pytest.raises(HoldoutError):
        Campaign(replace(config, holdout_path=bad, eval_tasks=evaluation), llm=ToyCreator(), policy=NoPolicy())

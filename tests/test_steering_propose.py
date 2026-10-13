"""ProposeSteer: mixed-outcome groups of R_k, one hypothesis per group, program admission."""

from __future__ import annotations

import json

import pytest
from test_steering_support import (
    REOPEN_CHECK,
    REOPEN_GUIDANCE,
    proposal_reply,
    reply,
    scratch,  # noqa: F401
    toy_records,
    write_rollouts,
)

from harness_programs.llm import ScriptedLLM
from harness_programs.rollout.runner import write_jsonl
from harness_programs.steering import prompts
from harness_programs.steering.propose import Hypothesis, Proposals, propose, propose_steer, uniquify
from harness_programs.steering.reservoir import (
    OBSERVATION_CHARS,
    build_groups,
    latest_step,
    load_records,
    render_group,
    sample_groups,
)

TEXT = prompts.DomainText(oracle=True)

# -- reservoir --------------------------------------------------------------------------------


def test_load_records_skips_validation_and_unplaced_rows(scratch):  # noqa: F811
    rows = toy_records(steps=(10,), n_tasks=2)
    rows[0].rollout["validate"] = True
    rows[1].rollout.pop("group_key")
    write_jsonl(scratch / "rollouts" / "step_000010.jsonl", rows)
    write_rollouts(scratch, toy_records(steps=(20,), n_tasks=2))
    assert latest_step(scratch) == 20
    reservoir = load_records(scratch, up_to=10)
    assert reservoir.files == ["rollouts/step_000010.jsonl"]
    assert (reservoir.n_records, reservoir.n_validation, reservoir.n_no_identity) == (8, 1, 1)
    assert len(reservoir.placed) == 6
    both = load_records(scratch)
    assert len(both.files) == 2 and len(both.placed) == 14
    assert load_records(scratch, after=10).files == ["rollouts/step_000020.jsonl"]


def _groups(**kw):
    from harness_programs.steering.reservoir import Placed, Reservoir
    records = toy_records(**kw)
    placed = [Placed(r, int(r.rollout["step"]), r.task_id) for r in records]
    return build_groups(Reservoir(placed=placed, files=[]))


def test_only_mixed_outcome_groups_are_kept():
    groups = _groups(n_tasks=1, mix=("solve", "loop"))
    assert len(groups) == 1 and groups[0].worst.reward == 0.0 and groups[0].best.reward == 1.0
    assert _groups(n_tasks=2, mix=("loop", "guess")) == []   # all fail
    assert _groups(n_tasks=2, mix=("solve", "solve")) == []  # all succeed
    assert _groups(n_tasks=2, mix=("solve",)) == []


def test_sampling_is_reproducible_and_bounded():
    groups = _groups(steps=(10, 20, 30), n_tasks=8, mix=("solve", "loop"))
    picked = sample_groups(groups, max_groups=20, seed=1)
    assert len(picked) == 20 and picked == sorted(picked, key=lambda g: g.key)
    assert [g.key for g in picked] == [g.key for g in sample_groups(groups, max_groups=20, seed=1)]
    assert [g.key for g in sample_groups(groups, max_groups=20)] == \
        [g.key for g in sample_groups(groups, max_groups=20)]
    assert sample_groups(groups[:5], max_groups=20) == groups[:5]


def test_render_group_truncates_observations_and_shows_gold():
    group = _groups(n_tasks=1, mix=("solve", "loop"))[0]
    text = render_group(group)
    assert "HIGHER-REWARD EPISODE" in text and "LOWER-REWARD EPISODE" in text
    assert "GOLD ANSWER" in text and json.dumps(group.gold, sort_keys=True) in text
    assert "...[truncated]" in text
    assert "it ran out of turns (8 of 8)" in text
    longest = max(len(chunk) for chunk in text.split("  -> ")[1:])
    assert longest < OBSERVATION_CHARS + 200


# -- propose ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def group():
    return _groups(n_tasks=1, mix=("solve", "loop", "loop", "solve"))[0]


def test_propose_one_clean_reply(group):
    llm = ScriptedLLM([proposal_reply()])
    hyp = propose(llm, group, text=TEXT)
    assert hyp is not None and hyp.name == "reopened_file" and hyp.group == group.key
    assert hyp.insight.startswith("the agent re-opens a file")
    assert hyp.code == REOPEN_CHECK and hyp.guidance == REOPEN_GUIDANCE and hyp.fault
    assert set(hyp.gold_strings) == {group.gold["file"], group.gold["word"]}
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert system["content"] == prompts.system_prompt(TEXT) and "HIGHER-REWARD" in user["content"]
    trigger = hyp.to_trigger()
    assert (trigger.name, trigger.code, trigger.fault) == (hyp.name, hyp.code, hyp.fault)
    assert Hypothesis.from_dict(hyp.to_dict()) == Hypothesis(**{**hyp.to_dict(), "gold_strings": []})


def test_the_fault_is_optional(group):
    hyp = propose(ScriptedLLM([proposal_reply(fault="")]), group, text=TEXT)
    assert hyp is not None and hyp.fault == ""


def test_propose_repairs_against_the_screens(group):
    llm = ScriptedLLM([proposal_reply(check="def check(ep):\n    return True\n"), proposal_reply()])
    cand = propose(llm, group, text=TEXT, revisions=2)
    assert cand is not None and cand.code == REOPEN_CHECK
    revise = llm.calls[1][-1]["content"]
    assert revise.startswith("Your trigger `reopened_file` was rejected by program admission")
    assert "reads nothing" in revise


def test_propose_repairs_a_gold_leak(group):
    leaky = f"If you have opened every file, answer {group.gold['file']} rather than opening one again."
    llm = ScriptedLLM([proposal_reply(guidance=leaky), proposal_reply()])
    assert propose(llm, group, text=TEXT).guidance == REOPEN_GUIDANCE
    assert "gold answer" in llm.calls[1][-1]["content"]


def test_propose_gives_up_after_the_revision_budget(group):
    bad = proposal_reply(guidance="Do better.")
    llm = ScriptedLLM([bad, bad, bad])
    assert propose(llm, group, text=TEXT, revisions=2) is None
    assert len(llm.calls) == 3


def test_propose_skip_and_unparseable_yield_nothing(group):
    assert propose(ScriptedLLM([reply({"skip": "no behaviour differs"})]), group, text=TEXT) is None
    assert propose(ScriptedLLM(["not json at all"]), group, text=TEXT) is None
    assert propose(ScriptedLLM([reply({"name": "x", "check": REOPEN_CHECK})]), group, text=TEXT) is None
    no_insight = json.loads(proposal_reply())
    no_insight.pop("insight")
    assert propose(ScriptedLLM([reply(no_insight)] * 3), group, text=TEXT) is None


def test_uniquify_suffixes_every_collision():
    def hyp(name, group):
        return Hypothesis(name=name, insight="", code="", guidance="", group=group)

    out = uniquify([hyp("a", "g1"), hyp("a", "g2"), hyp("b", "g3"), hyp("c", "g4")], taken=["c"])
    assert [h.name for h in out] == ["a__g1", "a__g2", "b", "c__g4"]


def test_propose_steer_keeps_admitted_and_distinct_hypotheses():
    groups = _groups(n_tasks=4, mix=("solve", "loop"))
    other = proposal_reply("other", check=REOPEN_CHECK.replace("!=", "> 1 +"))
    replies = iter([proposal_reply(), reply({"skip": "nothing"}), proposal_reply(), other])
    llm = ScriptedLLM(lambda messages: next(replies))
    out = propose_steer(llm, groups, text=TEXT, workers=1)
    assert out.groups == [g.key for g in groups]
    assert [h.name for h in out.hypotheses] == [f"reopened_file__{groups[0].key}", "other"]
    assert out.duplicates == {f"reopened_file__{groups[2].key}":
                              f"its check is byte-identical to reopened_file__{groups[0].key}'s"}
    assert Proposals.from_dict(json.loads(json.dumps(out.to_dict()))).groups == out.groups


def test_propose_steer_drops_copies_of_existing_triggers():
    groups = _groups(n_tasks=1, mix=("solve", "loop"))
    out = propose_steer(ScriptedLLM([proposal_reply()]), groups, text=TEXT, workers=1,
                        existing={"reopened_file": REOPEN_CHECK})
    assert out.hypotheses == []
    assert out.duplicates == {f"reopened_file__{groups[0].key}":
                              "its check is byte-identical to reopened_file's"}

"""A whole round on toy-domain rollouts: propose -> replay -> admit -> gate, scripted end to end."""

from __future__ import annotations

import json

import pytest
from test_steering_support import (
    REOPEN_FAULT,
    REOPEN_GUIDANCE,
    deaf_policy,
    gate_policy,
    proposal_reply,
    scratch,  # noqa: F401
    toy_gate,
    toy_records,
    write_rollouts,
)

from harness_programs.llm import ScriptedLLM
from harness_programs.domains.toy import write_split
from harness_programs.rollout.runner import write_jsonl
from harness_programs.splits import HoldoutError
from harness_programs.serving.program import load_program
from harness_programs.steering import prompts
from harness_programs.steering.__main__ import config_from, parse_args
from harness_programs.steering.gate import GateError
from harness_programs.steering.install import read_ledger
from harness_programs.steering.rounds import RoundConfig, completed_rounds, run_round
from harness_programs.steering.supervisor import due_step, supervise

ARTIFACTS = ("proposals.json", "replay.json", "admit.json", "gate.json", "round.json")


class Creator:
    """Answers as the proposer or as the judge, by system prompt, and records which."""

    def __init__(self, decision: str = "INTERVENE") -> None:
        self.decision = decision
        self.roles: list[str] = []
        self.llm = ScriptedLLM(self.reply)

    def reply(self, messages):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system == prompts.ADMIT_SYSTEM:
            self.roles.append("admit")
            return f"DECISION: {self.decision}\nRATIONALE: it persists.\nKEY EVIDENCE: half the episodes"
        self.roles.append("propose")
        return proposal_reply("reopened_file")


def _config(run_dir, **kw):
    kw.setdefault("gate", toy_gate(run_dir))
    kw.setdefault("total_steps", 100)
    return RoundConfig(run_dir=run_dir, program=run_dir / "program.json", domain="toy", workers=1,
                       **kw)


def _artifact(run_dir, step, name):
    return json.loads((run_dir / "steering" / f"round_{step:06d}" / name).read_text())


@pytest.fixture
def run_dir(scratch):  # noqa: F811
    write_rollouts(scratch, toy_records(steps=(10,)))
    return scratch


def test_a_round_installs_admitted_triggers_that_win_the_gate(run_dir):
    creator, policy = Creator(), ScriptedLLM(gate_policy)
    cfg = _config(run_dir)
    summary = run_round(cfg, 10, creator.llm, policy)
    assert (summary["groups"], summary["proposed"]) == (4, 1)
    assert summary["status"] == "installed" and summary["installed"] == summary["admitted"]
    assert creator.roles == ["propose"] * 4 + ["admit"]
    for name in ARTIFACTS:
        assert (run_dir / "steering" / "round_000010" / name).exists(), name

    proposals = _artifact(run_dir, 10, "proposals.json")
    assert len(proposals["duplicates"]) == 3  # four groups, one byte-identical check
    [name] = [h["name"] for h in proposals["hypotheses"]]
    assert name.startswith("reopened_file__step10-")
    assert proposals["hypotheses"][0]["insight"]
    evidence = _artifact(run_dir, 10, "replay.json")["evidence"][name]
    assert evidence["series"][0]["step"] == 10 and "forecast" not in evidence
    admission = _artifact(run_dir, 10, "admit.json")["admissions"][0]
    assert admission["decision"] == "INTERVENE" and "FAILURE RATE PER TRAINING STEP" in admission["report"]
    assert "FORECAST" not in admission["report"]
    gate = _artifact(run_dir, 10, "gate.json")
    assert gate["accepted"] and gate["candidate"] > gate["base"] and gate["triggers"] == [name]

    program = load_program(cfg.program)
    assert [t.name for t in program.triggers] == [name]
    assert program.triggers[0].guidance == REOPEN_GUIDANCE
    [row] = read_ledger(cfg.program)
    assert row["step"] == 10 and row["faults"] == {name: REOPEN_FAULT.strip()}
    assert completed_rounds(cfg) == [10]

    calls, policy_calls = len(creator.llm.calls), len(policy.calls)
    assert run_round(cfg, 10, creator.llm, policy) == summary
    assert len(creator.llm.calls) == calls and len(policy.calls) == policy_calls


def test_a_tie_at_the_gate_installs_nothing(run_dir):
    cfg = _config(run_dir)
    summary = run_round(cfg, 10, Creator().llm, ScriptedLLM(deaf_policy))
    gate = _artifact(run_dir, 10, "gate.json")
    assert gate["base"] == gate["candidate"] and not gate["accepted"]
    assert summary["status"] == "unchanged" and summary["installed"] == []
    assert not cfg.program.exists()


def test_a_stopped_round_resumes_without_repeating_model_calls(run_dir):
    creator = Creator()
    cfg = _config(run_dir)
    with pytest.raises(GateError):
        run_round(cfg, 10, creator.llm, policy=None)
    round_dir = run_dir / "steering" / "round_000010"
    assert (round_dir / "admit.json").exists() and not (round_dir / "round.json").exists()
    calls = len(creator.llm.calls)
    summary = run_round(cfg, 10, creator.llm, ScriptedLLM(gate_policy))
    assert len(creator.llm.calls) == calls and summary["status"] == "installed"


def test_monitor_holds_the_trigger_and_the_next_round_judges_it_again(run_dir):
    cfg = _config(run_dir)
    first = run_round(cfg, 10, Creator(decision="MONITOR").llm, ScriptedLLM(gate_policy))
    assert first["admitted"] == [] and first["installed"] == [] and first["gate"] is None
    assert not cfg.program.exists()
    [held] = _artifact(run_dir, 10, "admit.json")["held"]

    write_rollouts(run_dir, toy_records(steps=(20,)))
    later = run_round(cfg, 20, Creator().llm, ScriptedLLM(gate_policy))
    assert later["held_in"] == [held["name"]] and later["proposed"] == 0
    duplicates = _artifact(run_dir, 20, "proposals.json")["duplicates"]
    assert all(held["name"] in why for why in duplicates.values())
    series = _artifact(run_dir, 20, "replay.json")["evidence"][held["name"]]["series"]
    assert [p["step"] for p in series] == [10, 20]
    assert later["installed"] == [held["name"]]


def test_a_copy_of_an_installed_trigger_is_dropped_at_propose(run_dir):
    cfg = _config(run_dir)
    [installed] = run_round(cfg, 10, Creator().llm, ScriptedLLM(gate_policy))["installed"]
    write_rollouts(run_dir, toy_records(steps=(20,)))
    creator = Creator()
    summary = run_round(cfg, 20, creator.llm, ScriptedLLM(gate_policy))
    duplicates = _artifact(run_dir, 20, "proposals.json")["duplicates"]
    assert len(duplicates) == 4 and all(installed in why for why in duplicates.values())
    replay = _artifact(run_dir, 20, "replay.json")
    assert replay["installed"] == [installed] and list(replay["evidence"]) == [installed]
    assert summary["admitted"] == [] and "admit" not in creator.roles
    assert [t.name for t in load_program(cfg.program).triggers] == [installed]


def test_admit_all_skips_the_judge_but_not_the_gate(run_dir):
    cfg = _config(run_dir, admit_all=True)
    creator = Creator(decision="CONTINUE")
    summary = run_round(cfg, 10, creator.llm, ScriptedLLM(gate_policy))
    assert "admit" not in creator.roles and summary["installed"] == summary["admitted"] != []
    assert _artifact(run_dir, 10, "gate.json")["accepted"]


def test_continue_leaves_the_failure_to_the_policy(run_dir):
    summary = run_round(_config(run_dir), 10, Creator(decision="CONTINUE").llm, None)
    assert summary["admitted"] == [] and summary["gate"] is None
    assert _artifact(run_dir, 10, "admit.json")["dropped"] != []


def test_a_round_without_mixed_groups_calls_nothing(scratch):  # noqa: F811
    rows = toy_records(steps=(10,), mix=("loop", "loop"))
    write_jsonl(scratch / "rollouts" / "step_000010.jsonl", rows)
    creator = Creator()
    summary = run_round(_config(scratch), 10, creator.llm, None)
    assert summary["groups"] == 0 and summary["status"] == "unchanged" and not creator.llm.calls


def test_supervisor_runs_due_rounds_only(run_dir):
    assert due_step(9, 10, set()) is None
    assert due_step(27, 10, set()) == 20
    assert due_step(20, 10, {20}) is None
    write_rollouts(run_dir, toy_records(steps=(13,)))
    sleeps: list[float] = []
    state = supervise(_config(run_dir), Creator().llm, ScriptedLLM(gate_policy), every=10,
                      poll_seconds=0.5, max_polls=3, sleep=sleeps.append)
    assert state.rounds == [10] and not state.errors and sleeps == [0.5, 0.5]


def test_supervisor_stops_after_the_round_at_total_steps(run_dir):
    sleeps: list[float] = []
    state = supervise(_config(run_dir, total_steps=10), Creator().llm, ScriptedLLM(gate_policy),
                      every=10, max_polls=5, sleep=sleeps.append)
    assert state.rounds == [10] and sleeps == []


def test_supervisor_survives_a_failing_round(run_dir):
    state = supervise(_config(run_dir), Creator().llm, None, every=10, max_polls=2,
                      sleep=lambda s: None)
    assert state.rounds == [] and "GateError" in state.errors[10]


def test_cli_builds_the_round_config(scratch):  # noqa: F811
    train, holdout = (str(p) for p in write_split(scratch))
    args = parse_args(["round", "--run-dir", "runs/x", "--program", "runs/x/program.json",
                       "--domain", "toy", "--total-steps", "200", "--tasks", train, "--holdout", holdout,
                       "--eval-tasks", "synthetic", "--holdout-tasks", "16", "--step", "30"])
    cfg = config_from(args)
    assert cfg.domain == "toy" and cfg.total_steps == 200 and args.step == 30
    assert cfg.gate.holdout == holdout and cfg.gate.holdout_tasks == 16
    args = parse_args(["supervise", "--run-dir", "r", "--program", "p.json", "--domain", "toy",
                       "--total-steps", "50", "--tasks", train, "--holdout", holdout, "--admit-all",
                       "--round-every", "5"])
    cfg = config_from(args)
    assert cfg.admit_all and args.round_every == 5 and cfg.gate.holdout_tasks is None
    for removed in ("--no-gate", "--margin", "--games"):
        with pytest.raises(SystemExit):
            parse_args(["round", "--run-dir", "r", "--program", "p", "--domain", "toy",
                        "--total-steps", "5", "--tasks", "t", "--holdout", "h", removed])
    with pytest.raises(SystemExit):
        parse_args(["round", "--run-dir", "r", "--program", "p", "--domain", "toy", "--total-steps", "5",
                    "--holdout", "h"])


@pytest.mark.parametrize("bad", ["holdout_is_train", "holdout_is_eval", "holdout_is_the_whole_train_pool"])
def test_cli_refuses_a_holdout_that_is_not_held_out_training_data(scratch, bad):  # noqa: F811
    train, holdout = (str(p) for p in write_split(scratch))
    flags = {"holdout_is_train": ["--tasks", train, "--holdout", train],
             "holdout_is_eval": ["--tasks", train, "--holdout", holdout, "--eval-tasks", holdout],
             "holdout_is_the_whole_train_pool": ["--tasks", train, "--holdout", "synthetic"]}[bad]
    args = parse_args(["round", "--run-dir", "r", "--program", "p", "--domain", "toy", "--total-steps", "5",
                       *flags])
    with pytest.raises(HoldoutError):
        config_from(args)

"""Gate: H_(k-1) against H_(k-1) plus every admitted trigger, on held-out tasks, strict win."""

from __future__ import annotations

import json

import pytest
from test_steering_support import (  # noqa: F401
    REOPEN_CHECK,
    REOPEN_FAULT,
    REOPEN_GUIDANCE,
    deaf_policy,
    gate_policy,
    scratch,
    toy,
    toy_gate,
)

from harness_programs.llm import ScriptedLLM
from harness_programs.serving.program import Program, load_program
from harness_programs.serving.steer import Trigger
from harness_programs.steering.gate import (
    HOLDOUT_SPLIT,
    GateError,
    GateVerdict,
    accept,
    decide,
    holdout_tasks,
    measure,
)
from harness_programs.steering.install import read_ledger

REOPEN = Trigger("reopen", REOPEN_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT)
ANSWER = Trigger("answer_check", "def check(ep):\n    return len(ep.steps) > 6\n",
                 "When you have opened several files, answer with the one whose text holds the word.")


@pytest.fixture
def cfg(scratch):  # noqa: F811
    return toy_gate(scratch)


def test_only_a_strict_win_is_accepted():
    assert decide(0.5, 0.51)[0]
    accepted, reason = decide(0.5, 0.5)
    assert not accepted and "does not beat" in reason
    assert not decide(0.5, 0.4)[0]


def test_the_gate_plays_held_out_training_tasks(cfg, scratch):  # noqa: F811
    domain = toy()
    tasks = holdout_tasks(domain, cfg)
    assert HOLDOUT_SPLIT == "train" and len(tasks) == 8 and tasks == domain.load_tasks(cfg.holdout)
    held = {domain.task_key(t) for t in tasks}
    assert held <= {domain.task_key(t) for t in domain.load_tasks("synthetic", split="train")}
    assert held.isdisjoint(domain.task_key(t) for t in domain.load_tasks(str(scratch / "split" / "train_policy.jsonl")))
    assert held.isdisjoint(domain.task_key(t) for t in domain.load_tasks("synthetic", split="val"))


def test_both_arms_play_the_holdout_tasks_with_all_admitted_triggers(cfg, scratch):  # noqa: F811
    incumbent = Program(harness="toy", triggers=[Trigger("served_one", "def check(ep):\n    return False\n",
                                                         "Read the listing before opening any file at all.")])
    verdict = measure(toy(), ScriptedLLM(gate_policy), incumbent, [REOPEN, ANSWER], cfg,
                      work_dir=scratch, step=30)
    assert verdict.accepted and verdict.candidate > verdict.base
    assert (verdict.tasks, verdict.step, verdict.triggers) == (8, 30, ["reopen", "answer_check"])
    arm_a = load_program(scratch / "incumbent.program.json")
    arm_b = load_program(scratch / "candidate.program.json")
    assert [t.name for t in arm_a.triggers] == ["served_one"]
    assert [t.name for t in arm_b.triggers] == ["served_one", "reopen", "answer_check"]
    assert all(not t.fault for t in arm_b.triggers)
    for arm in ("incumbent", "candidate"):
        rows = [json.loads(x) for x in (scratch / f"{arm}.rollouts.jsonl").read_text().splitlines()]
        assert len(rows) == 8 and all(r["rollout"]["arm"] == arm and r["rollout"]["validate"] for r in rows)
    assert GateVerdict.from_dict(verdict.to_dict()) == verdict


def test_a_tie_is_rejected(cfg, scratch):  # noqa: F811
    verdict = measure(toy(), ScriptedLLM(deaf_policy), Program(harness="toy"), [REOPEN], cfg,
                      work_dir=scratch, step=30)
    assert verdict.base == verdict.candidate and not verdict.accepted


def test_the_gate_needs_admitted_triggers_and_holdout_tasks(cfg, scratch):  # noqa: F811
    with pytest.raises(GateError):
        measure(toy(), ScriptedLLM(gate_policy), Program(harness="toy"), [], cfg, work_dir=scratch, step=1)


def test_accept_installs_all_together_and_is_idempotent(scratch):  # noqa: F811
    path = scratch / "program.json"
    assert accept(path, "toy", [REOPEN, ANSWER], step=30) == ["reopen", "answer_check"]
    assert accept(path, "toy", [REOPEN, ANSWER], step=30) == ["reopen", "answer_check"]
    [row] = read_ledger(path)
    assert row["added"] == ["reopen", "answer_check"]
    assert [t.name for t in load_program(path).triggers] == ["reopen", "answer_check"]

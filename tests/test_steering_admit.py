"""Admit: the judge reads each trigger's replay evidence (its failure series, from E)."""

from __future__ import annotations

import pytest
from test_steering_support import REOPEN_CHECK, REOPEN_FAULT, REOPEN_GUIDANCE, toy_records

from harness_programs.llm import ScriptedLLM
from harness_programs.serving.steer import Trigger
from harness_programs.steering import prompts
from harness_programs.steering.admit import (
    CONTINUE,
    INTERVENE,
    MONITOR,
    Admission,
    Verdict,
    arbitrate,
    judge,
    parse,
    report,
)
from harness_programs.steering.replay import StepPoint, replay_evidence
from harness_programs.steering.reservoir import Placed

REOPEN = Trigger("reopen", REOPEN_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT)
INSTALLED = Trigger("installed_one", REOPEN_CHECK.replace("opened", "seen"),
                    "Before answering, check the file you name contains the word you were given.")


@pytest.fixture(scope="module")
def evidence():
    records = toy_records(steps=(10, 20), n_tasks=2)
    placed = [Placed(r, int(r.rollout["step"]), r.task_id) for r in records]
    found, dropped = replay_evidence(placed, [REOPEN, INSTALLED], installed={"installed_one"})
    assert not dropped
    return found


def _judge(llm, evidence, **kw):
    return judge(llm, REOPEN, evidence["reopen"], installed=[INSTALLED],
                 installed_evidence={"installed_one": evidence["installed_one"]}, step=20, **kw)


def test_parse_valid_decisions():
    v = parse("DECISION: intervene\nRATIONALE: rising.\nKEY EVIDENCE: 0.5 at step 20")
    assert v.decision == INTERVENE and v.rationale == "rising." and v.evidence == "0.5 at step 20"
    assert v.resolved
    assert parse("DECISION: MONITOR\nRATIONALE: x").decision == MONITOR
    assert parse("**DECISION: CONTINUE**").decision == CONTINUE


@pytest.mark.parametrize("reply", ["", "I think we should intervene.", "DECISION: ESCALATE",
                                   "DECISION:\nRATIONALE: none"])
def test_unparseable_is_no_decision_never_a_default(reply):
    v = parse(reply)
    assert v.decision is None and not v.resolved and v.error


def test_report_shows_series_guidance_and_installed_set(evidence):
    text = report(REOPEN, evidence["reopen"], [INSTALLED], {"installed_one": evidence["installed_one"]})
    assert "TRIGGER: reopen" in text and REOPEN_GUIDANCE in text
    assert text.count("0.500") >= 2  # half the episodes loop, at steps 10 and 20
    assert "FORECAST" not in text
    assert "INSTALLED TRIGGERS:" in text and "installed_one (current failure rate" in text
    assert "(none)" in report(REOPEN, evidence["reopen"], [], {})


def test_the_report_reads_the_series_from_the_evidence(evidence):
    ev = evidence["reopen"]
    planted = [StepPoint(step=999, episodes=1000, faulted=123, turns=1, fires=1)]
    text = report(REOPEN, type(ev)(ev.trigger, planted, ev.fires, ev.turns), [], {})
    assert "       999       123      1000   0.123" in text


def test_judge_asks_with_the_admit_prompt(evidence):
    llm = ScriptedLLM(["DECISION: INTERVENE\nRATIONALE: persistent.\nKEY EVIDENCE: half the episodes"])
    a = _judge(llm, evidence)
    assert a.decision == INTERVENE
    system, user = llm.calls[0]
    assert system["content"] == prompts.ADMIT_SYSTEM and "TRIGGER: reopen" in user["content"]
    assert "forecast" not in (system["content"] + user["content"]).lower()
    assert Admission.from_dict(a.to_dict()).decision == INTERVENE


def test_judge_failures_are_unresolved(evidence):
    def broken(messages):
        raise ConnectionError("endpoint down")

    assert _judge(ScriptedLLM(broken), evidence).decision is None
    assert _judge(ScriptedLLM(["no idea"]), evidence).decision is None
    assert _judge(None, evidence).decision is None


def test_admit_all_skips_the_judge(evidence):
    llm = ScriptedLLM(["DECISION: CONTINUE"])
    assert _judge(llm, evidence, admit_all=True).decision == INTERVENE and not llm.calls


def _verdict(name, decision):
    return Admission(name, 20, Verdict(decision), "")


def test_arbitrate_routes_each_decision():
    ruling = arbitrate([_verdict("a", INTERVENE), _verdict("b", MONITOR), _verdict("c", CONTINUE),
                        _verdict("d", None)])
    assert ruling.admitted == ["a"]
    assert ruling.held == ["b", "d"]
    assert ruling.dropped == ["c"]
    assert ruling.unresolved == ["d"]

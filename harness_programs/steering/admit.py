"""Admit (Algorithm 1, line 8): P <- Admit(P, H_{k-1}, E).

A judge reads each trigger's replay evidence -- its per-step failure series, taken from E and
never recomputed -- together with its guidance and the installed set (:func:`report`), and
returns one verdict:

* ``INTERVENE`` -- the failure holds steady or rises: the trigger and its guidance go to the gate.
* ``MONITOR`` -- the failure looks real but is not settled: the trigger is kept and replayed again
  next round, and its guidance is not shown.
* ``CONTINUE`` -- the failure keeps falling: it is left for the policy to learn.

A reply with no parseable decision is unresolved and held like ``MONITOR``. The ``admit_all``
ablation skips the judge and records ``INTERVENE`` for every trigger.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..llm import ChatModel, complete
from ..serving.steer import Trigger
from . import prompts
from .replay import Evidence

logger = logging.getLogger(__name__)

INTERVENE, MONITOR, CONTINUE = "INTERVENE", "MONITOR", "CONTINUE"
DECISIONS = (INTERVENE, MONITOR, CONTINUE)
ADMIT_ALL_RATIONALE = "admit-all ablation: no judge was asked; every trigger is admitted."

_DECISION = re.compile(r"DECISION:\s*([A-Za-z]+)")
_RATIONALE = re.compile(r"RATIONALE:\s*(.*?)(?=KEY EVIDENCE:|$)", re.S)
_EVIDENCE = re.compile(r"KEY EVIDENCE:\s*(.*)$", re.S)


@dataclass
class Verdict:
    """The judge's reply. ``decision`` is ``None`` when the reply could not be read."""

    decision: str | None
    rationale: str = ""
    evidence: str = ""
    raw: str = ""
    error: str = ""

    @property
    def resolved(self) -> bool:
        return self.decision in DECISIONS


def parse(reply: str) -> Verdict:
    """Read a judge reply. An unknown or missing decision is unresolved, never a default."""
    text = reply or ""
    found = _DECISION.search(text)
    rationale = _RATIONALE.search(text)
    evidence = _EVIDENCE.search(text)
    parts = {"rationale": rationale.group(1).strip() if rationale else "",
             "evidence": evidence.group(1).strip() if evidence else "", "raw": text}
    decision = found.group(1).upper() if found else ""
    if decision not in DECISIONS:
        why = f"unknown decision {decision!r}" if decision else "no DECISION line"
        return Verdict(None, error=why, **parts)
    return Verdict(decision, **parts)


def report(trigger: Trigger, evidence: Evidence, installed: Sequence[Trigger],
           installed_evidence: Mapping[str, Evidence]) -> str:
    """The text the judge reads about one trigger."""
    lines = [f"TRIGGER: {trigger.name}", f"GUIDANCE IT WOULD SERVE: {trigger.guidance}", "",
             "FAILURE RATE PER TRAINING STEP (episodes the failure occurred in / episodes replayed):",
             f"  {'step':>8}  {'faulted':>8}  {'episodes':>8}  {'rate':>6}"]
    lines += [f"  {p.step:>8}  {p.faulted:>8}  {p.episodes:>8}  {p.fault_rate:>6.3f}"
              for p in evidence.series]
    lines += ["", f"FIRES: on {evidence.fires} of {evidence.turns} turns ({evidence.fire_rate:.1%})",
              "", "INSTALLED TRIGGERS:"]
    for other in installed:
        rate = installed_evidence[other.name].current_rate if other.name in installed_evidence else None
        shown = "not replayed" if rate is None else f"current failure rate {rate:.3f}"
        lines.append(f"  - {other.name} ({shown}): {other.guidance}")
    if not installed:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


@dataclass
class Admission:
    """The verdict on one trigger and the report it was given on."""

    trigger: str
    step: int
    verdict: Verdict
    report: str

    @property
    def decision(self) -> str | None:
        return self.verdict.decision

    def to_dict(self) -> dict[str, Any]:
        return {"trigger": self.trigger, "step": self.step, "decision": self.decision,
                "verdict": asdict(self.verdict), "report": self.report}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Admission:
        return cls(trigger=row["trigger"], step=int(row["step"]), verdict=Verdict(**row["verdict"]),
                   report=row.get("report", ""))


def judge(llm: ChatModel | None, trigger: Trigger, evidence: Evidence, *,
          installed: Sequence[Trigger], installed_evidence: Mapping[str, Evidence], step: int,
          admit_all: bool = False) -> Admission:
    """One verdict. Transport and parse failures are unresolved, not decisions."""
    text = report(trigger, evidence, installed, installed_evidence)
    if admit_all:
        verdict = Verdict(INTERVENE, rationale=ADMIT_ALL_RATIONALE)
    elif llm is None:
        verdict = Verdict(None, error="no judge model configured")
    else:
        try:
            verdict = parse(complete(llm, text, system=prompts.ADMIT_SYSTEM))
        except Exception as exc:  # noqa: BLE001 -- a failed call is unresolved, not fatal
            verdict = Verdict(None, error=f"{type(exc).__name__}: {exc}")
    if not verdict.resolved:
        logger.info("admit %s: unresolved (%s)", trigger.name, verdict.error)
    return Admission(trigger=trigger.name, step=step, verdict=verdict, report=text)


@dataclass
class Ruling:
    """What the round does with its verdicts."""

    admitted: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


def arbitrate(admissions: Sequence[Admission]) -> Ruling:
    """INTERVENE -> admitted; MONITOR or unresolved -> held; CONTINUE -> dropped."""
    out = Ruling()
    for a in admissions:
        if a.decision == INTERVENE:
            out.admitted.append(a.trigger)
        elif a.decision == CONTINUE:
            out.dropped.append(a.trigger)
        else:
            out.held.append(a.trigger)
            if a.decision is None:
                out.unresolved.append(a.trigger)
    return out

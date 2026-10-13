"""A steering round: the steer update of Algorithm 1 (lines 6-9), written to disk stage by stage.

At training step k, after UpdatePolicy (line 5, the trainer) has produced pi_k and R_k,
:meth:`Round.run` executes

* line 6, ``P <- ProposeSteer(R_k)`` -- :meth:`Round.propose` -> ``proposals.json``
* line 7, ``E <- Replay(P, H_{k-1}, R_{0:k})`` -- :meth:`Round.replay` -> ``replay.json``
* line 8, ``P <- Admit(P, H_{k-1}, E)`` -- :meth:`Round.admit` -> ``admit.json``
* line 9, ``H_k <- Gate(pi_k, H_{k-1}, P, D_hold)`` -- :meth:`Round.gate` -> ``gate.json``

R_k is every rollout dumped after the previous completed round, up to step k; R_{0:k} is every
dump up to step k, including ``step_0.jsonl`` (the stage-1 rollouts R_0) when present. H_{k-1}
is the served program document. Triggers the judge held under MONITOR in the previous round
join P and are replayed and judged again.

Everything lives under ``<run_dir>/steering/round_<step>/``. A stage whose artifact exists is
loaded, not recomputed, so a round that stopped resumes without repeating a model call;
``round.json`` is written last and marks the round complete.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domains import load as load_domain
from ..domains.base import Domain
from ..llm import ChatModel
from ..serving.program import Program, load_program
from . import prompts
from .admit import arbitrate, judge
from .gate import GateConfig, GateError, GateVerdict, accept, measure
from .install import InstallError, installed_triggers
from .propose import Hypothesis, Proposals, propose_steer
from .replay import Evidence, replay_evidence
from .reservoir import MAX_GROUPS, Group, build_groups, load_records, sample_groups

logger = logging.getLogger(__name__)

STEERING_DIR = "steering"
ROUND_PREFIX = "round_"
PROPOSALS, REPLAY, ADMIT, GATE = "proposals.json", "replay.json", "admit.json", "gate.json"
DONE_FILE = "round.json"


@dataclass
class RoundConfig:
    """Everything a round needs besides its step and its models.

    ``total_steps`` is the last training step: the supervisor's last round runs there.
    """

    run_dir: Path
    program: Path
    domain: str
    total_steps: int
    gate: GateConfig
    max_groups: int = MAX_GROUPS
    seed: int | None = None
    admit_all: bool = False
    workers: int = 4
    revisions: int | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.program = Path(self.program)

    @property
    def steering_dir(self) -> Path:
        return self.run_dir / STEERING_DIR

    def round_dir(self, step: int) -> Path:
        return self.steering_dir / f"{ROUND_PREFIX}{step:06d}"


def write_json(path: Path, obj: Any) -> None:
    """Atomic: a reader (or a resumed round) sees the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(obj, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def completed_rounds(cfg: RoundConfig) -> list[int]:
    """Steps of every round that finished, oldest first."""
    if not cfg.steering_dir.is_dir():
        return []
    out = []
    for d in cfg.steering_dir.iterdir():
        if d.name.startswith(ROUND_PREFIX) and (d / DONE_FILE).exists():
            try:
                out.append(int(d.name[len(ROUND_PREFIX):]))
            except ValueError:
                continue
    return sorted(out)


def held_from(cfg: RoundConfig, previous: int | None) -> list[Hypothesis]:
    """The triggers the previous round held under MONITOR."""
    if previous is None or not (cfg.round_dir(previous) / ADMIT).exists():
        return []
    return [Hypothesis.from_dict(h) for h in read_json(cfg.round_dir(previous) / ADMIT).get("held", [])]


def domain_text(domain: Domain, groups: Sequence[Group]) -> prompts.DomainText:
    """The domain-specific prompt text: optional ``steering_reward`` / ``steering_scale``
    attributes on the domain, and whether the episodes carry a gold view."""
    gold = next((g.gold for g in groups if g.gold is not None), None)
    return prompts.DomainText(
        reward=getattr(domain, "steering_reward", None) or prompts.DEFAULT_REWARD,
        scale=getattr(domain, "steering_scale", None) or prompts.DEFAULT_SCALE,
        oracle=gold is not None, gold_shape=prompts.describe_gold(gold))


class Round:
    """One round at training step ``step``; each stage loads its artifact or computes it."""

    def __init__(self, cfg: RoundConfig, step: int, creator: ChatModel,
                 policy: ChatModel | None = None) -> None:
        self.cfg = cfg
        self.step = step
        self.creator = creator
        self.policy = policy
        self.dir = cfg.round_dir(step)
        self.domain = load_domain(cfg.domain)
        earlier = [s for s in completed_rounds(cfg) if s < step]
        self.previous = earlier[-1] if earlier else None
        self.installed = installed_triggers(cfg.program)
        self.held = held_from(cfg, self.previous)
        self.install_error = ""

    def run(self) -> dict[str, Any]:
        proposals = self.propose()
        evidence = self.replay(proposals)
        admitted = self.admit(proposals, evidence)
        verdict, installed = self.gate(admitted)
        summary = {
            "step": self.step, "status": "installed" if installed else "unchanged",
            "groups": len(proposals.groups), "proposed": len(proposals.hypotheses),
            "held_in": [h.name for h in self.held], "replayed": sorted(evidence),
            "admitted": [h.name for h in admitted],
            "gate": verdict.to_dict() if verdict else None, "installed": installed,
            "install_error": self.install_error,
        }
        write_json(self.dir / DONE_FILE, summary)
        return summary

    def _artifact(self, name: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.dir / name
        if path.exists():
            return read_json(path)
        row = {"step": self.step, **compute()}
        write_json(path, row)
        return row

    def candidates(self, proposals: Proposals) -> list[Hypothesis]:
        """P for Replay and Admit: this round's hypotheses and those held under MONITOR."""
        return [*self.held, *proposals.hypotheses]

    def pending(self, proposals: Proposals) -> list[Hypothesis]:
        """The candidates not already serving."""
        serving = {t.name for t in self.installed}
        return [h for h in self.candidates(proposals) if h.name not in serving]

    # Line 6 ---------------------------------------------------------------------------------

    def propose(self) -> Proposals:
        """P <- ProposeSteer(R_k)."""
        return Proposals.from_dict(self._artifact(PROPOSALS, self._propose))

    def _propose(self) -> dict[str, Any]:
        newest = load_records(self.cfg.run_dir, up_to=self.step, after=self.previous or 0)
        groups = sample_groups(build_groups(newest), max_groups=self.cfg.max_groups,
                               seed=self.cfg.seed)
        proposals = Proposals([], [])
        if groups:
            existing = {t.name: t.code for t in self.installed} | {h.name: h.code for h in self.held}
            proposals = propose_steer(self.creator, groups, text=domain_text(self.domain, groups),
                                      existing=existing, workers=self.cfg.workers,
                                      revisions=self.cfg.revisions)
        return {"records": newest.summary(), **proposals.to_dict()}

    # Line 7 ---------------------------------------------------------------------------------

    def replay(self, proposals: Proposals) -> dict[str, Evidence]:
        """E <- Replay(P, H_{k-1}, R_{0:k})."""
        row = self._artifact(REPLAY, lambda: self._replay(proposals))
        return {name: Evidence.from_dict(e) for name, e in row["evidence"].items()}

    def _replay(self, proposals: Proposals) -> dict[str, Any]:
        everything = load_records(self.cfg.run_dir, up_to=self.step)
        triggers = [*self.installed, *(h.to_trigger() for h in self.pending(proposals))]
        evidence, dropped = replay_evidence(everything.placed, triggers, installed={t.name for t in self.installed})
        return {"records": everything.summary(),
                "installed": [t.name for t in self.installed],
                "evidence": {n: e.to_dict() for n, e in evidence.items()}, "dropped": dropped}

    # Line 8 ---------------------------------------------------------------------------------

    def admit(self, proposals: Proposals, evidence: dict[str, Evidence]) -> list[Hypothesis]:
        """P <- Admit(P, H_{k-1}, E): the triggers the judge says to INTERVENE on."""
        row = self._artifact(ADMIT, lambda: self._admit(proposals, evidence))
        by_name = {h.name: h for h in self.candidates(proposals)}
        return [by_name[name] for name in row["admitted"]]

    def _admit(self, proposals: Proposals, evidence: dict[str, Evidence]) -> dict[str, Any]:
        pool = [h for h in self.pending(proposals) if h.name in evidence]
        installed_evidence = {t.name: evidence[t.name] for t in self.installed if t.name in evidence}
        admissions = [judge(self.creator, h.to_trigger(), evidence[h.name], installed=self.installed,
                            installed_evidence=installed_evidence, step=self.step,
                            admit_all=self.cfg.admit_all)
                      for h in pool]
        ruling = arbitrate(admissions)
        by_name = {h.name: h for h in pool}
        return {"admit_all": self.cfg.admit_all, "admissions": [a.to_dict() for a in admissions],
                **ruling.to_dict(), "held": [by_name[n].to_dict() for n in ruling.held]}

    # Line 9 ---------------------------------------------------------------------------------

    def gate(self, admitted: list[Hypothesis]) -> tuple[GateVerdict | None, list[str]]:
        """H_k <- Gate(pi_k, H_{k-1}, P, D_hold); returns the verdict and what was installed."""
        if not admitted:
            return None, []
        verdict = GateVerdict.from_dict(self._artifact(GATE, lambda: self._gate(admitted)))
        if not verdict.accepted:
            return verdict, []
        try:
            return verdict, accept(self.cfg.program, self.domain.name,
                                   [h.to_trigger() for h in admitted], step=self.step)
        except InstallError as exc:
            logger.error("round %d: install refused: %s", self.step, exc)
            self.install_error = str(exc)
            return verdict, []

    def _gate(self, admitted: list[Hypothesis]) -> dict[str, Any]:
        if self.policy is None:
            raise GateError("the gate needs a policy endpoint serving the current checkpoint")
        verdict = measure(self.domain, self.policy, self.incumbent(),
                          [h.to_trigger() for h in admitted], self.cfg.gate,
                          work_dir=self.dir / "gate", step=self.step)
        return verdict.to_dict()

    def incumbent(self) -> Program:
        """H_{k-1}: the served program document."""
        path = self.cfg.program
        return load_program(path) if path.exists() else Program(harness=self.domain.name)


def run_round(cfg: RoundConfig, step: int, creator: ChatModel,
              policy: ChatModel | None = None) -> dict[str, Any]:
    """Run (or resume) the round at ``step`` and return its summary."""
    done = cfg.round_dir(step) / DONE_FILE
    if done.exists():
        return read_json(done)
    return Round(cfg, step, creator, policy).run()


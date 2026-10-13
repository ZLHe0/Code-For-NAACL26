"""Gate (Algorithm 1, line 9): H_k <- Gate(pi_k, H_{k-1}, P, D_hold).

One comparison per round, under the current policy pi_k. Arm A serves the incumbent harness
H_{k-1}; arm B serves H_{k-1} plus every trigger admitted this round, together, because guidance
interacts. Both arms play the same held-out tasks D_hold. The admitted set is accepted only if
arm B scores strictly above arm A (a tie rejects); :func:`accept` then installs it atomically
(:mod:`.install`) and the program document becomes H_k. Otherwise H_k = H_{k-1}.

The policy is any :class:`~harness_programs.llm.ChatModel`; in a training run it is an
OpenAI-compatible endpoint serving the latest checkpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..domains.base import Domain, Task
from ..llm import ChatModel
from ..rollout.runner import run_batch, write_jsonl
from ..serving.program import Program, load_program, write_program
from ..serving.serve import ContextPipeline
from ..serving.steer import Trigger
from ..serving.tokens import char_counter
from ..splits import HOLDOUT_SPLIT
from .install import install, served

logger = logging.getLogger(__name__)

DEFAULT_SERVING_LIMIT = 1_000_000


class GateError(Exception):
    """The gate is misconfigured or could not measure an arm."""


@dataclass
class GateConfig:
    """How the gate measures.

    ``holdout`` is D_hold: held-out *training* tasks, carved out of the training split and
    disjoint from the tasks the policy trains on and from every evaluation set
    (:mod:`harness_programs.splits`; the CLI checks this before a round runs).
    """

    holdout: str
    holdout_tasks: int | None = None
    serving_limit: int = DEFAULT_SERVING_LIMIT
    history_budget: int | None = None
    workers: int = 8
    temperature: float | None = None
    max_per_episode: int = 0
    max_per_trigger: int = 0


@dataclass
class GateVerdict:
    """Arm A (H_{k-1}) against arm B (H_{k-1} plus every admitted trigger)."""

    triggers: list[str]
    accepted: bool
    base: float
    candidate: float
    tasks: int
    step: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> GateVerdict:
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})


def decide(base: float, candidate: float) -> tuple[bool, str]:
    """Accept only a strict improvement; a tie rejects."""
    accepted = candidate > base
    verb = "beats" if accepted else "does not beat"
    return accepted, f"H_(k-1) + admitted scores {candidate:.4f}, {verb} H_(k-1) at {base:.4f}"


def holdout_tasks(domain: Domain, cfg: GateConfig) -> list[Task]:
    tasks = domain.load_tasks(cfg.holdout, split=HOLDOUT_SPLIT, limit=cfg.holdout_tasks)
    if not tasks:
        raise GateError(f"no held-out tasks in {cfg.holdout!r}")
    return tasks


def score_arm(domain: Domain, tasks: Sequence[Task], policy: ChatModel, program: Program, *,
              work_dir: str | Path, label: str, cfg: GateConfig, step: int) -> float:
    """Mean reward of ``policy`` on ``tasks`` while ``program`` is served."""
    work_dir = Path(work_dir)
    program_path = work_dir / f"{label}.program.json"
    write_program(program_path, program)
    pipeline = ContextPipeline(domain.name, program_path=program_path, token_counter=char_counter(),
                               serving_limit=cfg.serving_limit, history_budget=cfg.history_budget,
                               max_steps=domain.max_turns, max_per_episode=cfg.max_per_episode,
                               max_per_trigger=cfg.max_per_trigger)
    records = run_batch(domain, tasks, policy, pipeline=pipeline, workers=cfg.workers,
                        temperature=cfg.temperature,
                        # not a training rollout: the reservoir skips it
                        rollout={"step": step, "validate": True, "arm": label})
    write_jsonl(work_dir / f"{label}.rollouts.jsonl", records)
    if not records:
        raise GateError(f"arm {label} produced no episodes")
    return sum(r.reward for r in records) / len(records)


def measure(domain: Domain, policy: ChatModel, incumbent: Program, admitted: Sequence[Trigger],
         cfg: GateConfig, *, work_dir: str | Path, step: int) -> GateVerdict:
    """Play H_{k-1} and H_{k-1} plus all of ``admitted`` on the held-out tasks; strict win accepts."""
    if not admitted:
        raise GateError("the gate needs at least one admitted trigger")
    tasks = holdout_tasks(domain, cfg)
    arm_b = replace(incumbent, triggers=[*incumbent.triggers, *map(served, admitted)])
    base = score_arm(domain, tasks, policy, incumbent, work_dir=work_dir, label="incumbent",
                     cfg=cfg, step=step)
    candidate = score_arm(domain, tasks, policy, arm_b, work_dir=work_dir, label="candidate",
                          cfg=cfg, step=step)
    accepted, reason = decide(base, candidate)
    logger.info("gate at step %d: %s", step, reason)
    return GateVerdict(triggers=[t.name for t in admitted], accepted=accepted, base=base,
                       candidate=candidate, tasks=len(tasks), step=step, reason=reason)


def accept(program_path: str | Path, domain: str, admitted: Sequence[Trigger], *, step: int) -> list[str]:
    """Install the admitted triggers not yet serving, in one atomic write; returns their names.

    Idempotent, so a round that stopped between the verdict and the install resumes safely.
    """
    path = Path(program_path)
    serving = {t.name for t in load_program(path).triggers} if path.exists() else set()
    pending = [t for t in admitted if t.name not in serving]
    if pending:
        install(path, domain, pending, step=step)
    return [t.name for t in admitted]

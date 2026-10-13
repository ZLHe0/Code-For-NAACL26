"""Gate (Algorithm 1, line 3): one held-out trial per program under the frozen base policy.

``Gate(pi_0, {}, P, D_hold)`` rolls out the incumbent -- no compression -- and every program in
P once on D_hold, the held-out training tasks, through the same serving pipeline, agent loop, tools and context
limit training uses. A program's score is its mean reward. H0 is the best program in P if it
scores strictly above no compression, and no compression otherwise; it is then held fixed for
the rest of training.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..domains.base import Domain, Task
from ..llm.client import ChatModel
from ..rollout.runner import EpisodeRecord, read_jsonl, run_batch, write_jsonl
from ..serving.compress import get_state_dsl
from ..serving.program import Program, contract_sha, write_program
from .episodes import Sizing

logger = logging.getLogger(__name__)

NAIVE_NAME = "naive_baseline"


def trial(domain: Domain, compression: dict[str, Any] | None, tasks: Sequence[Task], policy: ChatModel,
          sizing: Sizing, arm_dir: Path, *, workers: int = 8) -> float:
    """Mean reward of one program over ``tasks``, one rollout each; episodes kept under ``arm_dir``."""
    episodes = arm_dir / "episodes.jsonl"
    if episodes.exists():
        records: list[EpisodeRecord] = read_jsonl(episodes)
    else:
        program_path = arm_dir / "program.json"
        write_program(program_path, Program(harness=domain.name, compression=compression, triggers=[],
                                            contract_sha=contract_sha(get_state_dsl(domain.name).CONTRACT)))
        pipeline = sizing.pipeline(domain.name, domain.max_turns, program_path)
        records = run_batch(domain, tasks, policy, pipeline=pipeline, workers=workers,
                            rollout={"phase": "gate", "arm": arm_dir.name})
        write_jsonl(episodes, records)
    return round(sum(float(r.reward or 0.0) for r in records) / max(len(records), 1), 4)


def select(scores: dict[str, float]) -> str:
    """The best program if it scores strictly above no compression, else no compression."""
    rivals = [name for name in scores if name != NAIVE_NAME]
    best = max(rivals, key=lambda name: scores[name], default=None)
    return best if best is not None and scores[best] > scores[NAIVE_NAME] else NAIVE_NAME


def run(domain: Domain, proposals: Sequence[dict[str, Any]], tasks: Sequence[Task], policy: ChatModel,
        sizing: Sizing, out_dir: str | Path, *, workers: int = 8) -> dict[str, Any]:
    """Score no compression and every program in P on ``tasks``; pick H0."""
    if not tasks:
        raise ValueError(f"{domain.name}: the gate has no held-out tasks")
    arms: dict[str, dict[str, Any] | None] = {NAIVE_NAME: None}
    arms.update({p["name"]: {"name": p["name"], "code": p["code"]} for p in proposals})
    scores = {name: trial(domain, compression, tasks, policy, sizing, Path(out_dir) / name, workers=workers)
              for name, compression in arms.items()}
    winner = select(scores)
    logger.info("gate on %d held-out tasks: %s -> %s", len(tasks), scores, winner)
    return {"tasks": len(tasks), "baseline": NAIVE_NAME, "scores": scores, "winner": winner,
            "compression": arms[winner]}

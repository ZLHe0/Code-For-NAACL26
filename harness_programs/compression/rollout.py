"""Rollout (Algorithm 1, line 1): the base policy on the training tasks, append-only context.

Each training task is rolled out ``group_size`` times with no compression program, through the
same serving pipeline and context limit training will use. The episodes, each with its reward
and grouped by task, seed the reservoir R0. Pre-collected episodes in the same JSONL format can
be passed to the campaign instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..domains.base import Domain, Task
from ..llm.client import ChatModel
from ..rollout.runner import run_batch, write_jsonl
from .episodes import OptEpisode, Sizing, by_task

logger = logging.getLogger(__name__)

DEFAULT_GROUP_SIZE = 4


def run(domain: Domain, tasks: Sequence[Task], policy: ChatModel, sizing: Sizing, path: str | Path, *,
        group_size: int = DEFAULT_GROUP_SIZE, workers: int = 8) -> list[OptEpisode]:
    """Roll out ``policy`` ``group_size`` times per task with no program; write the episodes to ``path``."""
    pipeline = sizing.pipeline(domain.name, domain.max_turns)
    records = run_batch(domain, tasks, policy, pipeline=pipeline, replicates=group_size, workers=workers,
                        rollout={"phase": "rollout"})
    write_jsonl(path, records)
    return [OptEpisode.from_record(r) for r in records]


def summary(eps: list[OptEpisode], files: Sequence[str | Path]) -> dict[str, Any]:
    """The rollout artifact: where R0 lives, and how many of its tasks have mixed outcomes."""
    groups = by_task(eps)
    rewards = [ep.outcome for ep in eps if ep.outcome is not None]
    mixed = sum(1 for group in groups.values() if len({ep.outcome for ep in group}) > 1)
    logger.info("rollout: %d episodes over %d tasks, %d with mixed outcomes", len(eps), len(groups), mixed)
    return {"episode_files": [str(f) for f in files], "episodes": len(eps), "tasks": len(groups),
            "mixed_outcome_tasks": mixed,
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None}

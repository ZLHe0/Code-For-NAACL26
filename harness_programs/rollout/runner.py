"""Offline rollouts: run a policy through a domain with the serving pipeline in the loop.

The environment always sees the full transcript; the policy sees only what the pipeline
serves. Episodes are written as JSON lines, one :class:`EpisodeRecord` per line, the same
format the RL loop dumps during training and the steering optimizer reads.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..domains.base import Domain, Task
from ..llm.client import ChatModel
from ..serving.serve import ContextPipeline, EpisodeContext

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    episode_id: str
    harness: str
    task_id: str
    status: str                       # "answered" | "max_turns" | "error"
    num_steps: int
    max_steps: int
    messages: list[dict[str, Any]]    # the full transcript, never the compressed view
    reward: float
    final_output: str = ""
    gold: Any = None
    context_budget: int | None = None
    triggers: dict[str, Any] = field(default_factory=dict)
    served: list[dict[str, Any]] = field(default_factory=list)
    rollout: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> EpisodeRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in row.items() if k in known})


def run_episode(domain: Domain, task: Task, policy: ChatModel, *, context: EpisodeContext | None = None,
                history_budget: int | None = None, max_turns: int | None = None, temperature: float | None = None,
                rollout: dict[str, Any] | None = None) -> EpisodeRecord:
    env = domain.make_env(task)
    messages = env.reset()
    limit = max_turns or domain.max_turns
    status, served_log = "max_turns", []
    for _ in range(limit):
        if context is not None:
            try:
                turn = context.serve(messages)
            except ValueError as exc:
                logger.warning("task %s: %s", task.task_id, exc)
                status = "error"
                break
            prompt = turn.messages
            served_log.append({**turn.stats.to_dict(), "fired": turn.fired})
        else:
            prompt = messages
        reply = policy.chat(prompt, tools=domain.tools, temperature=temperature)
        assistant = {"role": "assistant", "content": reply.get("content") or ""}
        if reply.get("tool_calls"):
            assistant["tool_calls"] = reply["tool_calls"]
        messages.append(assistant)
        result = env.step(assistant)
        messages.extend(result.observations)
        if result.done:
            status = "answered"
            break
    return EpisodeRecord(
        episode_id=uuid.uuid4().hex[:12], harness=domain.name, task_id=task.task_id, status=status,
        num_steps=sum(1 for m in messages if m.get("role") == "assistant"), max_steps=limit, messages=messages,
        reward=float(env.reward()), final_output=env.final_output(), gold=env.gold_view(),
        context_budget=history_budget,
        triggers={"setup": context.setup, "firings": context.firings} if context is not None else {},
        served=served_log, rollout=dict(rollout or {}),
    )


def run_batch(domain: Domain, tasks: Sequence[Task], policy: ChatModel, *, pipeline: ContextPipeline | None,
              replicates: int = 1, workers: int = 8, temperature: float | None = None,
              rollout: dict[str, Any] | None = None) -> list[EpisodeRecord]:
    """Every task ``replicates`` times; each episode gets its own :class:`EpisodeContext`."""
    jobs = [(task, r) for task in tasks for r in range(replicates)]
    budget = getattr(pipeline, "_history_budget", None)

    def one(job: tuple[Task, int]) -> EpisodeRecord:
        task, replicate = job
        context = pipeline.new_episode() if pipeline is not None else None
        meta = {**(rollout or {}), "group_key": task.task_id, "replicate": replicate}
        return run_episode(domain, task, policy, context=context, history_budget=budget,
                           temperature=temperature, rollout=meta)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, jobs))


def write_jsonl(path: str | Path, records: Iterable[EpisodeRecord | dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            row = record.to_dict() if isinstance(record, EpisodeRecord) else record
            handle.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[EpisodeRecord]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(EpisodeRecord.from_dict(json.loads(line)))
    return rows

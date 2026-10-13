"""The domain interface: tasks, a per-episode environment, a reward, and a state DSL.

A domain module defines a :class:`Domain` and a state DSL (see
:mod:`harness_programs.serving.compress`), and registers both on import.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..serving.messages import Message, Messages


@dataclass
class Task:
    """One problem instance. ``data`` holds the domain-specific fields, including the gold label."""

    task_id: str
    prompt: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """What the environment returns for one assistant turn."""

    observations: Messages
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    """One episode of one task. Not shared across episodes."""

    def reset(self) -> Messages:
        """The pinned head: system prompt and task statement (plus any initial observation)."""

    def step(self, assistant: Message) -> StepResult:
        """Execute the assistant turn's action and return the observation message(s)."""

    def reward(self) -> float:
        """The episode's scalar reward, available once the episode is done (or cut off)."""

    def final_output(self) -> str:
        """The agent's committed answer as text, ``''`` if it never answered."""

    def gold_view(self) -> Any:
        """The curated answer view an offline fault predicate reads as ``ep.gold``."""


class Domain(Protocol):
    name: str
    max_turns: int
    tools: Sequence[dict[str, Any]] | None

    def load_tasks(self, path: str, *, split: str = "train", limit: int | None = None) -> list[Task]: ...

    def task_key(self, task: Task) -> str:
        """The task's identity across files (content, not ``task_id``); see :mod:`harness_programs.splits`."""
        ...

    def make_env(self, task: Task) -> Environment: ...


_DOMAINS: dict[str, Domain] = {}


def register_domain(domain: Domain) -> None:
    _DOMAINS[domain.name] = domain


def registered() -> dict[str, Domain]:
    return dict(_DOMAINS)

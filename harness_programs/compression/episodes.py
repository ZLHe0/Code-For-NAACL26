"""The reservoir R0 that Rollout (Algorithm 1, line 1) produces, and what a program is fit for.

R0 is a list of :class:`OptEpisode` read from :class:`~harness_programs.rollout.runner.EpisodeRecord`
JSONL, each with its reward, grouped by task with :func:`by_task`. :class:`Sizing` is the
context limit training will serve under; :class:`Target` binds it to a domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..rollout.runner import EpisodeRecord, read_jsonl
from ..serving.compress import get_state_dsl
from ..serving.serve import ContextPipeline
from ..serving.tokens import TokenCounter, char_counter


@dataclass(frozen=True)
class Sizing:
    """The serving budget a program is optimized for.

    Args:
        serving_limit: the largest prompt the policy server accepts.
        history_budget: history tokens allowed beyond the pinned head; ``None`` means the whole
            transcript must fit in ``serving_limit``.
        counter: counts tokens as the policy server would. The default approximates with
            characters; pass :func:`~harness_programs.serving.tokens.chat_template_counter` to match a
            real tokenizer.
    """

    serving_limit: int
    history_budget: int | None = None
    counter: TokenCounter = field(default_factory=char_counter, compare=False, repr=False)

    @property
    def label(self) -> str:
        if self.history_budget is None:
            return f"limit={self.serving_limit}"
        return f"history={self.history_budget},limit={self.serving_limit}"

    def pipeline(self, domain: str, max_steps: int, program_path: str | Path | None = None) -> ContextPipeline:
        """The serving pipeline training uses at this sizing; ``program_path=None`` serves no program."""
        return ContextPipeline(domain, program_path=program_path, token_counter=self.counter,
                               serving_limit=self.serving_limit, history_budget=self.history_budget,
                               max_steps=max_steps)


@dataclass
class Target:
    """One domain bound to the sizing a program is being optimized for."""

    domain: str
    sizing: Sizing

    @property
    def dsl(self) -> Any:
        return get_state_dsl(self.domain)

    @property
    def contract(self) -> str:
        return self.dsl.CONTRACT

    @property
    def desc(self) -> str:
        return getattr(self.dsl, "DESC", "") or f"An agent in the {self.domain} domain."

    def hook(self, name: str, default: Any = None) -> Any:
        return getattr(self.dsl, name, default)


@dataclass
class OptEpisode:
    """One rollout as the optimizer sees it."""

    task_id: str
    messages: list[dict[str, Any]]
    outcome: float | None = None
    status: str = ""
    gold: Any = None

    @classmethod
    def from_record(cls, record: EpisodeRecord) -> OptEpisode:
        return cls(task_id=record.task_id, messages=list(record.messages), outcome=record.reward,
                   status=record.status, gold=record.gold)


def load_episodes(paths: str | Path | list[str | Path], *, harness: str | None = None) -> list[OptEpisode]:
    """Episodes from one or more JSONL files, optionally filtered to one harness."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return [OptEpisode.from_record(record) for path in paths for record in read_jsonl(path)
            if not harness or record.harness == harness]


def by_task(eps: list[OptEpisode]) -> dict[str, list[OptEpisode]]:
    """R0 grouped by task, in first-seen order."""
    groups: dict[str, list[OptEpisode]] = {}
    for ep in eps:
        groups.setdefault(ep.task_id, []).append(ep)
    return groups

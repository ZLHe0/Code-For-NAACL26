"""The serving pipeline: what the policy is shown on each turn.

Per turn, from the full transcript:

1. **Steering.** Each installed trigger's ``check`` runs on the episode prefix; the guidance texts
   that fire form one ephemeral guidance message.
2. **Compression.** The generated ``compress(state)`` rewrites what past turns cost.
3. **Budget.** Oldest turns are dropped until the prompt fits the pinned head plus the
   episode's history budget, with room reserved for the guidance message.
4. The guidance message is appended last. It is never written into the transcript.

The program document is re-read at the start of every episode, so a trigger installed while
training runs is live from the next episode on.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import BudgetedContext, BudgetStats
from .compress import build_stage1, get_state_dsl
from .messages import Messages
from .program import Program, ProgramError, check_contract, load_program
from .steer import TriggerRuntime, build_runtime
from .tokens import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ServedTurn:
    messages: Messages
    stats: BudgetStats
    fired: list[str] = field(default_factory=list)


class EpisodeContext:
    """One episode's view of the program, frozen at episode start."""

    def __init__(self, program: Program, budget: BudgetedContext, runtime: TriggerRuntime, max_steps: int) -> None:
        self.program = program
        self._budget = budget
        self._runtime = runtime
        self._max_steps = max_steps
        self.firings: list[list[str]] = []

    @property
    def setup(self) -> dict[str, Any]:
        return {"harness": self.program.harness,
                "compression": (self.program.compression or {}).get("name"),
                **self._runtime.setup}

    def serve(self, messages: Messages) -> ServedTurn:
        tail, fired = self._runtime.tail(messages, max_steps=self._max_steps)
        served, stats = self._budget(messages, tail=tail)
        self.firings.append(fired)
        return ServedTurn(messages=[*served, *tail], stats=stats, fired=fired)


class ContextPipeline:
    """Builds an :class:`EpisodeContext` per episode from the current program document.

    Args:
        domain: the harness the program is written for (selects the state DSL).
        program_path: the program document; ``None`` serves the raw transcript.
        token_counter: counts tokens as the policy server would.
        serving_limit: the largest prompt the server accepts.
        history_budget: history tokens allowed above the pinned head; ``None`` for no budget.
    """

    def __init__(self, domain: str, *, program_path: str | os.PathLike[str] | None, token_counter: TokenCounter,
                 serving_limit: int, history_budget: int | None, max_steps: int = 0,
                 trim_overflow: bool = False, max_per_episode: int = 0, max_per_trigger: int = 0) -> None:
        self.domain = domain
        self.program_path = Path(program_path) if program_path else None
        self._count = token_counter
        self._serving_limit = serving_limit
        self._history_budget = history_budget
        self._max_steps = max_steps
        self._trim_overflow = trim_overflow
        self._caps = {"max_per_episode": max_per_episode, "max_per_trigger": max_per_trigger}
        self._mtime: float | None = None
        self._built: tuple[Program, Any, TriggerRuntime] | None = None

    def _build(self) -> tuple[Program, Any, TriggerRuntime]:
        if self.program_path is None:
            return Program(harness=self.domain), None, TriggerRuntime()
        mtime = self.program_path.stat().st_mtime
        if self._built is not None and mtime == self._mtime:
            return self._built
        program = load_program(self.program_path)
        if program.harness != self.domain:
            raise ProgramError(f"program is for {program.harness!r}, serving {self.domain!r}")
        dsl = get_state_dsl(self.domain)
        check_contract(program, dsl.CONTRACT)
        stage1 = build_stage1(self.domain, program.compression_code) if program.compression_code else None
        runtime = build_runtime(program.triggers, **self._caps)
        if self._built is not None:
            logger.info("program reloaded: %d trigger(s)", len(program.triggers))
        self._built, self._mtime = (program, stage1, runtime), mtime
        return self._built

    def new_episode(self) -> EpisodeContext:
        program, stage1, runtime = self._build()
        budget = BudgetedContext(stage1, token_counter=self._count, serving_limit=self._serving_limit,
                                 history_budget=self._history_budget, trim_overflow=self._trim_overflow)
        return EpisodeContext(program, budget, runtime, self._max_steps)

"""The supervisor: alternates the steer update with UpdatePolicy (Algorithm 1, line 5).

UpdatePolicy is the trainer (:mod:`harness_programs.rl`), which keeps training under whatever program
document is served and dumps its rollouts. The supervisor runs beside it and starts a steering
round (:mod:`.rounds`) every ``ROUND_EVERY`` training steps: it polls the run directory for
rollout dumps, and whenever the newest dumped step reaches the next multiple of ``every`` that
has no completed round, it runs that round (resuming a partial one if a previous supervisor
stopped mid-round). A round that raises is logged and retried on the next poll; it never stops
the loop. The trainer is not touched: the only thing a round writes that
training reads is the program document, which serving reloads on change.

The loop ends after ``max_rounds`` completed rounds, once a round at or past ``until_step``
(default: the config's ``total_steps``) has run, or on ``KeyboardInterrupt``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..llm import ChatModel
from .reservoir import latest_step
from .rounds import RoundConfig, completed_rounds, run_round

logger = logging.getLogger(__name__)

ROUND_EVERY = 10
DEFAULT_POLL_SECONDS = 60.0


@dataclass
class SupervisorState:
    rounds: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)


def due_step(latest: int, every: int, done: set[int]) -> int | None:
    """The newest multiple of ``every`` at or below ``latest`` without a completed round.

    Only the newest is due: a supervisor that fell behind catches up on the run as it is now,
    not on stale windows.
    """
    if latest < every:
        return None
    step = latest - latest % every
    return None if step in done else step


def supervise(cfg: RoundConfig, creator: ChatModel, policy: ChatModel | None = None, *,
              every: int = ROUND_EVERY, poll_seconds: float = DEFAULT_POLL_SECONDS,
              max_rounds: int | None = None, until_step: int | None = None,
              sleep: Callable[[float], None] = time.sleep,
              max_polls: int | None = None) -> SupervisorState:
    """Watch ``cfg.run_dir`` and run rounds as training reaches them."""
    if every <= 0:
        raise ValueError("every must be positive")
    until_step = cfg.total_steps if until_step is None else until_step
    state = SupervisorState()
    polls = 0
    try:
        while True:
            polls += 1
            step = due_step(latest_step(cfg.run_dir), every, set(completed_rounds(cfg)))
            if step is not None:
                try:
                    summary = run_round(cfg, step, creator, policy)
                    state.rounds.append(step)
                    state.errors.pop(step, None)
                    logger.info("round %d done: %s", step, summary.get("status"))
                except Exception as exc:  # noqa: BLE001 -- a failed round must not stop the loop
                    logger.exception("round %d failed; will retry", step)
                    state.errors[step] = f"{type(exc).__name__}: {exc}"
            if max_rounds is not None and len(state.rounds) >= max_rounds:
                break
            if state.rounds and state.rounds[-1] >= until_step:
                break
            if max_polls is not None and polls >= max_polls:
                break
            sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("supervisor interrupted after %d round(s)", len(state.rounds))
    return state

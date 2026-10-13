"""Budget fallback: drop whole oldest turns until the served prompt fits.

The served prompt is built in two layers. Compression (:mod:`.compress`) rewrites what each
past turn costs; this module then keeps the pinned head plus the largest suffix of turns that
fits ``floor + history_budget`` tokens, where ``floor`` is the measured cost of the pinned head.
The dropped span is replaced by a one-line stub. Compressing harder therefore buys more turns
of reachable history.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .messages import Messages, is_assistant, pinned_head, turn_starts

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Messages], int]
Stage1 = Callable[[Messages], Messages]

TRUNCATION_STUB = "[earlier turns omitted]"
TRIM_MARKER = "\n[... truncated to fit the context budget]"


def keep_recent_turns(messages: Messages, k: int, stub: str = TRUNCATION_STUB) -> Messages:
    """The pinned head, a stub, and the last ``k`` turns. Identity when nothing is dropped."""
    starts = turn_starts(messages)
    if not starts or k >= len(starts):
        return list(messages)
    cut = starts[len(starts) - k]
    return [*pinned_head(messages), {"role": "user", "content": stub}, *messages[cut:]]


@dataclass
class BudgetStats:
    """What the budget stage did on one call, recorded per turn."""

    floor_tokens: int = 0
    budget: int = 0
    raw_tokens: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    turns_total: int = 0
    turns_kept: int = 0
    evicted: bool = False
    overflow: bool = False
    trimmed_chars: int = 0

    @property
    def stage1_saved(self) -> int:
        return self.raw_tokens - self.tokens_before

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class BudgetedContext:
    """Compression followed by oldest-turn eviction, against a fixed per-episode budget.

    Args:
        stage1: the compression program as ``messages -> messages`` (``None`` for none).
        token_counter: counts the tokens of a message list as the policy server would.
        serving_limit: hard ceiling; a prompt above it cannot be served faithfully.
        history_budget: tokens of history allowed above the pinned head (``None`` = no budget).
        trim_overflow: when even one turn exceeds the budget, cut the largest observation.
    """

    def __init__(self, stage1: Stage1 | None, *, token_counter: TokenCounter, serving_limit: int,
                 history_budget: int | None = None, trim_overflow: bool = False,
                 stub: str = TRUNCATION_STUB) -> None:
        self._stage1 = stage1
        self._count = token_counter
        self._serving_limit = int(serving_limit)
        self._history_budget = None if history_budget is None else int(history_budget)
        self._trim_overflow = trim_overflow
        self._stub = stub
        self._last_k: int | None = None

    def _budget(self, managed: Messages) -> tuple[int, int]:
        if self._history_budget is None:
            return self._serving_limit, 0
        head = pinned_head(managed)
        floor = self._count(head) if head else 0
        budget = floor + self._history_budget
        if budget > self._serving_limit:
            raise ValueError(
                f"history_budget={self._history_budget} is not deliverable: the pinned head costs {floor} "
                f"tokens, above the serving limit of {self._serving_limit} once added")
        return budget, floor

    def _fits(self, messages: Messages, k: int, budget: int) -> bool:
        return self._count(keep_recent_turns(messages, k, self._stub)) <= budget

    def _search(self, managed: Messages, n_turns: int, budget: int) -> int:
        """Largest ``k`` in ``[1, n_turns - 1]`` whose window fits; token count is monotone in ``k``.

        Resumes from the previous call's ``k``: an append-only transcript moves it by a step
        or two per turn, which is cheaper than a fresh binary search.
        """
        if self._last_k is not None:
            k = max(1, min(self._last_k, n_turns - 1))
            if self._fits(managed, k, budget):
                while k + 1 <= n_turns - 1 and self._fits(managed, k + 1, budget):
                    k += 1
                return k
            while k > 1:
                k -= 1
                if self._fits(managed, k, budget):
                    return k
            return 1
        lo, hi, best = 1, n_turns - 1, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._fits(managed, mid, budget):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best

    def _trim(self, served: Messages, budget: int) -> tuple[Messages, int]:
        """Cut the tail of the largest non-assistant messages until the window fits."""
        head = len(pinned_head(served))
        out, removed, done = list(served), 0, set()
        while self._count(out) > budget:
            candidates = [i for i in range(head, len(out))
                          if i not in done and not is_assistant(out[i]) and isinstance(out[i].get("content"), str)]
            if not candidates:
                break
            i = max(candidates, key=lambda j: len(out[j]["content"]))
            done.add(i)
            text = out[i]["content"]
            lo, hi, best = 0, len(text), None
            while lo <= hi:
                mid = (lo + hi) // 2
                probe = list(out)
                probe[i] = {**out[i], "content": text[:mid] + TRIM_MARKER}
                if self._count(probe) <= budget:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            new_text = "" if best is None else text[:best] + TRIM_MARKER
            removed += len(text) - (best or 0)
            out[i] = {**out[i], "content": new_text}
        return out, removed

    def __call__(self, messages: Messages, tail: Messages | None = None) -> tuple[Messages, BudgetStats]:
        """Fit ``messages`` to the budget, reserving room for an ephemeral guidance ``tail``.

        The tail itself is not appended here; the caller appends it after this returns.
        """
        managed = self._stage1(messages) if self._stage1 is not None else list(messages)
        budget, floor = self._budget(managed)
        tail_tokens = self._count(tail) if tail else 0
        budget -= tail_tokens
        n_turns = len(turn_starts(managed))
        before = self._count(managed)
        stats = BudgetStats(
            floor_tokens=floor, budget=budget,
            raw_tokens=self._count(messages) if self._stage1 is not None else before,
            tokens_before=before, tokens_after=before, turns_total=n_turns, turns_kept=n_turns,
        )
        served = managed
        if n_turns > 1 and before > budget:
            k = self._search(managed, n_turns, budget)
            served = keep_recent_turns(managed, k, self._stub)
            self._last_k = k
            stats.evicted, stats.turns_kept = True, k
        else:
            self._last_k = n_turns
        stats.tokens_after = self._count(served)
        if stats.tokens_after > budget and self._trim_overflow:
            served, stats.trimmed_chars = self._trim(served, budget)
            stats.tokens_after = self._count(served)
        stats.overflow = stats.tokens_after > budget
        if stats.overflow:
            logger.warning("minimal window is %d tokens against a budget of %d; serving it", stats.tokens_after, budget)
        if stats.tokens_after + tail_tokens > self._serving_limit:
            raise ValueError(
                f"minimal servable window is {stats.tokens_after + tail_tokens} tokens, above the serving "
                f"limit of {self._serving_limit}; raise the prompt length or shrink the pinned head")
        return served, stats


def serving_limit_from(prompt_length: int, model_window: int, per_turn_max_tokens: int) -> int:
    """The largest prompt that is both stored untruncated and accepted by the inference server."""
    return min(int(prompt_length), int(model_window) - int(per_turn_max_tokens))

"""Scripted stand-ins for the creator LLM and the policy, for tests and offline examples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..serving.messages import Message, Messages


class ScriptedLLM:
    """Replies from a function of the conversation, or from a fixed list in order.

    Every call is recorded in ``.calls`` so a test can assert on the prompts it was sent.
    """

    def __init__(self, replies: Sequence[str] | Callable[[Messages], str | Message]) -> None:
        self._replies = replies
        self._i = 0
        self.calls: list[Messages] = []

    def chat(self, messages: Messages, *, tools: Sequence[dict[str, Any]] | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> Message:
        del tools, temperature, max_tokens
        self.calls.append(list(messages))
        if callable(self._replies):
            out = self._replies(list(messages))
        else:
            out = self._replies[min(self._i, len(self._replies) - 1)]
            self._i += 1
        return out if isinstance(out, dict) else {"role": "assistant", "content": out}

"""Token counters: how many tokens a message list costs the policy."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from .messages import Messages

TokenCounter = Callable[[Messages], int]


def char_counter(chars_per_token: float = 4.0) -> TokenCounter:
    """An approximate counter for tests and offline tooling without a tokenizer."""

    def count(messages: Messages) -> int:
        size = 0
        for m in messages:
            size += len(str(m.get("content") or "")) + 8
            for call in m.get("tool_calls") or []:
                size += len(json.dumps(call.get("function") or {}))
        return int(size / chars_per_token)

    return count


def chat_template_counter(tokenizer: Any, tools: Sequence[dict[str, Any]] | None = None) -> TokenCounter:
    """Count exactly as the policy is prompted: through the tokenizer's chat template."""

    def count(messages: Messages) -> int:
        ids = tokenizer.apply_chat_template(list(messages), tools=list(tools) if tools else None,
                                            add_generation_prompt=True, tokenize=True)
        return len(ids)

    return count

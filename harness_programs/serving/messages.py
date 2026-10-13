"""Helpers over OpenAI-format chat messages.

Messages throughout the package are plain dicts::

    {"role": "system" | "user" | "assistant" | "tool", "content": str,
     "tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}],
     "tool_call_id": str}

An assistant message carries its action in one of two ways: native tool calls (CodeSearch),
or an inline ``<action>...</action>`` / ``<answer>...</answer>`` span (SQL, ALFWorld). The
helpers here split an assistant turn into its free-text rationale and its action carrier.
Compression may rewrite the rationale; the carrier is always served verbatim, so no
compression program can make a rollout unparseable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

Message = dict[str, Any]
Messages = list[Message]

ACTION_SPAN_RE = re.compile(r"<(action|answer)>.*?</\1>", re.DOTALL | re.IGNORECASE)
THINK_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def content_of(message: Message) -> str:
    raw = message.get("content")
    return raw if isinstance(raw, str) else str(raw or "")


def is_assistant(message: Message) -> bool:
    return message.get("role") == "assistant"


def tool_calls_of(message: Message) -> list[dict[str, Any]]:
    """The message's tool calls as ``[{"id", "name", "args"}]`` with arguments decoded."""
    out: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw or "{}")
            except (TypeError, json.JSONDecodeError):
                args = {"raw": raw}
        out.append({"id": call.get("id", ""), "name": fn.get("name", ""), "args": args if isinstance(args, dict) else {"raw": args}})
    return out


def make_tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks, which hold drafts the turn did not commit to."""
    return THINK_SPAN_RE.sub("", text)


def committed_action(text: str) -> re.Match[str] | None:
    """The first ``<action>``/``<answer>`` span outside any reasoning block."""
    return ACTION_SPAN_RE.search(strip_think(text))


def uses_explicit_carrier(messages: Sequence[Message]) -> bool:
    """Whether any assistant turn in the transcript carries an explicit action.

    Once one turn is marked (tool call or tagged span), an unmarked turn is a malformed turn
    whose content is all rationale rather than an action.
    """
    for m in messages:
        if is_assistant(m) and (m.get("tool_calls") or committed_action(content_of(m)) is not None):
            return True
    return False


def split_reasoning(message: Message, *, explicit_carrier: bool = False) -> tuple[str, str]:
    """Split an assistant message into ``(reasoning, action)``.

    ``action`` is returned verbatim; a tagged span keeps its tags. With tool calls all inline
    text is rationale. With no carrier at all, the content is the action unless the
    transcript uses explicit carriers elsewhere.
    """
    text = strip_think(content_of(message))
    if message.get("tool_calls"):
        return text.strip(), ""
    match = ACTION_SPAN_RE.search(text)
    if match is None:
        return (text.strip(), "") if explicit_carrier else ("", text)
    reasoning = (text[: match.start()] + " " + text[match.end():]).strip()
    return reasoning, match.group(0)


def apply_reasoning(message: Message, reasoning: str, *, explicit_carrier: bool = False) -> Message:
    """A copy of an assistant message with its rationale replaced and its carrier intact."""
    text = strip_think(content_of(message))
    if message.get("tool_calls"):
        new_text = reasoning
    else:
        match = ACTION_SPAN_RE.search(text)
        if match is None:
            new_text = reasoning if explicit_carrier else text
        else:
            new_text = f"{reasoning}\n{match.group(0)}" if reasoning else match.group(0)
    return {**message, "content": new_text}


def turn_starts(messages: Sequence[Message]) -> list[int]:
    """Index of every assistant message; each one begins a turn."""
    return [i for i, m in enumerate(messages) if is_assistant(m)]


def pinned_head(messages: Sequence[Message]) -> Messages:
    """Everything before the first assistant message: the system prompt and the task.

    The head is never dropped and never rewritten; the context budget is measured from it.
    """
    starts = turn_starts(messages)
    return list(messages[: starts[0]]) if starts else list(messages)

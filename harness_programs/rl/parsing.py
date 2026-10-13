"""Turn one decoded generation into an OpenAI-format assistant message.

During RL the policy is sampled token-in/token-out, so there is no chat server to split a
completion into ``content`` and ``tool_calls``. This module does that split for the two action
carriers the domains use:

* **Native tool calls** (CodeSearch). The policy emits Hermes-style blocks,
  ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``, which become OpenAI
  ``tool_calls`` entries with stable ids so the environment can answer each call by id.
* **Text actions** (SQL, ALFWorld, toy). The ``<action>``/``<answer>`` span stays inside
  ``content`` untouched; the environment parses it itself.

A malformed call is kept rather than dropped: its raw text goes into ``arguments`` (which
:func:`harness_programs.serving.messages.tool_calls_of` decodes as ``{"raw": ...}``), so the
environment can answer it with an error and the transcript stays consistent with the tokens
the policy actually produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)


@dataclass
class ParsedTurn:
    """The parts of one generation: free text, decoded tool calls, and per-call parse errors."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: int = 0


def _call_id(prefix: str, turn: int, index: int) -> str:
    return f"{prefix}{turn}_{index}"


def _decode_call(body: str) -> tuple[str, str, str | None]:
    """``(name, arguments_json, error)`` for one ``<tool_call>`` body."""
    body = body.strip()
    try:
        guidance = json.loads(body)
    except json.JSONDecodeError as exc:
        return "", body, f"invalid JSON in tool call: {exc.msg}"
    if not isinstance(guidance, dict):
        return "", body, "tool call is not a JSON object"
    name = guidance.get("name")
    name = name if isinstance(name, str) else ""
    args = guidance.get("arguments", guidance.get("parameters", {}))
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except json.JSONDecodeError:
            return name, args, "tool call arguments are not valid JSON"
        args = decoded
    if not isinstance(args, dict):
        return name, json.dumps(args), "tool call arguments are not a JSON object"
    error = None if name else "tool call has no name"
    return name, json.dumps(args), error


def parse_tool_calls(text: str, *, turn: int = 0, id_prefix: str = "call_",
                     max_calls: int | None = None) -> ParsedTurn:
    """Split ``text`` into content and Hermes tool calls.

    Args:
        text: the decoded generation.
        turn: the episode turn index, folded into the call ids so they are unique per episode.
        id_prefix: prefix of the generated call ids.
        max_calls: keep at most this many calls (the rest are counted in ``skipped``);
            ``None`` keeps all.

    An unterminated ``<tool_call>`` at the end of a length-truncated generation is parsed as far
    as it goes and reported as an error rather than silently discarded.
    """
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    skipped = 0
    for match in TOOL_CALL_RE.finditer(text):
        if max_calls is not None and len(calls) >= max_calls:
            skipped += 1
            continue
        name, arguments, error = _decode_call(match.group(1))
        if error:
            errors.append(error)
        calls.append({"id": _call_id(id_prefix, turn, len(calls)), "type": "function",
                      "function": {"name": name, "arguments": arguments}})
    content = TOOL_CALL_RE.sub("", text).strip() if calls or skipped else text
    return ParsedTurn(content=content, tool_calls=calls, errors=errors, skipped=skipped)


def to_assistant_message(text: str, *, parse_tools: bool, turn: int = 0,
                         max_calls: int | None = None) -> tuple[dict[str, Any], ParsedTurn]:
    """The assistant message the environment steps on, plus the parse details.

    With ``parse_tools`` false (text-action domains) the generation passes through verbatim as
    ``content``; a ``<tool_call>`` block in it is just text.
    """
    if not parse_tools:
        parsed = ParsedTurn(content=text)
        return {"role": "assistant", "content": text}, parsed
    parsed = parse_tool_calls(text, turn=turn, max_calls=max_calls)
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    return message, parsed


__all__ = ["ParsedTurn", "TOOL_CALL_RE", "parse_tool_calls", "to_assistant_message"]

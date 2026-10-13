"""Hermes tool-call parsing of raw generations (harness_programs.rl.parsing)."""

from __future__ import annotations

import json

from harness_programs.rl.parsing import parse_tool_calls, to_assistant_message
from harness_programs.serving.messages import tool_calls_of


def test_single_call_becomes_openai_tool_call_with_id():
    text = 'I will search.\n<tool_call>\n{"name": "grep", "arguments": {"pattern": "foo"}}\n</tool_call>'
    parsed = parse_tool_calls(text, turn=3)
    assert parsed.content == "I will search."
    assert parsed.errors == []
    assert parsed.tool_calls == [{"id": "call_3_0", "type": "function",
                                  "function": {"name": "grep", "arguments": json.dumps({"pattern": "foo"})}}]


def test_multiple_calls_get_distinct_ids_and_max_calls_drops_extras():
    block = '<tool_call>{{"name": "{n}", "arguments": {{}}}}</tool_call>'
    text = "".join(block.format(n=n) for n in ("a", "b", "c"))
    parsed = parse_tool_calls(text, turn=1)
    assert [c["id"] for c in parsed.tool_calls] == ["call_1_0", "call_1_1", "call_1_2"]
    assert [c["function"]["name"] for c in parsed.tool_calls] == ["a", "b", "c"]
    capped = parse_tool_calls(text, turn=1, max_calls=1)
    assert [c["function"]["name"] for c in capped.tool_calls] == ["a"]
    assert capped.skipped == 2
    assert capped.content == ""


def test_string_arguments_and_parameters_key_are_accepted():
    a = parse_tool_calls('<tool_call>{"name": "x", "arguments": "{\\"k\\": 1}"}</tool_call>')
    b = parse_tool_calls('<tool_call>{"name": "x", "parameters": {"k": 1}}</tool_call>')
    assert json.loads(a.tool_calls[0]["function"]["arguments"]) == {"k": 1}
    assert json.loads(b.tool_calls[0]["function"]["arguments"]) == {"k": 1}
    assert a.errors == b.errors == []


def test_malformed_call_is_kept_with_raw_arguments_and_an_error():
    parsed = parse_tool_calls('<tool_call>{"name": "grep", "arguments": {"pattern": </tool_call>')
    assert len(parsed.tool_calls) == 1 and len(parsed.errors) == 1
    raw = parsed.tool_calls[0]["function"]["arguments"]
    assert raw.startswith('{"name": "grep"')
    message = {"role": "assistant", "content": "", "tool_calls": parsed.tool_calls}
    assert tool_calls_of(message)[0]["args"] == {"raw": raw}


def test_unterminated_trailing_call_is_parsed():
    parsed = parse_tool_calls('thinking <tool_call>{"name": "ls", "arguments": {"path": "."}}')
    assert parsed.content == "thinking"
    assert parsed.tool_calls[0]["function"]["name"] == "ls"
    assert parsed.errors == []


def test_text_without_calls_is_unchanged():
    text = "  Reasoning.\n<action>open file_a.txt</action>  "
    parsed = parse_tool_calls(text)
    assert parsed.content == text and parsed.tool_calls == []


def test_text_action_domains_pass_generation_through_verbatim():
    text = 'Plan.\n<tool_call>{"name": "x", "arguments": {}}</tool_call><action>list</action>'
    message, parsed = to_assistant_message(text, parse_tools=False)
    assert message == {"role": "assistant", "content": text}
    assert parsed.tool_calls == []


def test_tool_domains_get_tool_calls_only_when_present():
    message, _ = to_assistant_message("no calls here", parse_tools=True)
    assert message == {"role": "assistant", "content": "no calls here"}
    message, parsed = to_assistant_message('<tool_call>{"name": "x", "arguments": {}}</tool_call>',
                                           parse_tools=True, turn=2)
    assert message["tool_calls"][0]["id"] == "call_2_0"
    assert tool_calls_of(message) == [{"id": "call_2_0", "name": "x", "args": {}}]
    assert parsed.errors == []

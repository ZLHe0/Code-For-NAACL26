"""A minimal OpenAI-compatible chat client (stdlib only), used for both roles in the method.

* The **creator** LLM writes compression programs, triggers and judgments.
* The **policy** is served by an inference server with an OpenAI-compatible endpoint during
  offline rollouts.

Configuration comes from arguments or the environment::

    HARNESS_LLM_BASE_URL   e.g. http://localhost:8000/v1
    HARNESS_LLM_API_KEY    optional
    HARNESS_LLM_MODEL      model name the endpoint expects
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from ..serving.messages import Message, Messages

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(Exception):
    pass


class ChatModel(Protocol):
    def chat(self, messages: Messages, *, tools: Sequence[dict[str, Any]] | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> Message: ...


class OpenAIChat:
    """POSTs to ``{base_url}/chat/completions`` and returns the assistant message dict."""

    def __init__(self, *, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout_s: float = 600.0, retries: int = 3, prefix: str = "HARNESS_LLM") -> None:
        self.base_url = (base_url or os.environ.get(f"{prefix}_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get(f"{prefix}_MODEL") or ""
        self.api_key = api_key if api_key is not None else os.environ.get(f"{prefix}_API_KEY", "")
        if not self.base_url or not self.model:
            raise LLMError(f"set {prefix}_BASE_URL and {prefix}_MODEL (or pass base_url/model)")
        self.timeout_s = timeout_s
        self.retries = retries

    def chat(self, messages: Messages, *, tools: Sequence[dict[str, Any]] | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> Message:
        body: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools:
            body["tools"] = list(tools)
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                    guidance = json.loads(response.read())
                message = guidance["choices"][0]["message"]
                return {k: v for k, v in message.items() if v is not None}
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                if attempt == self.retries:
                    raise LLMError(f"chat completion failed after {attempt + 1} attempt(s): {exc}") from exc
                logger.warning("chat completion failed (%s); retrying in %.0fs", exc, delay)
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")


def complete(llm: ChatModel, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
    messages: Messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return str(llm.chat(messages, temperature=temperature).get("content") or "")


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in ``text``: a fenced block if present, else the first balanced ``{...}``."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        while start >= 0:
            depth, in_str, escape = 0, False, False
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_str:
                    escape = not escape and ch == "\\"
                    if ch == '"' and not escape:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(candidate[start:i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            return obj
                        break
            start = candidate.find("{", start + 1)
    raise LLMError("no JSON object found in the reply")


def ask_json(llm: ChatModel, prompt: str, *, required: Sequence[str] = (), attempts: int = 2,
             system: str | None = None, temperature: float | None = None) -> dict[str, Any]:
    """Ask for a JSON object, re-asking when the reply is unparseable or missing ``required`` keys."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            obj = extract_json(complete(llm, prompt, system=system, temperature=temperature))
        except LLMError as exc:
            last = exc
            continue
        missing = [k for k in required if k not in obj]
        if not missing:
            return obj
        last = LLMError(f"reply is missing {missing}")
    raise LLMError(str(last))

"""Scripted agents for the toy domain: a policy and a creator LLM that need no model server.

They exist so the whole pipeline -- stage 1 campaign, ``program.json``, served rollouts -- runs
offline in seconds, in the example and in the tests. Neither is clever:

* :func:`toy_policy` reads only what it was *served*. It lists the tree, opens files it has not
  seen opened, and answers when the live observation contains the target word. When the
  listing or its own earlier actions are evicted it re-lists and re-opens, which is exactly the
  failure a tight context budget causes in real agents. It follows guidance that names the files
  not yet opened.
* :class:`ToyCreator` answers the proposer, repair and reviewer prompts of stage 1. It proposes
  one program that digests old file bodies and one that rewrites the live turn, which the
  refine loop drops because it breaks the serving contract, and it accepts whatever runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..serving.messages import Message, Messages, content_of
from .toy import _ACTION_RE, _OBS_RE

_TARGET_RE = re.compile(r"word '(\w+)'")
_FILE_RE = re.compile(r"^file_\w+\.txt$")
_GUIDE_RE = re.compile(r"Not yet opened: ([^\n]*)")

# A trigger of the shape stage 2 generates: a trigger over the episode so far, a static guidance,
# and a renderer that writes the guidance from the episode. It restates the state eviction hides.
UNOPENED_TRIGGER = '''\
def check(ep):
    return any(a.startswith("open ") for a in ep.actions)
'''

UNOPENED_RENDER = '''\
def render(ep):
    listing = []
    for step in ep.steps:
        if step.action == "list":
            body = step.observation.replace("<observation>", " ").replace("</observation>", " ")
            listing = [name for name in body.split() if name.endswith(".txt")]
    opened = {a[5:].strip() for a in ep.actions if a.startswith("open ")}
    rest = [name for name in listing if name not in opened]
    return "Not yet opened: " + ", ".join(rest) if rest else ""
'''

DIGEST_PROGRAM = '''\
def compress(state):
    for turn in state[:-1]:
        if turn.action.startswith("open "):
            turn.digest = "[" + turn.action + ": no match]"
    return state
'''

LIVE_EDIT_PROGRAM = '''\
def compress(state):
    for turn in state:
        turn.observation = turn.observation[:40]
    return state
'''


def _reply(text: str) -> Message:
    return {"role": "assistant", "content": text}


def toy_policy(messages: Messages) -> Message:
    head = next((content_of(m) for m in messages if m.get("role") == "user"), "")
    target = (_TARGET_RE.search(head) or [None, ""])[1]
    actions = [a.strip() for m in messages if m.get("role") == "assistant"
               for a in _ACTION_RE.findall(content_of(m))]
    last_ai = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    last = next((content_of(m) for m in messages[last_ai + 1:] if "<observation>" in content_of(m)), "")
    if target and "<observation>" in last and target in last and actions and actions[-1].startswith("open "):
        return _reply(f"The live file has the word.\n<answer>{actions[-1][5:]}</answer>")
    guided = [re.findall(r"file_\w+\.txt", g.group(1)) for m in messages[last_ai + 1:]
              for g in [_GUIDE_RE.search(content_of(m))] if g]
    if guided and guided[-1]:
        return _reply(f"Guidance says it is still unopened.\n<action>open {guided[-1][0]}</action>")
    listing: list[str] = []
    for m in messages:
        body = (_OBS_RE.search(content_of(m)) or [None, ""])[1].strip()
        names = body.splitlines()
        if names and all(_FILE_RE.match(n) for n in names):
            listing = names
    if not listing:
        return _reply("I need the file list.\n<action>list</action>")
    opened = {a[5:].strip() for a in actions if a.startswith("open ")}
    for name in listing:
        if name not in opened:
            return _reply(f"Open the next unseen file.\n<action>open {name}</action>")
    return _reply(f"Nothing matched; guess.\n<answer>{listing[0]}</answer>")


class ToyCreator:
    """Routes on the system prompt: proposer and repair, or reviewer."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, messages: Messages, **_: Any) -> Message:
        system = content_of(messages[0]) if messages and messages[0].get("role") == "system" else ""
        if '"verdict"' in system:
            self.calls.append("review")
            return _reply(json.dumps({"verdict": "accept", "diagnosis": "keeps every action and the live turn"}))
        self.calls.append("propose")
        return _reply(json.dumps({"proposals": [
            {"name": "digest_opened_files", "rationale": "old file bodies never matched; keep only the action",
             "code": DIGEST_PROGRAM},
            {"name": "truncate_everything", "rationale": "cut every observation to 40 chars",
             "code": LIVE_EDIT_PROGRAM},
        ]}))

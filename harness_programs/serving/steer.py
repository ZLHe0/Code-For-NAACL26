"""Steering: a set of triggers, each paired with a short piece of guidance.

A trigger is a small program over the state that fires on one detectable failure mode:

* ``check(ep) -> bool`` -- the trigger program. Runs at serve time on the episode prefix before
  each assistant turn. It sees no reward, gold label or final outcome.
* ``guidance`` -- the text served on the turns where ``check`` fires. An optional
  ``render(ep) -> str`` may write it from the episode instead.
* ``fault(ep) -> bool`` -- whether the failure the trigger targets occurred, evaluated offline on
  a complete episode (``ep.gold`` carries the domain's answer view). Replay counts it per
  training step. It is never served; it defaults to ``check``.

Fired guidance texts are stacked into one ephemeral user message appended after the budget stage.
Guidance is never written into the transcript, so it does not accumulate across turns and a
compression program never sees it.

Failure policy: a trigger that fails to compile, or raises on the sample episode, fails at
load. At serve time a raising trigger does not fire and counts a strike.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .messages import Message, Messages, content_of, split_reasoning, tool_calls_of, uses_explicit_carrier
from .sandbox import CodeCache, code_fingerprint, compile_entrypoint, lenient_policy, screen_source

logger = logging.getLogger(__name__)

GUIDANCE_BLOCK = "Guidance to follow for your next step:\n{text}"
RENDER_MAX_CHARS = 2000

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")
_ACTION_TAG = re.compile(r"^<action>(.*?)</action>\s*$", re.DOTALL | re.IGNORECASE)


def norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def tokens(text: str) -> set[str]:
    return set(norm(text).split())


_HELPERS = {"norm": norm, "tokens": tokens}
TRIGGER_POLICY = lenient_policy("check", _HELPERS)
FAULT_POLICY = lenient_policy("fault", _HELPERS)
RENDER_POLICY = lenient_policy("render", _HELPERS)

_CHECK_CACHE = CodeCache(TRIGGER_POLICY, label="trigger trigger")
_RENDER_CACHE = CodeCache(RENDER_POLICY, label="trigger renderer")


class TriggerError(Exception):
    """A trigger failed to compile, or raised on the sample episode at load time."""


# ---------------------------------------------------------------------------------------------
# The episode view a predicate reads
# ---------------------------------------------------------------------------------------------


@dataclass
class Step:
    """One agent turn: the action taken, and the observation that followed."""

    index: int
    action: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observation: str = ""
    reasoning: str = ""

    @property
    def command(self) -> str:
        if self.action:
            return self.action
        return "; ".join(f"{c['name']}({json.dumps(c['args'], sort_keys=True)})" for c in self.tool_calls)


@dataclass
class Episode:
    """The trajectory as the agent saw it. In flight, the prefix so far.

    ``final_output`` and ``status`` are empty at serve time, and ``gold`` is ``None``; only an
    offline fault predicate, graded on a complete episode, sees them set.
    """

    task_text: str
    steps: list[Step] = field(default_factory=list)
    max_steps: int = 0
    final_output: str = ""
    status: str = ""
    gold: Any = None

    @property
    def question(self) -> str:
        return self.task_text

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[str]:
        return [s.command for s in self.steps]

    @property
    def observations(self) -> list[str]:
        return [s.observation for s in self.steps]

    @property
    def reasonings(self) -> list[str]:
        return [s.reasoning for s in self.steps]

    @property
    def queries(self) -> list[str]:
        """Every tool-call argument value, flattened in order."""
        return [str(v) for s in self.steps for c in s.tool_calls for v in (c.get("args") or {}).values()]


def episode_from_messages(messages: Sequence[Message], *, max_steps: int = 0, status: str = "",
                          gold: Any = None) -> Episode:
    """Build an :class:`Episode` from OpenAI-format messages.

    The task is the last message of the pinned head; each assistant message opens a step, and
    the tool/user messages after it are its observation.
    """
    messages = list(messages)
    first_ai = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
    head = [m for m in messages[:first_ai] if m.get("role") != "system"]
    task_text = content_of(head[-1]).strip() if head else ""
    explicit = uses_explicit_carrier(messages)
    steps: list[Step] = []
    for message in messages[first_ai:]:
        role = message.get("role")
        if role == "assistant":
            reasoning, action = split_reasoning(message, explicit_carrier=explicit)
            tag = _ACTION_TAG.match(action.strip())
            calls = [{"name": c["name"], "args": c["args"]} for c in tool_calls_of(message)]
            steps.append(Step(index=len(steps), action=tag.group(1).strip() if tag else action.strip(),
                              tool_calls=calls, reasoning=reasoning))
        elif role in ("tool", "user") and steps:
            text = content_of(message).strip()
            steps[-1].observation = f"{steps[-1].observation}\n{text}".strip()
    return Episode(task_text=task_text, steps=steps, max_steps=max_steps, status=status, gold=gold)


def sample_episode(*, gold: Any = None) -> Episode:
    """A small two-step episode used to smoke-test triggers at load time.

    A trigger is smoked without gold, so a trigger that reads ``ep.gold`` fails here exactly
    as it would in training.
    """
    return Episode(
        task_text="Sample task statement.",
        steps=[
            Step(index=0, action="SELECT 1", observation="1", reasoning="Check the connection."),
            Step(index=1, tool_calls=[{"name": "terminal", "args": {"command": "ls"}}],
                 observation="a.py\nb.py", reasoning="List the tree."),
        ],
        max_steps=10,
        gold=gold,
    )


# ---------------------------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """A named trigger and the guidance it serves. ``fault`` defaults to the trigger itself."""

    name: str
    code: str
    guidance: str
    fault: str = ""
    render: str = ""

    @property
    def fault_code(self) -> str:
        return self.fault or self.code

    def to_served(self) -> dict[str, str]:
        """The fields written into a served program; the fault is never served."""
        row = {"name": self.name, "code": self.code, "guidance": self.guidance}
        if self.render:
            row["render"] = self.render
        return row


def screen_trigger(code: str) -> list[str]:
    return screen_source(code, TRIGGER_POLICY)


def screen_fault(code: str) -> list[str]:
    return screen_source(code, FAULT_POLICY)


def compile_trigger(code: str) -> Callable[[Episode], Any]:
    try:
        return compile_entrypoint(code, TRIGGER_POLICY)
    except ValueError as exc:
        raise TriggerError(str(exc)) from exc


def compile_fault(code: str) -> Callable[[Episode], Any]:
    try:
        return compile_entrypoint(code, FAULT_POLICY)
    except ValueError as exc:
        raise TriggerError(str(exc)) from exc


def compile_render(code: str) -> Callable[[Episode], Any]:
    try:
        return compile_entrypoint(code, RENDER_POLICY)
    except ValueError as exc:
        raise TriggerError(str(exc)) from exc


class TriggerRuntime:
    """Serves a fixed list of triggers. Stateless across episodes.

    Optional dose caps bound how often guidance is served within one episode; ``0`` means
    unbounded. Prior firings are recomputed from the prefix rather than remembered, so
    concurrent episodes cannot share state.
    """

    def __init__(self, triggers: Sequence[Trigger] = (), *, max_per_episode: int = 0, max_per_trigger: int = 0) -> None:
        if max_per_episode < 0 or max_per_trigger < 0:
            raise TriggerError("dose caps must be non-negative; 0 means unbounded")
        self._triggers = list(triggers)
        self._max_per_episode = max_per_episode
        self._max_per_trigger = max_per_trigger

    @property
    def triggers(self) -> list[Trigger]:
        return list(self._triggers)

    @property
    def setup(self) -> dict[str, Any]:
        return {"installed": [{"name": c.name, "code_sha": code_fingerprint(c.code)} for c in self._triggers]}

    def _holds(self, trigger: Trigger, episode: Episode) -> bool:
        fn = _CHECK_CACHE.compiled(trigger.code)
        if fn is None:
            return False
        try:
            held = bool(fn(episode))
        except Exception:  # noqa: BLE001 -- a raising trigger must not kill the rollout
            if _CHECK_CACHE.strike(trigger.code) or _CHECK_CACHE.strikes(trigger.code) == 1:
                logger.warning("trigger %s raised; not firing this turn", trigger.name, exc_info=True)
            return False
        _CHECK_CACHE.clear_strikes(trigger.code)
        return held

    def _guidance(self, trigger: Trigger, episode: Episode) -> str:
        if not trigger.render:
            return trigger.guidance
        fn = _RENDER_CACHE.compiled(trigger.render)
        if fn is None:
            return trigger.guidance
        try:
            out = fn(episode)
        except Exception:  # noqa: BLE001 -- fall back to the static guidance
            _RENDER_CACHE.strike(trigger.render)
            return trigger.guidance
        _RENDER_CACHE.clear_strikes(trigger.render)
        if not (isinstance(out, str) and out.strip()):
            return trigger.guidance
        return out if len(out) <= RENDER_MAX_CHARS else out[:RENDER_MAX_CHARS] + " ..."

    def _prior_fires(self, trigger: Trigger, episode: Episode, stop_at: int) -> int:
        seen = 0
        for j in range(len(episode.steps)):
            prefix = Episode(task_text=episode.task_text, steps=episode.steps[:j], max_steps=episode.max_steps)
            if self._holds(trigger, prefix):
                seen += 1
                if seen >= stop_at:
                    break
        return seen

    def fire(self, messages: Sequence[Message], *, max_steps: int = 0) -> list[tuple[str, str]]:
        """Every ``(name, guidance)`` whose trigger holds on the episode so far."""
        if not self._triggers:
            return []
        episode = episode_from_messages(messages, max_steps=max_steps)
        held = [c for c in self._triggers if self._holds(c, episode)]
        if held and self._max_per_episode:
            spent = sum(self._prior_fires(c, episode, self._max_per_episode + 1) for c in self._triggers)
            held = held[: max(0, self._max_per_episode - spent)]
        if held and self._max_per_trigger:
            held = [c for c in held if self._prior_fires(c, episode, self._max_per_trigger) < self._max_per_trigger]
        return [(c.name, self._guidance(c, episode)) for c in held]

    @staticmethod
    def guidance_message(fired: Sequence[tuple[str, str]]) -> Messages:
        text = "\n\n".join(p for _, p in fired if p.strip())
        return [{"role": "user", "content": GUIDANCE_BLOCK.format(text=text)}] if text else []

    def tail(self, messages: Sequence[Message], *, max_steps: int = 0) -> tuple[Messages, list[str]]:
        """This turn's ephemeral guidance message (at most one) and the triggers behind it."""
        fired = self.fire(messages, max_steps=max_steps)
        return (self.guidance_message(fired), [n for n, _ in fired]) if fired else ([], [])


def build_runtime(triggers: Sequence[Trigger], **caps: int) -> TriggerRuntime:
    """Compile and smoke-run every trigger; raises :class:`TriggerError` at load time."""
    seen: set[str] = set()
    for trigger in triggers:
        if not trigger.name or trigger.name in seen:
            raise TriggerError(f"trigger names must be non-empty and unique: {trigger.name!r}")
        seen.add(trigger.name)
        try:
            compile_trigger(trigger.code)(sample_episode())
        except TriggerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TriggerError(f"trigger {trigger.name!r} raised on the sample episode: {exc!r}") from exc
        if trigger.render:
            out = compile_render(trigger.render)(sample_episode())
            if not isinstance(out, str):
                raise TriggerError(f"trigger {trigger.name!r} render returned {type(out).__name__}, not str")
    return TriggerRuntime(triggers, **caps)


def trigger_strike_counts() -> dict[str, int]:
    return _CHECK_CACHE.strike_counts()

"""Compression: a generated ``compress(state) -> state`` program over the domain's turn state.

Each domain registers a *state DSL* describing how its transcript is parsed into typed turns
(the ``state`` a compression program edits) and how edited turns are rendered back into
messages. A compression program may only change what a turn *costs* -- rewrite an
observation, clear a rationale, replace a turn with a digest. It cannot add or remove turns,
cannot rewrite the pinned task statement, and cannot touch an action carrier. Dropping whole
turns to meet the token budget is the job of the budget stage (:mod:`.budget`), which runs
after compression.

Failure policy: a program that fails to compile, or raises on the domain's sample transcript,
fails at load time. At serve time a raise degrades that turn to the uncompressed transcript
and counts a strike against the circuit breaker.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .messages import (
    Message,
    Messages,
    apply_reasoning,
    content_of,
    is_assistant,
    pinned_head,
    uses_explicit_carrier,
)
from .sandbox import CodeCache, call_with_deadline, compile_entrypoint, lenient_policy, screen_source

logger = logging.getLogger(__name__)

COMPRESSION_POLICY = lenient_policy("compress")
_SERVING_CACHE = CodeCache(COMPRESSION_POLICY, label="compression program")

NOOP_PROGRAM = "def compress(state):\n    return state\n"


class CompressionError(Exception):
    """A compression program failed to compile or returned something that is not state."""


class HeadRewritten(CompressionError):
    """A compression program modified the pinned task statement. Never degraded."""


@runtime_checkable
class StateDSL(Protocol):
    """What a domain provides to make its transcript compressible.

    Implemented as module-level names, so the domain module itself is the DSL.

    ``Turn`` is the type a program iterates. ``parse_turn`` works per observation; a domain
    whose model turn can produce several observations groups them in the optional
    ``finalize_turns`` hook and flattens them back in ``expand_state``.
    """

    Turn: type
    CONTRACT: str

    def is_observation(self, message: Message) -> bool: ...

    def parse_turn(self, *, obs_message: Message, ai_message: Message | None, msg_idx: int,
                   ai_idx: int, explicit_carrier: bool) -> Any: ...

    def render_turn(self, turn: Any) -> str: ...

    def rebuild_obs(self, message: Message, text: str, turn: Any) -> Message: ...

    def sample_messages(self) -> Messages: ...


_REQUIRED = ("Turn", "CONTRACT", "is_observation", "parse_turn", "render_turn", "rebuild_obs", "sample_messages")
_DSLS: dict[str, Any] = {}


def register_state_dsl(domain: str, dsl: Any) -> None:
    missing = [name for name in _REQUIRED if not hasattr(dsl, name)]
    if missing:
        raise ValueError(f"state DSL for {domain!r} is missing: {', '.join(missing)}")
    _DSLS[domain] = dsl


def get_state_dsl(domain: str) -> Any:
    if domain not in _DSLS:
        # Domains register on import; import lazily so the serving core has no domain imports.
        from .. import domains

        domains.load(domain)
    if domain not in _DSLS:
        raise KeyError(f"no state DSL registered for {domain!r}; registered={sorted(_DSLS)}")
    return _DSLS[domain]


# ---------------------------------------------------------------------------------------------
# Transcript <-> state
# ---------------------------------------------------------------------------------------------


def pair_observations(messages: Sequence[Message], is_observation: Callable[[Message], bool]) -> list[tuple[int, int]]:
    """``[(obs_idx, ai_idx)]`` for every observation, oldest first.

    A tool observation pairs with the assistant message that issued its ``tool_call_id``;
    a free-text observation pairs with the nearest preceding assistant message. ``ai_idx``
    is ``-1`` when nothing produced the observation.
    """
    ai_by_call: dict[str, int] = {}
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            if call.get("id"):
                ai_by_call[call["id"]] = i
    pairs: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if not is_observation(m):
            continue
        ai_idx = ai_by_call.get(m.get("tool_call_id") or "", -1)
        if ai_idx < 0:
            ai_idx = next((k for k in range(i - 1, -1, -1) if is_assistant(messages[k])), -1)
        pairs.append((i, ai_idx))
    return pairs


def parse_state(messages: Sequence[Message], dsl: Any) -> list[Any]:
    """Transcript -> ``list[Turn]``, oldest first."""
    explicit = uses_explicit_carrier(messages)
    turns = [
        dsl.parse_turn(obs_message=messages[i], ai_message=messages[a] if a >= 0 else None,
                       msg_idx=i, ai_idx=a, explicit_carrier=explicit)
        for i, a in pair_observations(messages, dsl.is_observation)
    ]
    finalize = getattr(dsl, "finalize_turns", None)
    turns = list(finalize(turns)) if finalize is not None else turns
    for j, turn in enumerate(turns):
        turn.index, turn.n_turns = j, len(turns)
    return turns


def reserialize(messages: Sequence[Message], turns: Sequence[Any], dsl: Any) -> Messages:
    """``list[Turn]`` -> transcript. Messages no turn refers to pass through untouched.

    An assistant message is rewritten only when the program changed its rationale, so the
    no-op program returns the transcript byte-identical.
    """
    by_obs = {t._msg_idx: t for t in turns if getattr(t, "_msg_idx", -1) >= 0}
    by_ai: dict[int, Any] = {}
    for t in turns:
        if getattr(t, "_ai_idx", -1) >= 0:
            by_ai.setdefault(t._ai_idx, t)
    explicit = uses_explicit_carrier(messages)
    out: Messages = []
    for i, m in enumerate(messages):
        if i in by_obs:
            out.append(dsl.rebuild_obs(m, dsl.render_turn(by_obs[i]), by_obs[i]))
        elif i in by_ai and by_ai[i].reasoning != by_ai[i]._orig_reasoning:
            out.append(apply_reasoning(m, by_ai[i].reasoning, explicit_carrier=explicit))
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------------------------
# Compile and serve
# ---------------------------------------------------------------------------------------------


def screen_program(code: str) -> list[str]:
    return screen_source(code, COMPRESSION_POLICY)


def compile_program(code: str) -> Callable[[Any], Any]:
    try:
        return compile_entrypoint(code, COMPRESSION_POLICY)
    except ValueError as exc:
        raise CompressionError(str(exc)) from exc


def _check_invariants(messages: Sequence[Message], out: Sequence[Message]) -> None:
    if len(out) != len(messages):
        raise CompressionError(
            f"compress() changed the transcript length ({len(messages)} -> {len(out)} messages); "
            "a program may change what a turn costs, never whether it exists")
    for before, after in zip(pinned_head(messages), out):
        if content_of(after) != content_of(before):
            raise HeadRewritten("compress() rewrote the pinned task statement; the budget is measured from it")
    for before, after in zip(messages, out):
        if is_assistant(before) and before.get("tool_calls") != after.get("tool_calls"):
            raise CompressionError("compress() changed an action carrier")


def apply_program(fn: Callable[[Any], Any], messages: Sequence[Message], dsl: Any) -> Messages:
    """One pass: parse -> ``compress`` -> re-serialize, with invariants checked."""
    state = parse_state(copy.deepcopy(list(messages)), dsl)
    new_state = call_with_deadline(fn, state)
    if not isinstance(new_state, list):
        raise CompressionError(f"compress() returned {type(new_state).__name__}, not a list of turns")
    expand = getattr(dsl, "expand_state", None)
    if expand is not None:
        new_state = list(expand(new_state))
    out = reserialize(messages, new_state, dsl)
    _check_invariants(messages, out)
    return out


Stage1 = Callable[[Messages], Messages]


def make_stage1(code: str, dsl: Any, *, cache: CodeCache | None = None,
                on_raise: Callable[[BaseException], None] | None = None) -> Stage1:
    """Wrap a program as ``messages -> messages``, degrading to pass-through on a raise."""
    breaker = _SERVING_CACHE if cache is None else cache

    def stage1(messages: Messages) -> Messages:
        fn = breaker.compiled(code)
        if fn is None:
            return list(messages)
        try:
            out = apply_program(fn, messages, dsl)
        except HeadRewritten:
            raise
        except Exception as exc:  # noqa: BLE001 -- a raising program must not kill the rollout
            if on_raise is not None:
                on_raise(exc)
            if breaker.strike(code) or breaker.strikes(code) == 1:
                logger.warning("compression program raised; serving this turn uncompressed", exc_info=True)
            return list(messages)
        breaker.clear_strikes(code)
        return out

    return stage1


def _smoke(domain: str, code: str) -> Any:
    dsl = get_state_dsl(domain)
    wrong = {type(t).__name__ for t in parse_state(dsl.sample_messages(), dsl) if not isinstance(t, dsl.Turn)}
    if wrong:
        raise CompressionError(f"state DSL for {domain!r} declares Turn={dsl.Turn.__name__} but yields {sorted(wrong)}")
    fn = compile_program(code)
    try:
        apply_program(fn, dsl.sample_messages(), dsl)
    except CompressionError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surfaced as a load failure
        raise CompressionError(f"program raised on the {domain} sample transcript: {exc!r}") from exc
    return dsl


def build_stage1(domain: str, code: str) -> Stage1:
    """Compile, smoke-test on the domain's sample transcript, and wrap for serving."""
    return make_stage1(code, _smoke(domain, code))


class MeasuredStage1:
    """A Stage-1 wrapper for offline measurement: private breaker that never trips, raise counter.

    Serving wants a forgiving breaker; measurement wants every failure counted, so a program
    that raises on every episode is never silently measured as the no-op.
    """

    def __init__(self, domain: str, code: str) -> None:
        dsl = _smoke(domain, code)
        self.code = code
        self.raises = 0
        self.calls = 0
        cache = CodeCache(COMPRESSION_POLICY, poison_limit=1 << 40, label="compression program (measured)")
        self._stage1 = make_stage1(code, dsl, cache=cache, on_raise=self._count)

    def _count(self, _exc: BaseException) -> None:
        self.raises += 1

    def __call__(self, messages: Messages) -> Messages:
        self.calls += 1
        return self._stage1(messages)


def serving_strike_counts() -> dict[str, int]:
    return _SERVING_CACHE.strike_counts()

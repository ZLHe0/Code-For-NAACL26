"""Screening, compiling and calling model-generated code.

Every piece of generated code in the system -- a compression program, a trigger trigger, a
fault predicate, a guidance renderer -- is a single top-level Python function. A
:class:`CodePolicy` fixes its entrypoint name, the imports it may use and the builtins it may
not, so each flavour of generated code is a policy rather than a separate sandbox.

The restriction is enforced by an AST screen, not by the exec namespace. The screen rejects
known escape hatches by name and bans dunder attribute access. It is a guard against
accidental misuse by a code-writing LLM, not a security boundary against an adversary.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import itertools
import json
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Deterministic, side-effect-free modules only, so replay and grading are reproducible.
ALLOWED_IMPORTS = frozenset({"re", "json", "math", "itertools", "collections"})

FORBIDDEN_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "hasattr", "breakpoint", "exit", "quit",
    }
)

# A source that raises this many times in a row is disabled for the rest of the process.
POISON_LIMIT = 5

# Wall-clock ceiling on one call into generated code. Termination is undecidable, so the
# only available bound is a clock applied at call time.
CALL_TIMEOUT_S = 10.0

_DUNDER_MSG = "dunder attribute access is not allowed"


class CodeTimeout(Exception):
    """Generated code did not return within its wall-clock budget."""


def call_with_deadline(fn: Callable[..., Any], *args: Any, seconds: float | None = None) -> Any:
    """Call ``fn(*args)`` in the current thread, interrupting it after ``seconds``.

    The interrupt is an asynchronous exception delivered between bytecodes. It stops a
    runaway Python loop, but not a single long-running C call.
    """
    seconds = CALL_TIMEOUT_S if seconds is None else seconds
    target = threading.get_ident()
    fired = threading.Event()

    def interrupt() -> None:
        fired.set()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(target), ctypes.py_object(CodeTimeout))

    timer = threading.Timer(seconds, interrupt)
    timer.daemon = True
    timer.start()
    try:
        return fn(*args)
    except CodeTimeout:
        raise CodeTimeout(f"did not return within {seconds:g}s") from None
    finally:
        timer.cancel()
        if fired.is_set():
            # The timer may fire after fn returned; clear any exception still in flight.
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(target), None)


def stdlib_scope() -> dict[str, Any]:
    """A fresh copy of the names every policy's exec scope starts from."""
    return {
        "re": re, "json": json, "math": math, "itertools": itertools,
        "Counter": Counter, "defaultdict": defaultdict,
    }


def safe_getattr(obj: Any, name: str, *default: Any) -> Any:
    """``getattr`` that refuses dunder names at call time."""
    if isinstance(name, str) and name.startswith("__"):
        raise ValueError(_DUNDER_MSG)
    return getattr(obj, name, *default)


def safe_hasattr(obj: Any, name: str) -> bool:
    """``hasattr`` that refuses dunder names at call time."""
    if isinstance(name, str) and name.startswith("__"):
        raise ValueError(_DUNDER_MSG)
    return hasattr(obj, name)


@dataclass(frozen=True)
class CodePolicy:
    """What one flavour of generated code is allowed to be."""

    entrypoint: str
    allowed_imports: frozenset[str] = ALLOWED_IMPORTS
    forbidden_names: frozenset[str] = FORBIDDEN_NAMES
    base_scope: dict[str, Any] = field(default_factory=stdlib_scope)
    filename: str = "<generated>"


def lenient_policy(entrypoint: str, extra_scope: dict[str, Any] | None = None) -> CodePolicy:
    """A policy that allows ``getattr``/``hasattr`` through dunder-refusing shims.

    Code-writing LLMs write defensive ``getattr(x, "field", "")`` constantly; banning the
    names outright rejects most otherwise-good programs. The shims keep the dunder ban.
    """
    scope = {**stdlib_scope(), "getattr": safe_getattr, "hasattr": safe_hasattr, **(extra_scope or {})}
    return CodePolicy(
        entrypoint=entrypoint,
        forbidden_names=FORBIDDEN_NAMES - {"getattr", "hasattr"},
        base_scope=scope,
        filename=f"<{entrypoint}>",
    )


def screen_source(code: str, policy: CodePolicy) -> list[str]:
    """Return the reasons ``code`` violates ``policy``; an empty list means clean."""
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.module if isinstance(node, ast.ImportFrom) else node.names[0].name) or ""
            if mod.split(".")[0] not in policy.allowed_imports:
                errors.append(f"import of {mod!r} not allowed")
        if isinstance(node, ast.Name) and node.id in policy.forbidden_names:
            errors.append(f"use of {node.id!r} not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute {node.attr!r} not allowed")
    # An async entrypoint would return a coroutine, which is always truthy.
    if not any(isinstance(n, ast.FunctionDef) and n.name == policy.entrypoint for n in tree.body):
        errors.append(f"no top-level (non-async) {policy.entrypoint}() function defined")
    return errors


def compile_entrypoint(
    code: str, policy: CodePolicy, scope_overrides: dict[str, Any] | None = None
) -> Callable[..., Any]:
    """Screen ``code`` and return its entrypoint; raises ``ValueError`` if it is not clean."""
    errors = screen_source(code, policy)
    if errors:
        raise ValueError("; ".join(errors))
    scope: dict[str, Any] = dict(policy.base_scope)
    scope.update(scope_overrides or {})
    exec(compile(ast.parse(code), policy.filename, "exec"), scope)  # noqa: S102 -- screened above
    fn = scope.get(policy.entrypoint)
    if not callable(fn):
        raise ValueError(f"no {policy.entrypoint}() function defined")
    return fn


def code_fingerprint(code: str) -> str:
    """A short content hash of a source string, used as a stable identifier."""
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()[:8]  # noqa: S324 -- an id


class CodeCache:
    """Compile cache and circuit breaker for one flavour of generated code.

    A source compiles once per process. A source that fails to compile, or raises
    ``poison_limit`` times in a row, is disabled, so a bad artifact degrades its own feature
    instead of failing every turn of every rollout.
    """

    def __init__(self, policy: CodePolicy, *, poison_limit: int = POISON_LIMIT, label: str = "code") -> None:
        self._policy = policy
        self._limit = poison_limit
        self._label = label
        self._compiled: dict[str, Callable[..., Any]] = {}
        self._strikes: dict[str, int] = {}
        self._poisoned: set[str] = set()

    def compiled(self, code: str) -> Callable[..., Any] | None:
        """The compiled entrypoint, or ``None`` when the source is disabled."""
        key = code_fingerprint(code)
        if key in self._poisoned:
            return None
        fn = self._compiled.get(key)
        if fn is None:
            try:
                fn = compile_entrypoint(code, self._policy)
            except Exception:  # noqa: BLE001 -- a bad artifact must not break serving
                logger.warning("%s failed to compile; disabled (hash=%s)", self._label, key)
                self._poisoned.add(key)
                return None
            self._compiled[key] = fn
        return fn

    def strike(self, code: str) -> bool:
        """Record one raise; returns True when this strike disabled the source."""
        key = code_fingerprint(code)
        count = self._strikes.get(key, 0) + 1
        self._strikes[key] = count
        if count >= self._limit and key not in self._poisoned:
            self._poisoned.add(key)
            logger.warning("%s raised %d times in a row; disabled (hash=%s)", self._label, count, key)
            return True
        return False

    def clear_strikes(self, code: str) -> None:
        self._strikes[code_fingerprint(code)] = 0

    def strikes(self, code: str) -> int:
        return self._strikes.get(code_fingerprint(code), 0)

    def strike_counts(self) -> dict[str, int]:
        return {k: v for k, v in self._strikes.items() if v}

    def is_poisoned(self, code: str) -> bool:
        return code_fingerprint(code) in self._poisoned

"""Domains. Each module registers its :class:`~.base.Domain` and state DSL on import."""

from __future__ import annotations

import importlib

from .base import Domain, Environment, StepResult, Task, register_domain, registered

AVAILABLE = ("codesearch", "sql", "alfworld", "toy")


def load(name: str) -> Domain:
    """Import ``harness_programs.domains.<name>`` (registering it) and return its :class:`Domain`."""
    if name not in AVAILABLE:
        raise KeyError(f"unknown domain {name!r}; available: {', '.join(AVAILABLE)}")
    importlib.import_module(f"{__name__}.{name}")
    return registered()[name]


__all__ = ["AVAILABLE", "Domain", "Environment", "StepResult", "Task", "load", "register_domain"]

"""The RL bridge: GRPO training with the harness serving pipeline in the rollout loop.

* :mod:`.agent_loop` -- the verl agent loop (``HarnessAgentLoop``) and its verl-free episode core.
* :mod:`.traces` -- episode-to-row conversion (per-turn or merged) and instance-level GRPO.
* :mod:`.parsing` -- Hermes tool-call parsing of raw generations.
* :mod:`.step_marker` -- the trainer-to-rollout step marker and the per-step episode dump.

Importing this package never imports verl, torch or transformers; the agent loop module is loaded
on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

from .parsing import ParsedTurn, parse_tool_calls, to_assistant_message
from .step_marker import EpisodeDumpWriter, dump_path, read_global_step, read_step, write_global_step
from .traces import (
    Trace,
    TurnRecord,
    instance_grpo_advantages,
    last_trace_is_reward,
    merged_trace,
    num_mini_batch,
    per_turn_traces,
)

_LAZY = {"HarnessAgentLoop", "EpisodeRunner", "LoopSettings", "Limits", "make_rows", "task_from_row", "task_to_row"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.agent_loop"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HarnessAgentLoop", "EpisodeDumpWriter", "EpisodeRunner", "Limits", "LoopSettings", "ParsedTurn", "Trace",
    "make_rows",
    "TurnRecord", "dump_path", "instance_grpo_advantages", "last_trace_is_reward", "merged_trace", "num_mini_batch",
    "parse_tool_calls", "per_turn_traces", "read_global_step", "read_step", "task_from_row", "task_to_row",
    "to_assistant_message", "write_global_step",
]

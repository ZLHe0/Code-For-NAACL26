"""Step plumbing between the trainer and the agent loop, and the episode dump.

The steering supervisor reads training rollouts as they are produced, bucketed by training
step. The agent loop runs inside rollout workers that stock verl does not tell which step they
are serving, so the step travels through a marker file instead:

* the trainer writes the current global step to ``<run_dir>/global_step`` at the top of every
  step (:func:`write_global_step`; one line in the training loop, see ``docs/rl.md``);
* the agent loop reads it (:func:`read_global_step`) when an episode finishes and appends the
  episode, in the :class:`~harness_programs.rollout.runner.EpisodeRecord` format, to
  ``<run_dir>/rollouts/step_<N>.jsonl`` (:class:`EpisodeDumpWriter`).

A runtime that does hand the agent loop its trajectory identity (step, sample index, replicate,
validation flag) takes precedence over the marker; the marker is the fallback that needs no
change to the rollout workers. Dumps are append-only and written under an advisory file lock,
so concurrent workers on one host never interleave lines, and each file reads back with
:func:`harness_programs.rollout.read_jsonl`.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from collections.abc import Hashable
from pathlib import Path
from typing import Any

from ..rollout.runner import EpisodeRecord, read_jsonl

try:  # advisory locking is POSIX-only; elsewhere the in-process lock is all there is
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

STEP_FILE = "global_step"
ROLLOUTS_DIR = "rollouts"
UNKNOWN_STEP = "unknown"


def marker_path(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / STEP_FILE


def write_global_step(run_dir: str | os.PathLike[str], step: int) -> Path:
    """Atomically record ``step`` as the trainer's current global step."""
    path = marker_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{STEP_FILE}.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(f"{int(step)}\n")
    os.replace(tmp, path)
    return path


def read_global_step(run_dir: str | os.PathLike[str] | None) -> int | None:
    """The step last written by the trainer, or ``None`` if there is no readable marker."""
    if not run_dir:
        return None
    try:
        return int(marker_path(run_dir).read_text().strip())
    except (OSError, ValueError):
        return None


def dump_path(run_dir: str | os.PathLike[str], step: int | None) -> Path:
    """``<run_dir>/rollouts/step_<N>.jsonl``; episodes with no known step go to ``step_unknown.jsonl``."""
    name = UNKNOWN_STEP if step is None else str(int(step))
    return Path(run_dir) / ROLLOUTS_DIR / f"step_{name}.jsonl"


def rollout_meta(*, step: int | None, group_key: Any, replicate: int | None, validate: bool) -> dict[str, Any]:
    """The ``rollout`` block of an episode record: where the episode sits in training."""
    return {"step": None if step is None else int(step), "group_key": group_key,
            "replicate": None if replicate is None else int(replicate), "validate": bool(validate)}


class ReplicateCounter:
    """Numbers the episodes of one group within one step, as seen by this process.

    Used only when the runtime does not provide the replicate index itself. Rollouts of one group
    may be split across workers, so the numbering is unique per process, not per group; it is a
    label for the dump, never an input to training.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[Hashable, ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def next(self, step: int | None, group_key: Any, validate: bool) -> int:
        key = (step, json.dumps(group_key, default=str), bool(validate))
        with self._lock:
            value = self._counts[key]
            self._counts[key] = value + 1
            return value


class EpisodeDumpWriter:
    """Appends episode records to the per-step dump files under ``<run_dir>/rollouts``."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self._lock = threading.Lock()

    def write(self, record: EpisodeRecord | dict[str, Any]) -> Path:
        row = record.to_dict() if isinstance(record, EpisodeRecord) else dict(record)
        step = (row.get("rollout") or {}).get("step")
        path = dump_path(self.run_dir, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(row, default=str) + "\n").encode()
        with self._lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
        return path


def read_step(run_dir: str | os.PathLike[str], step: int | None) -> list[EpisodeRecord]:
    """Every episode dumped for ``step`` (training and validation), or ``[]`` if none yet."""
    path = dump_path(run_dir, step)
    return read_jsonl(path) if path.exists() else []


__all__ = [
    "EpisodeDumpWriter", "ROLLOUTS_DIR", "ReplicateCounter", "STEP_FILE", "dump_path", "marker_path",
    "read_global_step", "read_step", "rollout_meta", "write_global_step",
]

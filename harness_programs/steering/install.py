"""Install H_k: the helper the Gate (Algorithm 1, line 9) uses to write an accepted trigger set.

The program document is the only thing serving reads, and it reloads on change, so an install is
live from the next episode. An install

* reads the current program and checks it is for this domain,
* refuses a trigger whose name is already installed (or repeated in the request): a trigger is
  never replaced in place, so every firing record stays attributable to one source,
* preflights the complete new document with :func:`~harness_programs.serving.steer.build_runtime` --
  the same compile and smoke serving runs at load -- before anything is written,
* writes atomically (:func:`~harness_programs.serving.program.write_program`), and
* appends one row to the ledger ``<program>.installs.jsonl`` (which also keeps each trigger's
  offline fault, since the served document does not).

Nothing here removes a trigger. Uninstalling is a decision for a person, not for a round.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..serving.program import Program, ProgramError, load_program, write_program
from ..serving.sandbox import code_fingerprint
from ..serving.steer import Trigger, build_runtime

try:
    import fcntl
except ImportError:  # pragma: no cover -- non-POSIX
    fcntl = None  # type: ignore[assignment]

LEDGER_SUFFIX = ".installs.jsonl"
LOCK_SUFFIX = ".lock"


class InstallError(Exception):
    """The install was refused; the program on disk is unchanged."""


def ledger_path(program_path: str | os.PathLike[str]) -> Path:
    path = Path(program_path)
    return path.with_name(path.name + LEDGER_SUFFIX)


@contextlib.contextmanager
def _locked(program_path: Path) -> Iterator[None]:
    """Serialise installs into one program across processes (best effort off POSIX)."""
    lock = program_path.with_name(program_path.name + LOCK_SUFFIX)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_ledger(program_path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = ledger_path(program_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def installed_triggers(program_path: str | os.PathLike[str]) -> list[Trigger]:
    """The triggers serving now, with the offline fault each was installed with.

    The served program never carries faults (serving does not grade); the ledger keeps them so
    a later round can replay an incumbent on the quantity it was admitted for.
    """
    path = Path(program_path)
    if not path.exists():
        return []
    faults: dict[str, str] = {}
    for row in read_ledger(path):
        faults.update(row.get("faults") or {})
    return [replace(c, fault=faults.get(c.name, "")) for c in load_program(path).triggers]


def served(trigger: Trigger) -> Trigger:
    """The trigger as serving holds it: no offline fault."""
    return Trigger(name=trigger.name, code=trigger.code, guidance=trigger.guidance,
                   render=trigger.render)


def install(program_path: str | os.PathLike[str], domain: str, triggers: Sequence[Trigger], *,
            step: int) -> Program:
    """Append ``triggers`` to the program at ``program_path``; returns the new program.

    Raises :class:`InstallError` (leaving the file untouched) on a harness mismatch, a duplicate
    name, or a trigger that fails the serving preflight. A missing program is created for
    ``domain`` with no compression.
    """
    path = Path(program_path)
    if not triggers:
        raise InstallError("nothing to install")
    serving = [served(t) for t in triggers]
    with _locked(path):
        try:
            current = load_program(path) if path.exists() else Program(harness=domain)
        except ProgramError as exc:
            raise InstallError(f"cannot read the current program: {exc}") from exc
        if current.harness != domain:
            raise InstallError(f"program is for {current.harness!r}, installing for {domain!r}")
        names = [c.name for c in current.triggers]
        for trigger in serving:
            if trigger.name in names:
                raise InstallError(f"a trigger named {trigger.name!r} is already installed; "
                                   "installs are additive and never replace a trigger")
            names.append(trigger.name)
        updated = replace(current, triggers=[*current.triggers, *serving])
        try:
            build_runtime(updated.triggers)
        except Exception as exc:  # noqa: BLE001 -- a render may raise past build_runtime's wrap
            raise InstallError(f"preflight refused the new program: {exc}") from exc
        write_program(path, updated)
        row = {
            "step": step, "path": path.name, "harness": domain,
            "added": [c.name for c in serving],
            "serving": [c.name for c in updated.triggers],
            "fingerprints": {c.name: code_fingerprint(c.code) for c in serving},
            "faults": {c.name: c.fault for c in triggers if c.fault},
            "at": round(time.time(), 3),
        }
        with ledger_path(path).open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return updated

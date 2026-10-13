"""The program document: one JSON file holding a harness's compression program and triggers.

::

    {
      "harness": "codesearch",
      "compression": {"name": "...", "code": "def compress(state): ..."} | null,
      "triggers": [{"name": "...", "code": "def check(ep): ...", "guidance": "...", "render": "..."}],
      "contract_sha": "<hash of the state-DSL contract the compression was written against>"
    }

Stage 1 (compression optimization) writes ``compression``; Stage 2 (steering) appends to
``triggers`` while training runs. The document is re-read at the start of every episode, so an
installed trigger is live from the next episode on. Writes are atomic.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .steer import Trigger


class ProgramError(Exception):
    """The program document is malformed, or written against a different state DSL."""


def contract_sha(contract: str) -> str:
    return hashlib.sha256(contract.strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class Program:
    harness: str
    compression: dict[str, str] | None = None
    triggers: list[Trigger] = field(default_factory=list)
    contract_sha: str = ""

    @property
    def compression_code(self) -> str | None:
        return self.compression["code"] if self.compression else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "compression": self.compression,
            "triggers": [c.to_served() for c in self.triggers],
            "contract_sha": self.contract_sha,
        }


def program_from_dict(doc: dict[str, Any]) -> Program:
    if not isinstance(doc, dict) or not isinstance(doc.get("harness"), str) or not doc["harness"]:
        raise ProgramError("a program needs a non-empty string 'harness'")
    compression = doc.get("compression")
    if compression is not None:
        if not isinstance(compression, dict) or not isinstance(compression.get("code"), str):
            raise ProgramError("'compression' must be null or {name, code}")
        compression = {"name": str(compression.get("name") or "compression"), "code": compression["code"]}
    triggers = []
    for row in doc.get("triggers") or []:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("name", "code", "guidance")):
            raise ProgramError(f"each trigger needs string name/code/guidance: {row!r}")
        triggers.append(Trigger(name=row["name"], code=row["code"], guidance=row["guidance"],
                                render=str(row.get("render") or "")))
    return Program(harness=doc["harness"], compression=compression, triggers=triggers,
                   contract_sha=str(doc.get("contract_sha") or ""))


def load_program(path: str | os.PathLike[str]) -> Program:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProgramError(f"cannot read program {path}: {exc}") from exc
    return program_from_dict(doc)


def check_contract(program: Program, contract: str) -> None:
    """Refuse a compression written against a different state DSL than the one now served."""
    if program.compression is not None and program.contract_sha and program.contract_sha != contract_sha(contract):
        raise ProgramError(
            f"program was optimized against state contract {program.contract_sha}, but the "
            f"{program.harness} DSL is now {contract_sha(contract)}; re-run compression optimization")


def write_program(path: str | os.PathLike[str], program: Program) -> None:
    """Write atomically: a reader sees the old document or the new one, never a torn one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(program.to_dict(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

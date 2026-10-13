"""H0 as a program document: the output of the Gate (Algorithm 1, line 3) that training serves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..serving.compress import get_state_dsl
from ..serving.program import Program, contract_sha, load_program, write_program


def run(domain: str, compression: dict[str, Any] | None, path: str | Path) -> Program:
    """Install ``compression`` (``None`` for no compression) as the program document at ``path``.

    Triggers already in the document are kept, so stage 1 can be re-run under a live stage 2.
    """
    path = Path(path)
    triggers = []
    if path.exists():
        existing = load_program(path)
        if existing.harness == domain:
            triggers = existing.triggers
    program = Program(harness=domain, compression=compression, triggers=triggers,
                      contract_sha=contract_sha(get_state_dsl(domain).CONTRACT))
    write_program(path, program)
    return program

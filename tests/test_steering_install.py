"""Installs: additive, duplicate-refusing, preflighted, atomic, and ledgered."""

from __future__ import annotations

import json

import pytest
from test_steering_support import REOPEN_CHECK, REOPEN_FAULT, REOPEN_GUIDANCE, scratch  # noqa: F401

from harness_programs.serving.program import Program, load_program, write_program
from harness_programs.serving.steer import Trigger
from harness_programs.steering.install import (
    InstallError,
    install,
    installed_triggers,
    ledger_path,
    read_ledger,
)

REOPEN = Trigger("reopen", REOPEN_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT)
LATE = Trigger("late", "def check(ep):\n    return len(ep.actions) >= 6\n",
               "If you are past the sixth turn, stop exploring and answer with your best file.")


def test_install_creates_appends_and_ledgers(scratch):  # noqa: F811
    path = scratch / "program.json"
    install(path, "toy", [REOPEN], step=10)
    install(path, "toy", [LATE], step=20)
    program = load_program(path)
    assert program.harness == "toy"
    assert [c.name for c in program.triggers] == ["reopen", "late"]
    assert all(not c.fault for c in program.triggers)  # the served document never carries faults
    rows = read_ledger(path)
    assert [r["added"] for r in rows] == [["reopen"], ["late"]]
    assert rows[1]["serving"] == ["reopen", "late"]
    assert [r["step"] for r in rows] == [10, 20]
    assert rows[0]["faults"] == {"reopen": REOPEN_FAULT} and rows[1]["faults"] == {}
    assert rows[0]["path"] == "program.json"
    assert ledger_path(path).name == "program.json.installs.jsonl"
    triggers = {c.name: c for c in installed_triggers(path)}
    assert triggers["reopen"].fault == REOPEN_FAULT and triggers["late"].fault == ""


def test_install_keeps_the_existing_program(scratch):  # noqa: F811
    path = scratch / "program.json"
    compression = {"code": "def compress(turn):\n    return turn\n"}
    write_program(path, Program(harness="toy", compression=compression, contract_sha="abc"))
    install(path, "toy", [REOPEN], step=10)
    program = load_program(path)
    assert program.compression_code == compression["code"] and program.contract_sha == "abc"


def _unchanged(path, before):
    assert path.read_text() == before
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".") or p.suffix == ".tmp"]


def test_duplicate_names_are_refused_and_nothing_is_written(scratch):  # noqa: F811
    path = scratch / "program.json"
    install(path, "toy", [REOPEN], step=10)
    before = path.read_text()
    with pytest.raises(InstallError, match="already installed"):
        install(path, "toy", [Trigger("reopen", LATE.code, LATE.guidance)], step=20)
    with pytest.raises(InstallError, match="already installed"):
        install(path, "toy", [LATE, LATE], step=20)
    _unchanged(path, before)
    assert len(read_ledger(path)) == 1


def test_harness_mismatch_is_refused(scratch):  # noqa: F811
    path = scratch / "program.json"
    install(path, "toy", [REOPEN], step=10)
    before = path.read_text()
    with pytest.raises(InstallError, match="program is for 'toy'"):
        install(path, "sql", [LATE], step=20)
    _unchanged(path, before)


def test_preflight_failure_leaves_the_program_untouched(scratch):  # noqa: F811
    path = scratch / "program.json"
    install(path, "toy", [REOPEN], step=10)
    before = path.read_text()
    raising = Trigger("raising", "def check(ep):\n    return ep.steps[5].command\n", LATE.guidance)
    bad_render = Trigger("bad_render", LATE.code, LATE.guidance, render="def render(ep):\n    return 3\n")
    for trigger in (raising, bad_render):
        with pytest.raises(InstallError, match="preflight"):
            install(path, "toy", [trigger], step=20)
    _unchanged(path, before)
    assert [r["added"] for r in read_ledger(path)] == [["reopen"]]


def test_unreadable_program_is_refused(scratch):  # noqa: F811
    path = scratch / "program.json"
    path.write_text(json.dumps({"no": "harness"}))
    with pytest.raises(InstallError, match="cannot read"):
        install(path, "toy", [REOPEN], step=10)
    with pytest.raises(InstallError, match="nothing to install"):
        install(path, "toy", [], step=10)

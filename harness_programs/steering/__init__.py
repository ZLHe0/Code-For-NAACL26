"""Stage 2 of Algorithm 1: the steering harness, evolved alongside the policy during RL.

A *trigger* (:class:`~harness_programs.serving.steer.Trigger`) is a ``check(ep)`` program run at
serving time on the episode so far, the ``guidance`` served when it fires (optionally written by
``render(ep)``), and an offline ``fault(ep)`` that says whether the failure it targets occurred.
Every ``ROUND_EVERY`` training steps (UpdatePolicy, line 5) the supervisor runs one round of the
steer update (:mod:`.rounds`):

* line 6, ProposeSteer -- :func:`.propose.propose_steer`
* line 7, Replay -- :func:`.replay.replay_evidence`
* line 8, Admit -- :func:`.admit.judge` and :func:`.admit.arbitrate`
* line 9, Gate -- :func:`.gate.measure` and :func:`.gate.accept` (via :func:`.install.install`)

Run it with ``python -m harness_programs.steering supervise|round ...``; see ``docs/steering.md``.
"""

from __future__ import annotations

from .admit import Admission, Ruling, Verdict, arbitrate, judge, parse, report
from .gate import GateConfig, GateError, GateVerdict, accept, decide, measure
from .install import InstallError, install, installed_triggers, ledger_path, read_ledger
from .propose import Hypothesis, Proposals, propose, propose_steer
from .replay import Evidence, Replay, ReplaySet, replay_evidence, replay_many
from .reservoir import Group, build_groups, load_records, sample_groups
from .rounds import Round, RoundConfig, run_round
from .screen import behaviour_screen, dedup, static_screen
from .supervisor import ROUND_EVERY, supervise

__all__ = [
    "ROUND_EVERY", "Admission", "Evidence", "GateConfig", "GateError", "GateVerdict",
    "Group", "Hypothesis", "InstallError", "Proposals", "Replay", "ReplaySet", "Round",
    "RoundConfig", "Ruling", "Verdict", "accept", "arbitrate", "behaviour_screen", "build_groups",
    "decide", "dedup", "install", "installed_triggers", "judge", "ledger_path",
    "load_records", "measure", "parse", "propose", "propose_steer", "read_ledger", "replay_evidence",
    "replay_many", "report", "run_round", "sample_groups", "static_screen", "supervise",
]

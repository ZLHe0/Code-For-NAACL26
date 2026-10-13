"""Replay (Algorithm 1, line 7): E <- Replay(P, H_{k-1}, R_{0:k}).

Every proposed trigger, every installed trigger and every trigger held under MONITOR is executed
over all stored rollouts R_{0:k}, exactly where serving would run it: for every assistant turn the
check is evaluated on the *prefix* before that turn (no gold, status or final output), and the
fault once on the complete episode, with its status, final output and the domain's gold view.
That gives, per training step, how many episodes the failure occurred in (the fault rate) and,
as secondary information, how often the trigger would have fired (the fire rate).

Each series is the curve of one failure over training: falling, holding steady or rising. The
series of all triggers together are the replay evidence E (:func:`replay_evidence`), which Admit
reads.

Replay fails *closed*, the opposite of serving: a program that raises on any episode is dropped
whole rather than having the raise read as evidence. Behaviour screens
(:func:`.screen.behaviour_screen`) drop degenerate proposed and held triggers; installed
triggers are only recorded.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..serving.messages import Message
from ..serving.sandbox import call_with_deadline
from ..serving.steer import (
    RENDER_MAX_CHARS,
    Trigger,
    Episode,
    compile_fault,
    compile_render,
    compile_trigger,
    episode_from_messages,
    sample_episode,
)
from .reservoir import Placed
from .screen import behaviour_screen

logger = logging.getLogger(__name__)

RENDER_SAMPLES = 3
# Wall-clock ceiling on replaying one trigger over one episode (every prefix plus the fault).
EPISODE_DEADLINE_S = 10.0


class ReplayError(Exception):
    """The replay request itself is malformed (no triggers, repeated names)."""


@dataclass
class ReplayedEpisode:
    episode_id: str
    step: int
    group_key: str
    fires: int
    turns: int
    fired_turns: list[int]
    fault: bool
    status: str


@dataclass
class StepPoint:
    """One training step of a trigger's series."""

    step: int
    episodes: int
    faulted: int
    turns: int
    fires: int

    @property
    def fault_rate(self) -> float:
        return self.faulted / self.episodes if self.episodes else 0.0

    @property
    def fire_rate(self) -> float:
        return self.fires / self.turns if self.turns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fault_rate": round(self.fault_rate, 6),
                "fire_rate": round(self.fire_rate, 6)}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> StepPoint:
        return cls(**{k: int(row[k]) for k in cls.__dataclass_fields__})


@dataclass
class Replay:
    """One trigger's reading over one reservoir."""

    trigger: str
    episodes: list[ReplayedEpisode] = field(default_factory=list)
    render_calls: int = 0
    render_raised: int = 0
    render_samples: list[str] = field(default_factory=list)

    @property
    def fires(self) -> int:
        return sum(e.fires for e in self.episodes)

    @property
    def turns(self) -> int:
        return sum(e.turns for e in self.episodes)

    def series(self) -> list[StepPoint]:
        """Per training step: faults over episodes, fires over turns."""
        points: dict[int, StepPoint] = {}
        for e in self.episodes:
            point = points.setdefault(e.step, StepPoint(e.step, 0, 0, 0, 0))
            point.episodes += 1
            point.faulted += int(e.fault)
            point.turns += e.turns
            point.fires += e.fires
        return [points[s] for s in sorted(points)]


@dataclass
class ReplaySet:
    readings: dict[str, Replay] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def turn_prefixes(messages: Sequence[Message]) -> list[list[Message]]:
    """The messages before each assistant turn: what serving held when it fired triggers."""
    return [list(messages[:i]) for i, m in enumerate(messages) if m.get("role") == "assistant"]


def prefix_episodes(whole: Episode) -> list[Episode]:
    """The prefix view before each turn, built from the parsed episode (no outcome, no gold)."""
    return [Episode(task_text=whole.task_text, steps=whole.steps[:j], max_steps=whole.max_steps)
            for j in range(len(whole.steps))]


@dataclass
class _Loaded:
    trigger: Trigger
    check: Any
    fault: Any
    render: Any


def _load(trigger: Trigger, gold_sample: Any) -> _Loaded:
    """Compile and smoke one trigger; raises with the cause on any failure."""
    check = compile_trigger(trigger.code)
    empty = Episode(task_text="Sample task statement.", steps=[], max_steps=10)
    check(empty)
    check(sample_episode())
    fault = check if not trigger.fault else compile_fault(trigger.fault)
    if trigger.fault:
        whole = sample_episode(gold=gold_sample)
        whole.status = "answered"
        fault(whole)
    render = None
    if trigger.render:
        render = compile_render(trigger.render)
        if not isinstance(render(sample_episode()), str):
            raise TypeError("render did not return a str on the sample episode")
    return _Loaded(trigger, check, fault, render)


def _replay_one(loaded: _Loaded, whole: Episode, prefixes: list[Episode],
                reading: Replay) -> tuple[list[int], bool]:
    fired: list[int] = []
    for j, prefix in enumerate(prefixes):
        if loaded.check(prefix):
            fired.append(j)
            if loaded.render is not None:
                _render_once(loaded.render, prefix, reading)
    return fired, bool(loaded.fault(whole))


def _render_once(render: Any, prefix: Episode, reading: Replay) -> None:
    reading.render_calls += 1
    try:
        out = render(prefix)
    except Exception:  # noqa: BLE001 -- serving falls back to the guidance; counted, not raised
        reading.render_raised += 1
        return
    if not isinstance(out, str) or not out.strip():
        return
    out = out[:RENDER_MAX_CHARS]
    if len(reading.render_samples) < RENDER_SAMPLES and out not in reading.render_samples:
        reading.render_samples.append(out)


def replay_many(placed: Sequence[Placed], triggers: Sequence[Trigger], *,
                deadline_s: float = EPISODE_DEADLINE_S) -> ReplaySet:
    """Replay every trigger over every placed record in one pass.

    Episode parsing is shared across triggers. A trigger that fails to load, raises, or runs
    past ``deadline_s`` on any one episode is refused into ``failures`` and dropped from the
    pass; only a malformed request raises.
    """
    if not triggers:
        raise ReplayError("replay_many was given no triggers")
    names = [c.name for c in triggers]
    if len(set(names)) != len(names):
        raise ReplayError(f"duplicate trigger name(s) in {names}; the readings would be "
                          "indistinguishable")
    gold_sample = next((p.record.gold for p in placed if p.record.gold is not None), None)
    out = ReplaySet()
    active: dict[str, _Loaded] = {}
    for trigger in triggers:
        try:
            active[trigger.name] = _load(trigger, gold_sample)
        except Exception as exc:  # noqa: BLE001 -- a bad draft is refused, not fatal
            out.failures[trigger.name] = f"could not be loaded: {type(exc).__name__}: {exc}"
            continue
        out.readings[trigger.name] = Replay(trigger.name)

    for p in placed:
        if not active:
            break
        record = p.record
        parsed = episode_from_messages(record.messages, max_steps=record.max_steps)
        whole = Episode(task_text=parsed.task_text, steps=parsed.steps, max_steps=record.max_steps,
                        final_output=record.final_output or "", status=record.status or "",
                        gold=record.gold)
        prefixes = prefix_episodes(whole)
        for name in list(active):
            reading = out.readings[name]
            try:
                fired, fault = call_with_deadline(_replay_one, active[name], whole, prefixes,
                                                  reading, seconds=deadline_s)
            except Exception as exc:  # noqa: BLE001 -- fail closed on the whole trigger
                out.failures[name] = (f"raised on episode {record.episode_id}: "
                                      f"{type(exc).__name__}: {exc}")
                del active[name]
                del out.readings[name]
                continue
            reading.episodes.append(ReplayedEpisode(
                episode_id=record.episode_id, step=p.step, group_key=p.group_key,
                fires=len(fired), turns=len(prefixes), fired_turns=fired, fault=fault,
                status=record.status))
    for name, why in out.failures.items():
        logger.info("replay refused %s: %s", name, why)
    return out


# ---------------------------------------------------------------------------------------------
# The replay evidence E
# ---------------------------------------------------------------------------------------------

@dataclass
class Evidence:
    """One trigger's replay evidence: its per-step failure series."""

    trigger: str
    series: list[StepPoint]
    fires: int = 0
    turns: int = 0
    render_samples: list[str] = field(default_factory=list)

    @property
    def current_rate(self) -> float | None:
        """The fault rate at the newest replayed step."""
        return self.series[-1].fault_rate if self.series else None

    @property
    def fire_rate(self) -> float:
        return self.fires / self.turns if self.turns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"trigger": self.trigger, "series": [p.to_dict() for p in self.series],
                "fires": self.fires, "turns": self.turns,
                "render_samples": list(self.render_samples)}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Evidence:
        return cls(trigger=row["trigger"], series=[StepPoint.from_dict(p) for p in row["series"]],
                   fires=int(row.get("fires", 0)),
                   turns=int(row.get("turns", 0)), render_samples=list(row.get("render_samples") or []))


def replay_evidence(placed: Sequence[Placed], triggers: Sequence[Trigger], *,
                    installed: Collection[str] = ()) -> tuple[dict[str, Evidence], dict[str, list[str]]]:
    """The replay evidence E of ``triggers`` over ``placed``, and the triggers dropped on the way.

    A trigger that cannot be replayed is dropped with its reason; so is a degenerate one, unless
    it is ``installed`` -- an installed trigger is evidence about H_{k-1}, never a candidate.
    """
    if not triggers or not placed:
        return {}, {}
    replayed = replay_many(placed, triggers)
    dropped = {name: [why] for name, why in replayed.failures.items()}
    evidence: dict[str, Evidence] = {}
    for name, reading in replayed.readings.items():
        reasons = [] if name in installed else behaviour_screen(reading)
        if reasons:
            dropped[name] = reasons
            continue
        evidence[name] = Evidence(trigger=name, series=reading.series(), fires=reading.fires, turns=reading.turns,
                                  render_samples=list(reading.render_samples))
    return evidence, dropped

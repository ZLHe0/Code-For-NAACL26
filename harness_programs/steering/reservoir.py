"""The rollout reservoir R_{0:k}, and the mixed-outcome groups ProposeSteer reads (Algorithm 1, line 6).

The trainer writes one ``rollouts/step_<N>.jsonl`` file per training step, each line an
:class:`~harness_programs.rollout.runner.EpisodeRecord`; ``step_0.jsonl`` holds the rollouts R_0 of the
starting policy. A record is placed on the training axis by its ``rollout.step`` (falling back
to the step in its file name) and grouped by ``rollout.group_key`` -- the rollouts of one task
drawn at one step. Validation records (``rollout.validate``) are counted and skipped.

Replay reads every file (:func:`load_records`). ProposeSteer reads only the newest rollouts R_k
and, of those, only the *mixed-outcome* groups: tasks on which some rollouts scored higher than
others (:func:`build_groups`, :func:`sample_groups`). Each group is shown to the creator model as
its lower- and higher-reward episode.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..rollout.runner import EpisodeRecord, read_jsonl
from ..serving.steer import episode_from_messages
from . import prompts

ROLLOUT_DIR = "rollouts"
STEP_FILE = re.compile(r"^step_(\d+)\.jsonl$")

# Groups shown to the proposer per round.
MAX_GROUPS = 20
# Characters of each observation and command shown in a prompt.
OBSERVATION_CHARS = 400
COMMAND_CHARS = 300
REASONING_CHARS = 200

ANSWERED = "answered"
FORFEIT = "max_turns"


@dataclass
class Placed:
    """One record with its place on the training axis."""

    record: EpisodeRecord
    step: int
    group_key: str


@dataclass
class Reservoir:
    """The records a stage reads, and what was skipped."""

    placed: list[Placed]
    files: list[str]
    n_records: int = 0
    n_validation: int = 0
    n_no_identity: int = 0

    def summary(self) -> dict[str, Any]:
        return {"files": list(self.files), "n_records": self.n_records,
                "n_placed": len(self.placed), "n_validation": self.n_validation,
                "n_no_identity": self.n_no_identity}


def rollout_files(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` for every rollout dump under ``run_dir``, oldest first."""
    root = Path(run_dir) / ROLLOUT_DIR
    if not root.is_dir():
        return []
    out = []
    for path in root.iterdir():
        match = STEP_FILE.match(path.name)
        if match:
            out.append((int(match.group(1)), path))
    return sorted(out)


def latest_step(run_dir: str | Path) -> int:
    """The newest training step with a rollout dump, or ``-1`` when there is none."""
    files = rollout_files(run_dir)
    return files[-1][0] if files else -1


def load_records(run_dir: str | Path, *, up_to: int | None = None, after: int | None = None) -> Reservoir:
    """Every record in the step files with ``after < step <= up_to`` (either bound optional)."""
    run_dir = Path(run_dir)
    chosen = [(s, p) for s, p in rollout_files(run_dir)
              if (up_to is None or s <= up_to) and (after is None or s > after)]
    reservoir = Reservoir(placed=[], files=[str(p.relative_to(run_dir)) for _, p in chosen])
    for file_step, path in chosen:
        for record in read_jsonl(path):
            reservoir.n_records += 1
            identity = record.rollout or {}
            if identity.get("validate"):
                reservoir.n_validation += 1
                continue
            step = identity.get("step", file_step)
            group_key = identity.get("group_key")
            if step is None or int(step) < 0 or group_key is None:
                reservoir.n_no_identity += 1
                continue
            reservoir.placed.append(Placed(record, int(step), str(group_key)))
    return reservoir


# ---------------------------------------------------------------------------------------------
# Mixed-outcome groups
# ---------------------------------------------------------------------------------------------

@dataclass
class Group:
    """One task's rollouts at one step, and the two episodes shown from it."""

    key: str
    step: int
    task_id: str
    worst: EpisodeRecord = field(repr=False)
    best: EpisodeRecord = field(repr=False)
    scores: tuple[float, ...] = ()

    @property
    def gold(self) -> Any:
        return self.worst.gold

    def distribution(self) -> str:
        shown = ", ".join(f"{s:.2f}" for s in sorted(self.scores, reverse=True))
        mean = sum(self.scores) / len(self.scores)
        return f"{shown}   (mean {mean:.2f}, spread {max(self.scores) - min(self.scores):.2f})"

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "step": self.step, "task_id": self.task_id,
                "worst": self.worst.episode_id, "best": self.best.episode_id,
                "scores": list(self.scores)}


def rank(records: Sequence[EpisodeRecord]) -> tuple[EpisodeRecord, EpisodeRecord]:
    """The lower- and higher-reward episode of a mixed group.

    The lower is the lowest scorer, the longest among ties; the higher is the top scorer whose
    length is closest to it, so length is not the difference the model reaches for.
    """
    worst = min(records, key=lambda r: (r.reward, -r.num_steps))
    top = max(r.reward for r in records)
    best = min((r for r in records if r.reward == top),
               key=lambda r: (abs(r.num_steps - worst.num_steps), r.num_steps))
    return worst, best


def build_groups(reservoir: Reservoir) -> list[Group]:
    """Every mixed-outcome group in the reservoir (rewards differ within it), ordered by key."""
    buckets: dict[tuple[int, str], list[EpisodeRecord]] = defaultdict(list)
    for placed in reservoir.placed:
        if placed.record.reward is not None:
            buckets[(placed.step, placed.group_key)].append(placed.record)
    groups = []
    for (step, group_key), members in sorted(buckets.items()):
        if len({m.reward for m in members}) < 2:
            continue
        worst, best = rank(members)
        groups.append(Group(key=f"step{step}-{group_key}", step=step, task_id=worst.task_id,
                            worst=worst, best=best, scores=tuple(float(m.reward) for m in members)))
    return groups


def _stable_seed(groups: Sequence[Group]) -> int:
    text = f"{groups[0].key}|{groups[-1].key}|{len(groups)}"
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def sample_groups(groups: Sequence[Group], *, max_groups: int = MAX_GROUPS,
                  seed: int | None = None) -> list[Group]:
    """At most ``max_groups`` groups drawn at random, ordered by key.

    The default seed is derived from the groups, so a round is reproducible from its dump.
    """
    if len(groups) <= max_groups:
        return list(groups)
    rng = random.Random(_stable_seed(groups) if seed is None else seed)
    return sorted(rng.sample(list(groups), max_groups), key=lambda g: g.key)


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def task_text(record: EpisodeRecord) -> str:
    return episode_from_messages(record.messages).task_text


def finish_line(record: EpisodeRecord) -> str:
    if record.status == ANSWERED:
        answer = clip(record.final_output, COMMAND_CHARS) or "(empty)"
        return f"it answered after {record.num_steps} turn(s): {answer}"
    if record.status == FORFEIT:
        return f"it ran out of turns ({record.num_steps} of {record.max_steps}) without answering"
    return f"the episode ended with status {record.status!r} after {record.num_steps} turn(s)"


def _context_note(served: dict[str, Any]) -> str:
    total, kept = served.get("turns_total"), served.get("turns_kept")
    if isinstance(total, int) and isinstance(kept, int) and kept < total:
        return f"[context held only the last {kept} of the {total} earlier turns]"
    return ""


def render_episode(record: EpisodeRecord, *, label: str,
                   observation_chars: int = OBSERVATION_CHARS) -> str:
    """One episode as the proposer reads it: score, how it ended, then every turn."""
    episode = episode_from_messages(record.messages, max_steps=record.max_steps)
    lines = [f"=== {label} -- reward {record.reward:.2f} ===", f"finish: {finish_line(record)}"]
    for step in episode.steps:
        note = _context_note(record.served[step.index]) if step.index < len(record.served) else ""
        if note:
            lines.append(f"  {note}")
        if step.reasoning:
            lines.append(f"turn {step.index} thought: {clip(step.reasoning, REASONING_CHARS)}")
        lines.append(f"turn {step.index}: {clip(step.command, COMMAND_CHARS)}")
        if step.observation:
            lines.append("  -> " + clip(step.observation, observation_chars).replace("\n", "\n     "))
    return "\n".join(lines)


def render_gold(gold: Any) -> str:
    if gold is None:
        return ""
    try:
        text = json.dumps(gold, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(gold)
    return ("GOLD ANSWER (what this task wanted -- evidence for your reasoning and for the "
            f"fault only; never for the trigger or the guidance):\n{clip(text, 1000)}\n")


def render_group(group: Group, *, scale: str = prompts.DEFAULT_SCALE,
                 observation_chars: int = OBSERVATION_CHARS) -> str:
    """Everything the proposer is shown about one group."""
    return prompts.fill(prompts.USER, {
        "n_total": str(len(group.scores)),
        "scale": scale,
        "distribution": group.distribution(),
        "task": clip(task_text(group.worst), 2000),
        "gold": render_gold(group.gold),
        "worst": render_episode(group.worst, label="LOWER-REWARD EPISODE",
                                observation_chars=observation_chars),
        "best": render_episode(group.best, label="HIGHER-REWARD EPISODE",
                               observation_chars=observation_chars),
    })

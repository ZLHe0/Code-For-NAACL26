"""Shared helpers for the steering tests: a repo-local scratch dir and synthetic toy rollouts.

No test here touches the network or a system temp directory.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
from pathlib import Path

import pytest

from harness_programs.domains import load
from harness_programs.domains.toy import write_split
from harness_programs.rollout.runner import EpisodeRecord, write_jsonl
from harness_programs.serving.messages import content_of
from harness_programs.steering.gate import GateConfig

SCRATCH_ROOT = Path(__file__).resolve().parent / ".scratch_steering"
_counter = itertools.count()

REOPEN_CHECK = '''\
def check(ep):
    opened = [a for a in ep.actions if a.startswith("open ")]
    return len(opened) != len(set(opened))
'''

REOPEN_FAULT = '''\
def fault(ep):
    opened = [a for a in ep.actions if a.startswith("open ")]
    return len(opened) != len(set(opened)) and not ep.final_output
'''

REOPEN_GUIDANCE = ("If you have already opened a file, do not open it again: open a file you have "
                  "not read yet, and answer as soon as one contains the word.")

ALWAYS_CHECK = '''\
def check(ep):
    return len(ep.steps) >= 0 and ep.task_text != "never"
'''

NEVER_CHECK = '''\
def check(ep):
    return any(a.startswith("delete ") for a in ep.actions)
'''

RAISING_CHECK = '''\
def check(ep):
    return ep.steps[-1].command.startswith("open ")
'''


@pytest.fixture
def scratch(request):
    """A fresh directory under ``tests/.scratch_steering``, removed after the test."""
    name = re.sub(r"\W+", "_", request.node.name)[:60]
    path = SCRATCH_ROOT / f"{name}_{next(_counter)}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
    if SCRATCH_ROOT.exists() and not any(SCRATCH_ROOT.iterdir()):
        SCRATCH_ROOT.rmdir()


def toy():
    return load("toy")


def toy_gate(directory: Path) -> GateConfig:
    """A gate on D_hold carved out of the toy's training tasks (never its evaluation pool)."""
    _, holdout = write_split(directory / "split", n_holdout=8)
    return GateConfig(holdout=str(holdout), holdout_tasks=8, workers=1)


def play(task, actions: list[str], *, step: int, replicate: int, served=None) -> EpisodeRecord:
    """Drive the real toy environment with a fixed action script."""
    domain = toy()
    env = domain.make_env(task)
    messages = env.reset()
    status = "max_turns"
    for action in actions[:domain.max_turns]:
        assistant = {"role": "assistant", "content": f"Next step.\n{action}"}
        messages.append(assistant)
        result = env.step(assistant)
        messages.extend(result.observations)
        if result.done:
            status = "answered"
            break
    return EpisodeRecord(
        episode_id=f"{task.task_id}-s{step}-r{replicate}", harness="toy", task_id=task.task_id,
        status=status, num_steps=sum(1 for m in messages if m["role"] == "assistant"),
        max_steps=domain.max_turns, messages=messages, reward=float(env.reward()),
        final_output=env.final_output(), gold=env.gold_view(), served=served or [],
        rollout={"step": step, "group_key": task.task_id, "replicate": replicate})


def solve_script(task) -> list[str]:
    names = sorted(task.data["files"])
    gold = task.data["gold"]
    actions = ["<action>list</action>"]
    for name in names:
        actions.append(f"<action>open {name}</action>")
        if name == gold:
            break
    return actions + [f"<answer>{gold}</answer>"]


def loop_script(task) -> list[str]:
    wrong = next(n for n in sorted(task.data["files"]) if n != task.data["gold"])
    return ["<action>list</action>"] + [f"<action>open {wrong}</action>"] * 10


def guess_script(task) -> list[str]:
    wrong = next(n for n in sorted(task.data["files"]) if n != task.data["gold"])
    return ["<action>list</action>", f"<answer>{wrong}</answer>"]


def toy_records(*, steps=(10,), n_tasks: int = 4, mix=("solve", "loop", "loop", "solve")) -> list[EpisodeRecord]:
    tasks = toy().load_tasks("synthetic", split="train")[:n_tasks]
    scripts = {"solve": solve_script, "loop": loop_script, "guess": guess_script}
    out = []
    for step in steps:
        for task in tasks:
            for r, kind in enumerate(mix):
                out.append(play(task, scripts[kind](task), step=step, replicate=r))
    return out


def write_rollouts(run_dir: Path, records: list[EpisodeRecord]) -> None:
    by_step: dict[int, list[EpisodeRecord]] = {}
    for r in records:
        by_step.setdefault(int(r.rollout["step"]), []).append(r)
    for step, rows in by_step.items():
        write_jsonl(run_dir / "rollouts" / f"step_{step:06d}.jsonl", rows)


def reply(obj) -> str:
    return json.dumps(obj)


def proposal_reply(name: str = "reopened_file", *, check: str = REOPEN_CHECK,
                   fault: str = REOPEN_FAULT, guidance: str = REOPEN_GUIDANCE) -> str:
    return reply({"insight": "the agent re-opens a file it has already read instead of moving on; "
                             "the higher-reward episode opened each file once",
                  "name": name, "check": check, "fault": fault, "guidance": guidance})


_TARGET = re.compile(r"word '(\w+)'")
_LISTING = re.compile(r"^file_\w+\.txt$")


def gate_policy(messages, *, listen: bool = True):
    """A scripted policy that loops on one file unless guidance tells it to move on."""
    texts = [content_of(m) for m in messages]
    head = next((t for m, t in zip(messages, texts) if m["role"] == "user"), "")
    target = (_TARGET.search(head) or [None, ""])[1]
    actions = [t.split("\n")[-1] for m, t in zip(messages, texts) if m["role"] == "assistant"]
    observations = [t for t in texts if t.startswith("<observation>")]
    guided = listen and bool(texts) and "Guidance to follow" in texts[-1]
    listing: list[str] = []
    for obs in observations:
        lines = obs.replace("<observation>", "").replace("</observation>", "").strip().splitlines()
        if lines and all(_LISTING.match(x) for x in lines):
            listing = lines
    if not actions:
        return {"role": "assistant", "content": "List first.\n<action>list</action>"}
    last_obs = observations[-1] if observations else ""
    if actions[-1].startswith("<action>open ") and target and target in last_obs:
        name = actions[-1][len("<action>open "):-len("</action>")]
        return {"role": "assistant", "content": f"Found it.\n<answer>{name}</answer>"}
    opened = {a[len("<action>open "):-len("</action>")] for a in actions if a.startswith("<action>open ")}
    if guided:
        fresh = [n for n in listing if n not in opened]
        if fresh:
            return {"role": "assistant", "content": f"Moving on.\n<action>open {fresh[0]}</action>"}
    first = listing[0] if listing else "file_a.txt"
    return {"role": "assistant", "content": f"Read it again.\n<action>open {first}</action>"}


def deaf_policy(messages):
    """The same policy, ignoring guidance: every harness scores the same with it."""
    return gate_policy(messages, listen=False)

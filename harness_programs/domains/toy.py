"""A tiny deterministic domain for tests and the offline end-to-end example.

The agent must name the file in a small synthetic tree that contains a target word. Actions are
free text in ``<action>`` tags (``list`` or ``open NAME``); the answer goes in ``<answer>``.
Observations are user messages wrapped in ``<observation>``, and file contents are long enough
that a compression program has something to save.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serving.compress import register_state_dsl
from ..serving.messages import Message, Messages, committed_action, content_of, split_reasoning
from .base import StepResult, Task, register_domain

NAME = "toy"
MAX_TURNS = 8
SYSTEM_PROMPT = (
    "You are searching a small file tree for the one file that contains a target word.\n"
    "Act with <action>list</action> or <action>open NAME</action>; answer with <answer>NAME</answer>.\n"
    "Write one sentence of reasoning before each action."
)
_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
_WORDS = ("alpha", "bravo", "cedar", "delta", "ember", "fjord", "gamma", "harbor", "iris", "jade")


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------


def make_tasks(n: int = 16, *, n_files: int = 6, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        files = {}
        target = rng.choice(_WORDS)
        others = [w for w in _WORDS if w != target]
        names = [f"file_{chr(97 + j)}.txt" for j in range(n_files)]
        gold = rng.choice(names)
        for name in names:
            lines = [" ".join(rng.choice(others) for _ in range(8)) for _ in range(12)]
            if name == gold:
                lines[rng.randrange(len(lines))] += f" {target}"
            files[name] = "\n".join(lines)
        tasks.append(Task(task_id=f"toy-{i:03d}", prompt=f"Which file contains the word '{target}'?",
                          data={"files": files, "target": target, "gold": gold}))
    return tasks


class ToyEnv:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.answer = ""

    def reset(self) -> Messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self.task.prompt}]

    def step(self, assistant: Message) -> StepResult:
        span = committed_action(content_of(assistant))
        if span is None:
            return StepResult([_obs("No <action> or <answer> found. Use the required tags.")])
        tag = span.group(1).lower()
        body = re.sub(r"</?(action|answer)>", "", span.group(0), flags=re.IGNORECASE).strip()
        if tag == "answer":
            self.answer = body
            return StepResult([], done=True)
        files = self.task.data["files"]
        if body == "list":
            return StepResult([_obs("\n".join(sorted(files)))])
        if body.startswith("open "):
            name = body[5:].strip()
            return StepResult([_obs(files.get(name, f"No such file: {name}"))])
        return StepResult([_obs(f"Unknown action: {body}")])

    def reward(self) -> float:
        return float(self.answer == self.task.data["gold"])

    def final_output(self) -> str:
        return self.answer

    def gold_view(self) -> dict[str, Any]:
        return {"file": self.task.data["gold"], "word": self.task.data["target"]}


def _obs(text: str) -> Message:
    return {"role": "user", "content": f"<observation>{text}</observation>"}


@dataclass
class ToyDomain:
    name: str = NAME
    max_turns: int = MAX_TURNS
    tools: Any = None

    def load_tasks(self, path: str, *, split: str = "train", limit: int | None = None) -> list[Task]:
        """Read tasks from a JSONL file of ``{task_id, prompt, data}``, or synthesize when ``path`` is ``"synthetic"``.

        ``"synthetic"`` is two disjoint pools: the training pool under ``split="train"``, and the
        evaluation pool under any other split. D_hold is carved out of the training pool
        (:func:`write_split`), never taken from the evaluation pool.
        """
        if path == "synthetic":
            tasks = make_tasks(seed=0 if split == "train" else 1)
        else:
            rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
            tasks = [Task(task_id=r["task_id"], prompt=r["prompt"], data=r["data"]) for r in rows]
        return tasks[:limit] if limit else tasks

    def task_key(self, task: Task) -> str:
        """The whole task: prompts repeat across tasks, the file trees do not."""
        return json.dumps([task.prompt, task.data], sort_keys=True)

    def make_env(self, task: Task) -> ToyEnv:
        return ToyEnv(task)


def write_split(directory: str | Path, n_holdout: int = 8, *, seed: int = 0) -> tuple[Path, Path]:
    """Carve the toy's D_hold out of its training pool, as ``python -m harness_programs.splits carve`` does.

    Writes the synthetic training pool to ``<directory>/train.jsonl`` and returns
    ``(train_policy.jsonl, train_holdout.jsonl)``: the tasks the policy trains on and the held-out
    training tasks the gates score on. The evaluation pool (``"synthetic"`` loaded under any split
    but ``train``) is a different set of tasks, and the carve checks D_hold against it.
    """
    from ..splits import carve_file, write_tasks

    domain = ToyDomain()
    pool = write_tasks(Path(directory) / "train.jsonl", domain.load_tasks("synthetic", split="train"))
    return carve_file(domain, pool, n_holdout, seed=seed, evaluation=["synthetic"])


# ---------------------------------------------------------------------------------------------
# State DSL
# ---------------------------------------------------------------------------------------------


@dataclass
class Turn:
    index: int
    n_turns: int
    action: str
    observation: str
    reasoning: str = ""
    digest: str | None = None
    _msg_idx: int = -1
    _ai_idx: int = -1
    _raw: str = ""
    _orig: str = ""
    _orig_reasoning: str = ""


CONTRACT = '''\
You write a Python function:

    def compress(state):
        """Transform the observation state of one file-search episode and
        return the transformed list of turns."""
        ...
        return state

`state` is a list[Turn], oldest first. Each Turn has:
    .index        int    0-based turn position (0 = oldest)
    .n_turns      int    number of turns in the episode so far
    .action       str    the action that produced this observation (read-only)
    .observation  str    the observation body (a file listing or a file's contents)
    .reasoning    str    the rationale the agent wrote before this action
    .digest       str|None   set this to replace the whole observation with a short string

You MAY: rewrite .observation, rewrite or clear .reasoning, set .digest on older turns.
You MUST return a list[Turn]. You may not fabricate observations.

How your output is served:

  * You are called ONCE per model turn, on the WHOLE episode so far. After you return, a
    second stage drops whole OLDEST turns until the prompt fits the token budget, so the
    more you compress, the more turns survive.
  * You cannot remove a turn; leaving one out of the list serves it unchanged.
  * The LAST turn in `state` is the observation the agent must answer RIGHT NOW.

The task statement is not in `state` and cannot be modified.

    # naive: keep everything
    def compress(state):
        return state
'''


def is_observation(message: Message) -> bool:
    return message.get("role") == "user" and "<observation>" in content_of(message)


def parse_turn(*, obs_message: Message, ai_message: Message | None, msg_idx: int, ai_idx: int,
               explicit_carrier: bool) -> Turn:
    raw = content_of(obs_message)
    match = _OBS_RE.search(raw)
    body = match.group(1) if match else raw
    reasoning, action = "", ""
    if ai_message is not None:
        reasoning, action = split_reasoning(ai_message, explicit_carrier=explicit_carrier)
    return Turn(index=0, n_turns=0, action=re.sub(r"</?action>", "", action).strip(), observation=body,
                reasoning=reasoning, _msg_idx=msg_idx, _ai_idx=ai_idx, _raw=raw, _orig=body,
                _orig_reasoning=reasoning)


def render_turn(turn: Turn) -> str:
    if turn.digest is None and turn.observation == turn._orig:
        return turn._raw
    return f"<observation>{turn.digest if turn.digest is not None else turn.observation}</observation>"


def rebuild_obs(message: Message, text: str, turn: Turn) -> Message:
    del turn
    return {**message, "content": text}


# -- stage-1 optimizer hooks (see docs/domains.md) ---------------------------------------------

DESC = ("A file-search agent: it lists a small tree, opens files one at a time, and must answer with "
        "the one file that contains a target word. Every observation is a listing or a whole file.")
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def label(turn: Turn) -> str:
    return f"{turn.action or '?':<24} {len(turn.observation):>5} chars"


def turn_text(turn: Turn) -> str:
    return f"{turn.action}\n{turn.observation}"


def gold_strings(gold: Any) -> list[str]:
    return [gold["word"]] if isinstance(gold, dict) and gold.get("word") else []


def parse_actions(content: str) -> list[str]:
    return [a.strip() for a in _ACTION_RE.findall(content)]


def sample_messages() -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Which file contains the word 'jade'?"},
        {"role": "assistant", "content": "List the tree first.\n<action>list</action>"},
        _obs("file_a.txt\nfile_b.txt"),
        {"role": "assistant", "content": "Open the first file.\n<action>open file_a.txt</action>"},
        _obs("alpha bravo cedar\ndelta ember fjord"),
    ]


register_state_dsl(NAME, sys.modules[__name__])
register_domain(ToyDomain())

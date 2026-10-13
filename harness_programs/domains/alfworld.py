"""ALFWorld: complete a household task in a text world by emitting free-text actions.

Every turn the agent sees one user message holding the full step prompt: the environment's
feedback, the admissible-action menu, and an instruction footer. It answers with
``<action>go to shelf 1</action>``. The first step prompt states the task and is the pinned
head of the transcript. The reward is binary: 1.0 iff the game is won.

The ``alfworld`` package (and its ``textworld`` dependency) is optional and imported only when
an episode starts. Games are read from ``game.tw-pddl`` files under an ALFWorld data
directory; nothing here downloads data.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serving.compress import register_state_dsl
from ..serving.messages import Message, Messages, content_of, split_reasoning, strip_think
from .base import StepResult, Task, register_domain

logger = logging.getLogger(__name__)

NAME = "alfworld"
MAX_TURNS = 50
DATA_ENV = "ALFWORLD_DATA"


# ---------------------------------------------------------------------------------------------
# Step prompt and action parsing
# ---------------------------------------------------------------------------------------------

ALFWORLD_NO_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are:
[
{admissible_actions}
].

{guidance}Now it's your turn to take an action.
Provide your chosen action, and only that action, enclosed in <action> </action> tags. Do not add any other text.
""".strip()

# Identical, except it asks for one sentence of plain-text reasoning before the action.
ALFWORLD_NO_HISTORY_REASON = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are:
[
{admissible_actions}
].

{guidance}Now it's your turn to take an action.
First, briefly explain your reasoning in one sentence. Then, provide your chosen action enclosed in <action> </action> tags.
""".strip()

ALFWORLD_OBS_ONLY = "Your observation is: {current_observation}"

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPECIALS_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>")


def format_admissible(commands: list[str]) -> str:
    """The admissible commands as the quoted, newline-joined menu block."""
    return "\n ".join(f"'{c}'" for c in commands)


def render_menu(commands: list[str]) -> str:
    return "Your admissible actions of the current situation are:\n" + f"[\n{format_admissible(commands)}\n]."


def build_step_prompt(observation: str, commands: list[str], *, reason: bool = True, guidance: str = "") -> str:
    template = ALFWORLD_NO_HISTORY_REASON if reason else ALFWORLD_NO_HISTORY
    return template.format(current_observation=observation, admissible_actions=format_admissible(commands),
                           guidance=guidance)


def parse_action(text: str) -> tuple[str, bool]:
    """``(action, valid)``: the first ``<action>`` body outside ``<think>``, lowercased.

    An empty action reaches the environment as a no-op. ``valid`` also requires the whole reply
    to be free of CJK characters. Validity is not admissibility: the environment rejects an
    action that is not in its menu with an observation.
    """
    raw = text or ""
    match = _ACTION_RE.search(strip_think(raw))
    if match is None:
        return "", False
    action = match.group(1).strip().lower()
    return action, bool(action) and _CJK_RE.search(raw) is None


def is_clean_action(text: str) -> bool:
    """Whether the whole reply is exactly one non-empty ``<action>`` span, and nothing else."""
    raw = strip_think(_SPECIALS_RE.sub("", text or "")).strip()
    if _CJK_RE.search(text or "") is not None or len(_ACTION_RE.findall(raw)) != 1:
        return False
    match = _ACTION_RE.fullmatch(raw)
    return match is not None and bool(match.group(1).strip())


def admissible_commands(info: dict[str, Any]) -> list[str]:
    """The menu from an unbatched info dict, without the ``help`` meta-command."""
    return [c for c in info.get("admissible_commands") or [] if c != "help"]


# ---------------------------------------------------------------------------------------------
# The game: one alfworld text environment bound to one game file
# ---------------------------------------------------------------------------------------------

# Which dataset key alfworld reads for each ``train_eval`` split.
SPLIT_DATA_KEY = {
    "train": "data_path",
    "eval_in_distribution": "eval_id_data_path",
    "eval_out_of_distribution": "eval_ood_data_path",
}

# A game load (PDDL translation and grammar parse) fails transiently often enough to retry.
_RESET_MAX_ATTEMPTS = 4

_EXPERT_GUARD_INSTALLED = False


def default_config(data_dir: str) -> dict[str, Any]:
    """The text-only environment config, with paths under an ALFWorld data directory."""
    root = Path(data_dir)
    return {
        "dataset": {
            "data_path": str(root / "json_2.1.1" / "train"),
            "eval_id_data_path": str(root / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(root / "json_2.1.1" / "valid_unseen"),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {"domain": str(root / "logic" / "alfred.pddl"), "grammar": str(root / "logic" / "alfred.twl2")},
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "visdom": False,
            "task": "alfred",
            "training_method": "dagger",
            "save_path": "./training/",
            "observation_pool_capacity": 3,
            "hide_init_receptacles": False,
            "training": {"batch_size": 1},
            "evaluate": {"run_eval": True},
        },
        # Read by the environment's init; the episode loop enforces its own turn cap.
        "dagger": {
            "action_space": "generation",
            "max_target_length": 20,
            "beam_width": 10,
            "generate_top_k": 5,
            "unstick_by_beam_search": False,
            "training": {"max_nb_steps_per_episode": 50},
            "fraction_assist": {"fraction_assist_anneal_episodes": 50000, "fraction_assist_anneal_from": 1.0,
                                "fraction_assist_anneal_to": 0.01},
            "fraction_random": {"fraction_random_anneal_episodes": 0, "fraction_random_anneal_from": 0.0,
                                "fraction_random_anneal_to": 0.0},
            "replay": {"replay_memory_capacity": 500000, "update_per_k_game_steps": 5, "replay_batch_size": 64,
                       "replay_sample_history_length": 4, "replay_sample_update_from": 2},
        },
    }


def load_config(config_path: str | None = None, data_dir: str | None = None) -> dict[str, Any]:
    """A YAML config file (environment variables expanded), else :func:`default_config`.

    The data directory falls back to the ``ALFWORLD_DATA`` environment variable.
    """
    if config_path:
        import yaml

        text = os.path.expandvars(Path(config_path).read_text(encoding="utf-8"))
        if f"${DATA_ENV}" in text:
            raise RuntimeError(f"{config_path} references ${DATA_ENV}, which is not set")
        return yaml.safe_load(text)
    data_dir = data_dir or os.environ.get(DATA_ENV)
    if not data_dir:
        raise RuntimeError(f"no ALFWorld data directory: pass data_dir, a config_path, or set ${DATA_ENV}")
    return default_config(data_dir)


def install_expert_index_guard() -> None:
    """Guard the handcoded expert against an upstream ``IndexError`` on an empty object list.

    The expert's plan is informational and the game state has already advanced when it is
    gathered, so the episode continues with an empty plan. Only the train split runs it.
    """
    global _EXPERT_GUARD_INSTALLED
    if _EXPERT_GUARD_INSTALLED:
        return
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredExpert
    except ImportError:
        return
    original = AlfredExpert._gather_infos

    def safe_gather_infos(self: Any) -> Any:
        try:
            return original(self)
        except IndexError:
            try:
                self.state["extra.expert_plan"] = []
            except Exception:  # noqa: BLE001
                pass
            logger.error("expert plan raised IndexError; continuing with an empty plan\n%s", traceback.format_exc())
            return None

    AlfredExpert._gather_infos = safe_gather_infos
    _EXPERT_GUARD_INSTALLED = True


class AlfworldGame:
    """One batched (size 1) alfworld environment pinned to a single game, unbatched.

    The game is selected by pointing the split's data path at the game's own directory, so the
    environment's directory scan finds exactly that game instead of the whole split.
    """

    def __init__(self, config: dict[str, Any], *, game_file: str, split: str = "train", seed: int = 0) -> None:
        if split not in SPLIT_DATA_KEY:
            raise ValueError(f"unknown alfworld split {split!r}; known: {sorted(SPLIT_DATA_KEY)}")
        if not os.path.exists(game_file):
            raise FileNotFoundError(f"alfworld game file not found: {game_file}")
        install_expert_index_guard()
        self._config = config
        self._game_file = game_file
        self._split = split
        self._seed = seed
        self._build()

    def _build(self) -> None:
        from alfworld.agents.environment import get_environment

        config = copy.deepcopy(self._config)
        config["dataset"][SPLIT_DATA_KEY[self._split]] = os.path.dirname(self._game_file)
        base = get_environment(config["env"]["type"])(config, train_eval=self._split)
        if list(base.game_files) != [self._game_file]:
            raise RuntimeError(f"pinning the {self._split} data path to {os.path.dirname(self._game_file)!r} "
                               f"selected {list(base.game_files)!r}, not [{self._game_file!r}]")
        self._env = base.init_env(batch_size=1)
        self._env.seed(self._seed)

    @staticmethod
    def _unbatch(info: dict[str, Any]) -> dict[str, Any]:
        return {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v) for k, v in info.items()}

    def reset(self) -> tuple[str, dict[str, Any]]:
        """``(observation, info)``; a failed game load is retried on a rebuilt environment."""
        last: Exception | None = None
        for attempt in range(_RESET_MAX_ATTEMPTS):
            try:
                obs, info = self._env.reset()
                return obs[0], self._unbatch(info)
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning("reset failed on %s (attempt %d/%d): %r", self._game_file, attempt + 1,
                               _RESET_MAX_ATTEMPTS, exc)
                if attempt + 1 < _RESET_MAX_ATTEMPTS:
                    try:
                        self._build()
                    except Exception:  # noqa: BLE001 -- the reset error is the one to report
                        pass
        assert last is not None
        raise last

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        """``(observation, done, info)``; the outcome is ``info['won']``."""
        obs, _scores, dones, info = self._env.step([action])
        return obs[0], bool(dones[0]), self._unbatch(info)


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------


class AlfworldEnv:
    def __init__(self, task: Task, *, reason: bool = True, config_path: str | None = None,
                 data_dir: str | None = None) -> None:
        self.task = task
        self.game_file = str(task.data["game_file"])
        self.split = str(task.data.get("split") or "train")
        self.seed = int(task.data.get("seed", 0))
        self.reason = reason
        self.config_path = config_path
        self.data_dir = task.data.get("data_dir") or data_dir
        self.game: AlfworldGame | None = None
        self.won = False
        self.status = "max_steps"
        self.invalid_actions = 0
        self.clean_actions = 0
        self.last_action = ""

    def reset(self) -> Messages:
        config = load_config(self.config_path, self.data_dir)
        self.game = AlfworldGame(config, game_file=self.game_file, split=self.split, seed=self.seed)
        observation, info = self.game.reset()
        return [{"role": "user", "content": build_step_prompt(observation, admissible_commands(info),
                                                              reason=self.reason)}]

    def step(self, assistant: Message) -> StepResult:
        if self.game is None:
            raise RuntimeError("step() before reset()")
        text = content_of(assistant)
        action, valid = parse_action(text)
        self.invalid_actions += int(not valid)
        self.clean_actions += int(is_clean_action(text))
        self.last_action = action
        try:
            observation, done, info = self.game.step(action)
        except Exception:  # noqa: BLE001 -- a mid-episode failure keeps the turns so far
            logger.error("step failed on %s", self.game_file, exc_info=True)
            self.status = "step_failed"
            return StepResult([], done=True, info={"status": self.status})
        self.won = bool(info.get("won", False))
        if done:
            self.status = "win" if self.won else "env_done"
            return StepResult([], done=True, info={"status": self.status})
        prompt = build_step_prompt(observation, admissible_commands(info), reason=self.reason)
        return StepResult([{"role": "user", "content": prompt}], info={"status": "running"})

    def reward(self) -> float:
        return float(self.won)

    def final_output(self) -> str:
        return self.last_action

    def gold_view(self) -> dict[str, Any]:
        return {"won": self.won, "status": self.status, "game_file": self.game_file,
                "num_invalid_actions": self.invalid_actions, "num_clean_actions": self.clean_actions}


def _resolve(base: Path, value: Any) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


@dataclass
class AlfworldDomain:
    name: str = NAME
    max_turns: int = MAX_TURNS
    tools: Any = None
    reason: bool = True
    config_path: str | None = None
    data_dir: str | None = None

    def load_tasks(self, path: str, *, split: str = "train", limit: int | None = None) -> list[Task]:
        """Read a JSONL of ``{task_id, prompt, data: {game_file, split?}}``; see docs/domains.md.

        ``game_file`` (and ``data_dir`` if given) resolve against the JSONL's directory unless
        absolute. A row without ``split`` takes ``split`` when it names an alfworld split.
        """
        base = Path(path).resolve().parent
        tasks = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            data = dict(row["data"])
            data["game_file"] = _resolve(base, data["game_file"])
            if data.get("data_dir"):
                data["data_dir"] = _resolve(base, data["data_dir"])
            if "split" not in data and split in SPLIT_DATA_KEY:
                data["split"] = split
            tasks.append(Task(task_id=str(row["task_id"]), prompt=row.get("prompt", ""), data=data))
        return tasks[:limit] if limit else tasks

    def task_key(self, task: Task) -> str:
        """The game: its task-type directory, trial directory and file, wherever the data is mounted."""
        return "/".join(Path(task.data["game_file"]).parts[-3:])

    def make_env(self, task: Task) -> AlfworldEnv:
        return AlfworldEnv(task, reason=self.reason, config_path=self.config_path, data_dir=self.data_dir)


# ---------------------------------------------------------------------------------------------
# State DSL
#
# One Turn per step prompt, split into the two handles that cost tokens: the environment's
# feedback and the admissible-action menu it reprints every step. The <action> span lives on
# the assistant message and is never rewritten. Turn 0 is the first user message, i.e. the
# pinned head: it renders verbatim, and a program that edits it is rejected.
# ---------------------------------------------------------------------------------------------

STEP_OBS_RE = re.compile(
    r"Your current observation is: (.*?)\n"
    r"Your admissible actions of the current situation are:",
    re.DOTALL,
)
MENU_RE = re.compile(r"admissible actions of the current situation are:\s*\[(.*?)\]\.", re.DOTALL | re.IGNORECASE)
TASK_RE = re.compile(r"your task is to:?\s*(.+?)(?:\n|$)", re.IGNORECASE)
NOTHING_RE = re.compile(r"^\s*nothing happens\.?\s*$", re.IGNORECASE)
_QUOTED_RE = re.compile(r"'([^']*)'")


def action_type(action: str) -> str:
    """Coarse verb class of an action ('' when there is none)."""
    a = action.strip().lower()
    if not a:
        return ""
    if a.startswith("go "):
        return "navigate"
    if a.startswith("take "):
        return "take"
    if a.startswith("put ") or (a.startswith("move ") and " to " in a):
        return "put"
    if a.startswith("open "):
        return "open"
    if a.startswith("close "):
        return "close"
    if a.startswith(("use ", "toggle ", "turn on", "turn off")):
        return "use"
    if a.startswith(("examine ", "look", "inventory")):
        return "examine"
    if a.startswith("clean "):
        return "clean"
    if a.startswith("heat "):
        return "heat"
    if a.startswith("cool "):
        return "cool"
    return "other"


@dataclass
class Turn:
    index: int                 # 0-based (0 = oldest)
    n_turns: int
    action: str                # free-text action taken before this observation
    action_type: str           # coarse verb class (navigate/take/put/open/...)
    feedback: str              # the environment's response text
    admissible: list[str]      # the admissible-action menu (rewrite or clear to compress)
    task: str                  # the episode goal, stated once in turn 0
    nothing_happened: bool     # True if the action had no effect
    reasoning: str = ""        # rationale written before the action ('' on a no-reason run)
    digest: str | None = None  # if set, replaces the whole observation
    _msg_idx: int = -1
    _ai_idx: int = -1
    _raw: str = ""             # the original message text, re-emitted when untouched
    _orig: tuple[Any, ...] = ()  # (feedback, admissible) as parsed, to detect an edit
    _orig_reasoning: str = ""


CONTRACT = '''\
You write a Python function:

    def compress(state):
        """Transform the observation state of one ALFWorld episode and
        return the transformed list of turns."""
        ...
        return state

`state` is a list[Turn], oldest first. Each Turn has:
    .index        int    0-based turn position (0 = oldest)
    .n_turns      int    number of turns in the episode so far
    .action       str    the free-text action taken before this observation
    .action_type  str    coarse verb class: navigate/take/put/open/close/use/examine/clean/heat/cool/other
    .feedback     str    the environment's response text (the observation body)
    .admissible   list[str]  the admissible-action menu shown for this turn
    .task         str    the episode goal
    .nothing_happened  bool  True if the action had no effect
    .reasoning    str    the rationale the agent wrote before this action ('' if the run
                         did not ask for reasoning); shortening it does not affect the action
    .digest       str|None   set this to replace the whole observation with a short string

You MAY: rewrite .feedback, rewrite or clear .admissible, rewrite or clear .reasoning,
set .digest on older turns, keep recent turns full. You MUST return a list[Turn]. You may
not fabricate feedback.

How your output is served -- three facts about the runtime, not advice:

  * You are called ONCE per model turn, on the WHOLE episode so far. `.n_turns` is the
    total number of turns, not a window size. After you return, a second stage measures
    the result and, if it still exceeds the token budget, drops whole OLDEST turns until
    it fits. So compressing a turn that stage was going to drop anyway saves nothing the
    agent can see; what buys context is making the turns it KEEPS cheaper.
  * The LAST turn in `state` is the observation the agent must answer RIGHT NOW. It is the
    live question, not history. Compressing it does not save history cost; it edits the
    thing being asked.
  * Turn 0 is the pinned head of this episode: its feedback states the task, it is never
    dropped, and editing it is REJECTED -- the token budget is measured on it, so rewriting
    it would redefine the budget. Start from turn 1.

The menu in `.admissible` is reprinted in full on every turn and is by far the largest
repeated cost in this family; the feedback beside it is usually a single sentence.

    # naive: keep everything
    def compress(state):
        return state

    # syntax demo (NOT a strategy): stub the oldest editable turn
    def compress(state):
        for t in state:
            if t.index == 1:
                t.digest = 'turn 1'
        return state
'''


def _split_obs(content: str) -> tuple[str, list[str]]:
    """A step prompt -> ``(feedback, admissible menu)``."""
    match = STEP_OBS_RE.search(content)
    feedback = match.group(1).strip() if match else content.strip()
    menu = MENU_RE.search(content)
    return feedback, (_QUOTED_RE.findall(menu.group(1)) if menu else [])


def is_observation(message: Message) -> bool:
    return message.get("role") == "user" and bool(STEP_OBS_RE.search(content_of(message)))


def parse_turn(*, obs_message: Message, ai_message: Message | None, msg_idx: int, ai_idx: int,
               explicit_carrier: bool) -> Turn:
    raw = content_of(obs_message)
    feedback, admissible = _split_obs(raw)
    action, reasoning = "", ""
    if ai_message is not None:
        reasoning, _ = split_reasoning(ai_message, explicit_carrier=explicit_carrier)
        action, _valid = parse_action(content_of(ai_message))
    return Turn(index=0, n_turns=0, action=action, action_type=action_type(action), feedback=feedback,
                admissible=admissible, task="", nothing_happened=bool(NOTHING_RE.match(feedback.strip())),
                reasoning=reasoning, _msg_idx=msg_idx, _ai_idx=ai_idx, _raw=raw,
                _orig=(feedback, tuple(admissible)), _orig_reasoning=reasoning)


def finalize_turns(turns: list[Turn]) -> list[Turn]:
    """Carry the goal, stated once in turn 0, onto every turn."""
    task = ""
    if turns:
        match = TASK_RE.search(turns[0].feedback)
        task = match.group(1).strip() if match else ""
    for turn in turns:
        turn.task = task
    return turns


def _resplice(raw: str, turn: Turn) -> str | None:
    """Substitute the two handles back into the original step prompt, keeping everything else.

    ``None`` when an anchor is missing or the anchors overlap, so the caller can fall back.
    """
    obs = STEP_OBS_RE.search(raw)
    menu = MENU_RE.search(raw)
    if obs is None or menu is None:
        return None
    edits = sorted(((obs.start(1), obs.end(1), turn.feedback),
                    (menu.start(1), menu.end(1), f"\n{format_admissible(turn.admissible)}\n")),
                   key=lambda edit: edit[0])
    out: list[str] = []
    cursor = 0
    for start, end, replacement in edits:
        if start < cursor:
            return None
        out.append(raw[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(raw[cursor:])
    return "".join(out)


def render_turn(turn: Turn) -> str:
    """Verbatim when untouched.

    An edited LIVE turn is spliced back into its original prompt, so the instruction footer that
    asks for the ``<action>`` tag survives. An edited past turn is rebuilt from the handles and
    relabelled, since "your current observation" is true only of the last turn.
    """
    if turn.digest is not None:
        return turn.digest
    if turn._raw and (turn.feedback, tuple(turn.admissible)) == turn._orig:
        return turn._raw
    last = turn.index == turn.n_turns - 1
    if last and turn._raw:
        spliced = _resplice(turn._raw, turn)
        if spliced is not None:
            return spliced
    lines = [f"Your current observation is: {turn.feedback}" if last
             else ALFWORLD_OBS_ONLY.format(current_observation=turn.feedback)]
    if turn.admissible:
        lines.append(render_menu(turn.admissible))
    return "\n".join(lines)


def rebuild_obs(message: Message, text: str, turn: Turn) -> Message:
    del turn
    return {**message, "content": text}


def sample_messages() -> Messages:
    menu = ["go to shelf 1", "take mug 1"]

    def obs(feedback: str) -> Message:
        return {"role": "user", "content": build_step_prompt(feedback, menu, reason=True)}

    return [
        obs("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room.\n\n"
            "Your task is to: put a mug in the sink."),
        {"role": "assistant", "content": "I should look around first.\n<action>go to shelf 1</action>"},
        obs("You arrive at shelf 1. On it you see a mug 1."),
        {"role": "assistant", "content": "Take the mug.\n<action>take mug 1</action>"},
        obs("You pick up the mug 1 from the shelf 1."),
    ]


# ---------------------------------------------------------------------------------------------
# Stage-1 optimizer hooks
# ---------------------------------------------------------------------------------------------

DESC = (
    "An embodied agent completes a household task by emitting free-text actions (go to / open / "
    "take / heat / put). Each observation is one line of environment feedback plus the full "
    '"admissible actions" menu -- dozens of candidate commands reprinted near-verbatim every '
    "turn. The goal is stated once, in the first observation.")

PINS_TURN0 = True


def parse_actions(content: str) -> list[str]:
    action, valid = parse_action(content)
    return [action] if valid else []


def label(turn: Turn) -> str:
    feedback = re.sub(r"\s+", " ", turn.feedback)[:42]
    return f'{turn.action_type or "-"}:"{turn.action[:24]}" -> {feedback!r} [{len(turn.admissible)} menu]'


def turn_text(turn: Turn) -> str:
    return turn.feedback


register_state_dsl(NAME, sys.modules[__name__])
register_domain(AlfworldDomain())

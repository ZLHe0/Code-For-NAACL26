"""ALFWorld domain: state DSL (pinned head, live-turn resplice), action parsing, episode logic.

The real environment needs the optional ``alfworld`` package and its game data; that test is
skipped without them. Everything else runs against a scripted stand-in for the game.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness_programs.domains import alfworld as aw
from harness_programs.domains.base import Task
from harness_programs.serving.compress import (NOOP_PROGRAM, CompressionError, HeadRewritten, apply_program, build_stage1,
                                        compile_program, parse_state)

MENU_PROGRAM = """
def compress(state):
    for t in state[1:]:
        t.reasoning = ''
        if t.index < t.n_turns - 1:
            t.admissible = []
        else:
            t.admissible = t.admissible[:1]
    return state
"""


def _apply(code: str, messages: list[dict]) -> list[dict]:
    return apply_program(compile_program(code), messages, aw)


# -- state DSL --------------------------------------------------------------------------------


def test_noop_is_identity():
    msgs = aw.sample_messages()
    assert build_stage1(aw.NAME, NOOP_PROGRAM)(msgs) == aw.sample_messages()


def test_parse_state():
    turns = parse_state(aw.sample_messages(), aw)
    assert len(turns) == 3
    assert turns[0]._ai_idx == -1 and turns[0].action == ""
    assert [t.action for t in turns[1:]] == ["go to shelf 1", "take mug 1"]
    assert [t.action_type for t in turns[1:]] == ["navigate", "take"]
    assert all(t.task == "put a mug in the sink." for t in turns)
    assert turns[2].admissible == ["go to shelf 1", "take mug 1"]
    assert turns[1].feedback == "You arrive at shelf 1. On it you see a mug 1."
    assert turns[1].reasoning == "I should look around first."


def test_menu_program_changes_only_observations_and_reasoning():
    msgs = aw.sample_messages()
    out = build_stage1(aw.NAME, MENU_PROGRAM)(msgs)
    assert len(out) == len(msgs) and out[0] == msgs[0]
    for i in (1, 3):
        assert aw.parse_action(out[i]["content"]) == aw.parse_action(msgs[i]["content"])
        assert out[i]["content"].startswith("<action>")
    past = out[2]["content"]
    assert past == "Your observation is: You arrive at shelf 1. On it you see a mug 1."
    live = out[4]["content"]
    assert live == msgs[4]["content"].replace("\n 'take mug 1'", "")
    assert "Then, provide your chosen action enclosed in <action> </action> tags." in live


def test_editing_turn_zero_is_rejected():
    code = "def compress(state):\n    state[0].digest = 'gone'\n    return state\n"
    with pytest.raises(HeadRewritten):
        _apply(code, aw.sample_messages())
    with pytest.raises(HeadRewritten):
        build_stage1(aw.NAME, code)
    assert issubclass(HeadRewritten, CompressionError)


def test_contract_demo_is_loadable():
    demo = aw.CONTRACT.split("# syntax demo (NOT a strategy): stub the oldest editable turn\n")[1]
    code = "\n".join(line[4:] for line in demo.splitlines())
    out = build_stage1(aw.NAME, code)(aw.sample_messages())
    assert out[2]["content"] == "turn 1"


def test_resplice_falls_back_without_anchors():
    turn = parse_state(aw.sample_messages(), aw)[-1]
    assert aw._resplice("no anchors here", turn) is None
    turn._raw = "unparseable"
    turn.feedback = "edited"
    assert aw.render_turn(turn).startswith("Your current observation is: edited\nYour admissible actions")


# -- action parsing ---------------------------------------------------------------------------


def test_parse_action_and_cleanliness():
    assert aw.parse_action("I go.\n<action> Go To Shelf 1 </action>") == ("go to shelf 1", True)
    assert aw.parse_action("<think><action>a</action></think><action>b</action>") == ("b", True)
    assert aw.parse_action("no tags") == ("", False)
    assert aw.parse_action("<action>去</action>") == ("去", False)
    assert aw.is_clean_action("<action>take mug 1</action><|im_end|>")
    assert not aw.is_clean_action("Because.\n<action>take mug 1</action>")
    assert not aw.is_clean_action("<action>a</action><action>b</action>")
    assert aw.parse_actions("<action>open fridge 1</action>") == ["open fridge 1"]
    assert aw.parse_actions("") == []


def test_action_type():
    cases = {"go to desk 1": "navigate", "move mug 1 to sink 1": "put", "put mug 1 in/on sink 1": "put",
             "use desklamp 1": "use", "look": "examine", "heat egg 1 with microwave 1": "heat", "": "",
             "jump": "other"}
    assert {a: aw.action_type(a) for a in cases} == cases


def test_admissible_menu_format():
    assert aw.admissible_commands({"admissible_commands": ["look", "help", "go to desk 1"]}) == ["look", "go to desk 1"]
    assert aw.render_menu(["a", "b"]) == "Your admissible actions of the current situation are:\n[\n'a'\n 'b'\n]."


# -- episode logic, on a scripted game --------------------------------------------------------


class ScriptedGame:
    def __init__(self, winning_action: str) -> None:
        self.winning_action = winning_action
        self.actions: list[str] = []

    def step(self, action: str):
        self.actions.append(action)
        won = action == self.winning_action
        info = {"won": won, "admissible_commands": ["look", "help", self.winning_action]}
        return ("You win." if won else "Nothing happens."), won, info


def _env(tmp_path: Path) -> aw.AlfworldEnv:
    game_file = tmp_path / "game.tw-pddl"
    game_file.write_text("{}")
    env = aw.AlfworldDomain().make_env(Task("g", "", {"game_file": str(game_file)}))
    env.game = ScriptedGame("put mug 1 in/on sinkbasin 1")
    return env


def test_episode_win(tmp_path):
    env = _env(tmp_path)
    step = env.step({"role": "assistant", "content": "Look first.\n<action>look</action>"})
    assert not step.done
    prompt = step.observations[0]
    assert prompt["role"] == "user" and aw.is_observation(prompt)
    assert "Your current observation is: Nothing happens." in prompt["content"]
    assert "'help'" not in prompt["content"]
    step = env.step({"role": "assistant", "content": "<action>put mug 1 in/on sinkbasin 1</action>"})
    assert step.done and step.info["status"] == "win"
    assert env.reward() == 1.0
    view = env.gold_view()
    json.dumps(view)
    assert view["won"] and view["num_clean_actions"] == 1 and view["num_invalid_actions"] == 0


def test_episode_invalid_action_reaches_env_as_noop(tmp_path):
    env = _env(tmp_path)
    env.step({"role": "assistant", "content": "I am not sure."})
    assert env.game.actions == [""]
    assert env.gold_view()["num_invalid_actions"] == 1 and env.reward() == 0.0


def test_step_failure_ends_episode(tmp_path):
    env = _env(tmp_path)

    def boom(action: str):
        raise RuntimeError("engine died")

    env.game.step = boom
    step = env.step({"role": "assistant", "content": "<action>look</action>"})
    assert step.done and env.gold_view()["status"] == "step_failed"


def test_load_tasks_and_config(tmp_path, monkeypatch):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"task_id": 1, "data": {"game_file": "games/a/game.tw-pddl",
                                                       "split": "eval_out_of_distribution"}}) + "\n"
                    + json.dumps({"task_id": 2, "prompt": "", "data": {"game_file": "/abs/game.tw-pddl"}}) + "\n")
    tasks = aw.AlfworldDomain().load_tasks(str(path), split="eval_in_distribution")
    assert tasks[0].data["game_file"] == str((tmp_path / "games/a/game.tw-pddl").resolve())
    assert tasks[0].data["split"] == "eval_out_of_distribution" and tasks[0].prompt == ""
    assert tasks[1].data == {"game_file": "/abs/game.tw-pddl", "split": "eval_in_distribution"}

    monkeypatch.delenv(aw.DATA_ENV, raising=False)
    with pytest.raises(RuntimeError):
        aw.load_config()
    config = aw.load_config(data_dir=str(tmp_path))
    assert config["logic"]["domain"] == str(tmp_path / "logic" / "alfred.pddl")
    pytest.importorskip("yaml")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("dataset:\n  data_path: $ALFWORLD_DATA/json_2.1.1/train\n")
    monkeypatch.setenv(aw.DATA_ENV, str(tmp_path))
    assert aw.load_config(str(cfg_file))["dataset"]["data_path"] == f"{tmp_path}/json_2.1.1/train"


# -- stage-1 optimizer hooks ------------------------------------------------------------------


def test_hooks():
    turns = parse_state(aw.sample_messages(), aw)
    assert aw.PINS_TURN0 is True
    assert aw.label(turns[1]) == "navigate:\"go to shelf 1\" -> 'You arrive at shelf 1. On it you see a mug' [2 menu]"
    assert aw.turn_text(turns[2]) == "You pick up the mug 1 from the shelf 1."
    assert aw.DESC and not hasattr(aw, "gold_strings")


# -- the real environment ---------------------------------------------------------------------


def test_real_environment_episode():
    pytest.importorskip("alfworld")
    data_dir = os.environ.get(aw.DATA_ENV)
    games = sorted(Path(data_dir).glob("json_2.1.1/valid_seen/*/*/game.tw-pddl")) if data_dir else []
    if not games:
        pytest.skip(f"no ALFWorld games under ${aw.DATA_ENV}")
    task = Task("real", "", {"game_file": str(games[0]), "split": "eval_in_distribution"})
    env = aw.AlfworldDomain().make_env(task)
    head = env.reset()
    assert aw.is_observation(head[0]) and "Your task is to" in head[0]["content"]
    step = env.step({"role": "assistant", "content": "<action>look</action>"})
    assert step.done or aw.is_observation(step.observations[0])
    assert env.reward() in (0.0, 1.0)

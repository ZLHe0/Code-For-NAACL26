"""CodeSearch domain: state DSL, localization reward, bash runner and environment."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from harness_programs.domains import codesearch as cs
from harness_programs.domains.base import Task
from harness_programs.serving.compress import NOOP_PROGRAM, apply_program, build_stage1, compile_program, parse_state
from harness_programs.serving.messages import make_tool_call

DIGEST_PROGRAM = """
def compress(state):
    for t in state[:-1]:
        t.reasoning = ''
        for c in t.calls:
            if c.cmd_kind == 'search':
                c.digest = '(%d hits)' % len(c.output.splitlines())
    return state
"""

GOLD = [{"file": "src/config.py",
         "changes": {"edited_modules": ["src/config.py:Loader"],
                     "edited_entities": ["src/config.py:Loader.parse"]}}]


def _apply(code: str, messages: list[dict]) -> list[dict]:
    return apply_program(compile_program(code), messages, cs)


# -- state DSL --------------------------------------------------------------------------------


def test_noop_is_identity():
    msgs = cs.sample_messages()
    assert build_stage1(cs.NAME, NOOP_PROGRAM)(msgs) == cs.sample_messages()


def test_parse_groups_calls_per_model_turn():
    turns = parse_state(cs.sample_messages(), cs)
    assert [len(t.calls) for t in turns] == [1, 2]
    assert [c.cmd_kind for c in turns[1].calls] == ["read", "nav"]
    assert turns[1].calls[1].command == "ls tests/"
    assert turns[0].reasoning == "Search for the symbol."
    assert [t.index for t in turns] == [0, 1] and all(t.n_turns == 2 for t in turns)


def test_digest_program_changes_only_observations_and_reasoning():
    msgs = cs.sample_messages()
    out = build_stage1(cs.NAME, DIGEST_PROGRAM)(msgs)
    assert len(out) == len(msgs)
    assert out[2]["content"] == "(1 hits)" and out[2]["tool_call_id"] == "a1"
    assert out[1]["tool_calls"] == msgs[1]["tool_calls"]
    assert out[1]["content"] != msgs[1]["content"]
    for before, after in zip(msgs[3:], out[3:]):
        assert before == after


def test_keep_beats_call_digest_beats_turn_digest():
    code = """
def compress(state):
    t = state[-1]
    t.digest = 'turn'
    t.calls[0].keep = True
    t.calls[0].digest = 'ignored'
    return state
"""
    out = _apply(code, cs.sample_messages())
    assert out[4]["content"] == cs.sample_messages()[4]["content"]
    assert out[5]["content"] == "turn"
    code2 = code.replace("    t.calls[0].keep = True\n", "")
    out2 = _apply(code2, cs.sample_messages())
    assert (out2[4]["content"], out2[5]["content"]) == ("ignored", "turn")


def test_call_has_no_reasoning_slot():
    call = parse_state(cs.sample_messages(), cs)[0].calls[0]
    with pytest.raises(AttributeError):
        call.reasoning = "x"


def test_classify_command():
    assert cs.classify_command("rg -n foo") == "search"
    assert cs.classify_command("LC_ALL=C grep -r foo .") == "search"
    assert cs.classify_command("git grep foo") == "search"
    assert cs.classify_command("sed -n 1,5p a.py | head") == "read"
    assert cs.classify_command("git ls-files") == "nav"
    assert cs.classify_command("wc -l a.py") == "measure"
    assert cs.classify_command("python -c 1") == "other"


def test_truncation_marker_is_recognised():
    out = cs.run_bash("yes x | head -c 5000", "/", max_output_bytes=100)
    call = cs.parse_turn(obs_message={"role": "tool", "tool_call_id": "z", "content": out}, ai_message=None,
                         msg_idx=0, ai_idx=-1, explicit_carrier=False)
    assert call.truncated


# -- reward -----------------------------------------------------------------------------------


def test_reward_perfect_partial_and_empty():
    perfect = [{"file": "src/config.py", "class_name": "Loader", "function_name": "parse"}]
    reward, parts = cs.multilevel_localization_f1_reward(perfect, GOLD)
    assert reward == pytest.approx(cs.MAX_REWARD)
    assert parts == {"file_reward": 1.0, "module_reward": 1.0, "entity_reward": 1.0}
    file_only, parts = cs.multilevel_localization_f1_reward([{"file": "src/config.py"}], GOLD)
    assert file_only == pytest.approx(1.0) and parts["entity_reward"] == 0.0
    two_files, _ = cs.multilevel_localization_f1_reward([{"file": "src/config.py"}, {"file": "x.py"}], GOLD)
    assert two_files == pytest.approx(2 / 3)
    assert cs.multilevel_localization_f1_reward(None, GOLD)[0] == 0.0


def test_malformed_location_empties_submission():
    assert cs.parse_structured_locations([{"file": "a.py"}, {"class_name": "X"}]) == ([], [], [])
    assert cs.multilevel_localization_f1_reward([{"file": "src/config.py"}, {"file": " "}], GOLD)[0] == 0.0


def test_function_without_class_counts_as_module_and_entity():
    files, modules, entities = cs.parse_structured_locations([{"file": "a.py", "function_name": "f"}])
    assert (files, modules, entities) == (["a.py"], ["a.py:f"], ["a.py:f"])


# -- bash and environment ----------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "config.py").write_text("class Loader:\n    def parse(self, path):\n        return path\n")
    (root / "README.md").write_text("A tiny project.\n")
    return root


def test_run_bash_reads_in_repo(repo):
    assert "class Loader" in cs.run_bash("cat src/config.py", str(repo))
    assert "src/config.py" in cs.run_bash("grep -rn parse .", str(repo))


def test_run_bash_rg(repo):
    if shutil.which("rg") is None:
        pytest.skip("rg not installed")
    assert "config.py" in cs.run_bash("rg -n Loader", str(repo))


def test_run_bash_caps_and_times_out(repo):
    out = cs.run_bash("yes x", str(repo), max_output_bytes=64)
    assert out.endswith("\n... (output truncated at 64 bytes)")
    assert len(out.split("\n...")[0]) == 64
    out = cs.run_bash("sleep 5", str(repo), timeout_s=1)
    assert "command timed out after 1s" in out
    assert cs.run_bash("ls", str(repo / "missing")).startswith("Error")


def _assistant(*calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": [make_tool_call(*c) for c in calls]}


def test_episode_step_finish_reward_and_gold_view(repo):
    task = Task("t0", "Parsing ignores the path.", {"repo_dir": str(repo), "gold": GOLD})
    env = cs.CodeSearchDomain().make_env(task)
    head = env.reset()
    assert len(head) == 1 and head[0]["role"] == "user" and "Parsing ignores the path." in head[0]["content"]

    step = env.step(_assistant(("c1", cs.TERMINAL_TOOL, {"command": "grep -rn Loader ."})))
    assert not step.done
    assert step.observations == [{"role": "tool", "tool_call_id": "c1",
                                  "content": step.observations[0]["content"]}]
    assert "config.py" in step.observations[0]["content"]

    locations = [{"file": "src/config.py", "class_name": "Loader", "function_name": "parse"}]
    step = env.step(_assistant(("c2", cs.FINISH_TOOL, {"locations": locations})))
    assert step.done and step.observations[0]["tool_call_id"] == "c2"
    assert env.reward() == pytest.approx(3.0)
    assert json.loads(env.final_output()) == locations
    view = env.gold_view()
    json.dumps(view)
    assert view["files"] == ["src/config.py"] and view["revealed"] == ["src/config.py:Loader.parse"]
    assert view["f1"] == pytest.approx(1.0) and view["coverage"] == 1.0 and view["wrong"] == []


def test_parallel_cap_and_no_tool_call(repo):
    task = Task("t1", "q", {"repo_dir": str(repo), "gold": GOLD})
    env = cs.CodeSearchEnv(task, max_parallel=1)
    env.reset()
    step = env.step(_assistant(("a", cs.TERMINAL_TOOL, {"command": "ls"}),
                               ("b", cs.TERMINAL_TOOL, {"command": "ls"})))
    assert [o["tool_call_id"] for o in step.observations] == ["a", "b"]
    assert step.observations[1]["content"].startswith("Skipped")
    step = env.step({"role": "assistant", "content": "I am done."})
    assert step.done and env.reward() == 0.0 and env.final_output() == ""


def test_load_tasks_resolves_relative_paths(tmp_path, repo):
    path = tmp_path / "tasks.jsonl"
    rows = [{"task_id": i, "prompt": "p", "data": {"repo_dir": "repo", "gold": json.dumps(GOLD)}} for i in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tasks = cs.CodeSearchDomain().load_tasks(str(path), limit=2)
    assert len(tasks) == 2 and tasks[0].task_id == "0"
    assert tasks[0].data["repo_dir"] == str(repo.resolve())
    assert cs.CodeSearchDomain().make_env(tasks[0]).file_changes == GOLD


# -- stage-1 optimizer hooks ------------------------------------------------------------------


def test_hooks():
    turns = parse_state(cs.sample_messages(), cs)
    assert "[search]" in cs.label(turns[0])
    assert "parse_config" in cs.turn_text(turns[0])
    assert cs.gold_strings({"files": ["src/config.py"]}) == ["src/config.py"]
    assert cs.gold_strings(GOLD) == ["src/config.py"]
    assert cs.gold_strings(None) == []
    assert cs.DESC


def test_evidence():
    episodes = [SimpleNamespace(messages=cs.sample_messages(), gold={"files": ["src/config.py"]}),
                SimpleNamespace(messages=cs.sample_messages(), gold={"files": ["nowhere.py"]})]
    stats, text = cs.evidence(episodes)
    assert stats["episodes_total"] == 2 and stats["episodes_with_gold"] == 1
    assert stats["gold_first_seen_in"] == {"search": 1}
    assert stats["commands_per_turn"]["max"] == 2
    assert "Commands per turn" in text

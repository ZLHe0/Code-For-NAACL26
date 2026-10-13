"""SQL domain: state DSL, action parsing, read-only executor, execution-match reward, environment."""

from __future__ import annotations

import json
import sqlite3

import pytest

from harness_programs.domains import sql
from harness_programs.domains.base import Task
from harness_programs.serving.compress import NOOP_PROGRAM, apply_program, build_stage1, compile_program, parse_state

DIGEST_PROGRAM = """
def compress(state):
    for t in state[:-1]:
        t.digest = '[%s: %d rows]' % (t.obs_kind, t.row_count)
        t.reasoning = ''
    return state
"""


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        INSERT INTO users VALUES (1, 'ann', 'oslo'), (2, 'bob', 'rome'), (3, 'cy', 'oslo');
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);
        INSERT INTO orders VALUES (1, 1, 9.5), (2, 1, 3.0), (3, 2, 7.25);
    """)
    conn.commit()
    conn.close()
    return str(path)


# -- state DSL --------------------------------------------------------------------------------


def test_noop_is_identity():
    msgs = sql.sample_messages()
    assert build_stage1(sql.NAME, NOOP_PROGRAM)(msgs) == sql.sample_messages()


def test_parse_state():
    turns = parse_state(sql.sample_messages(), sql)
    assert [t.obs_kind for t in turns] == ["schema", "table"]
    assert turns[0].query == "SELECT name FROM sqlite_master"
    assert turns[0].reasoning == "First inspect the schema."
    assert turns[1].row_count == 1 and not turns[1].truncated
    assert "<reminder>" not in turns[0].observation and "You have 5 turn(s) left" in turns[0]._reminder


def test_digest_program_preserves_actions_and_reattaches_reminder():
    msgs = sql.sample_messages()
    out = build_stage1(sql.NAME, DIGEST_PROGRAM)(msgs)
    assert len(out) == len(msgs)
    assert out[:1] == msgs[:1] and out[3:] == msgs[3:]
    assert out[2]["content"] == ("<observation>[schema: 0 rows]\n<reminder>You have 5 turn(s) left to complete "
                                 "the task.</reminder></observation>")
    assert out[1]["content"] == "<action>SELECT name FROM sqlite_master</action>"


def test_live_turn_edit_keeps_reminder():
    code = "def compress(state):\n    state[-1].observation = 'short'\n    return state\n"
    out = apply_program(compile_program(code), sql.sample_messages(), sql)
    assert out[-1]["content"].startswith("<observation>short\n<reminder>You have 4 turn(s) left")


def test_row_count_formats():
    assert sql._row_count("0\n1\n2\n3") == 3
    rendered = sql._format_json(["a"], [(1,), (2,)], 2000)
    assert sql._row_count(rendered) == 2
    cut = sql._format_json(["a"], [(i,) for i in range(10)], 100)
    assert cut.endswith("... (truncated)") and 1 <= sql._row_count(cut) < 10


# -- action parsing ---------------------------------------------------------------------------


def test_parse_sql_action():
    assert sql.parse_sql_action("Look.\n<action>SELECT 1</action>") == ("action", "SELECT 1", True)
    assert sql.parse_sql_action("<ANSWER> SELECT 2 </ANSWER><action>x</action>") == ("answer", "SELECT 2", True)
    assert sql.parse_sql_action("<action>SELECT 1</action> then <answer>SELECT 2</answer>")[0] == "action"
    assert sql.parse_sql_action("<think><answer>SELECT 9</answer></think><action>SELECT 1</action>")[1] == "SELECT 1"
    assert sql.parse_sql_action("<action>  </action>") == ("action", None, False)
    assert sql.parse_sql_action("no tags") == ("none", None, False)
    assert sql.parse_actions("<answer>SELECT 2</answer>") == ["answer:SELECT 2"]
    assert sql.parse_actions("nothing") == []


# -- executor ---------------------------------------------------------------------------------


def test_execute_json_and_table(db):
    out = sql.execute_sql(db, "SELECT name FROM users WHERE city = 'oslo' ORDER BY id")
    assert out.startswith("The result is:")
    guidance = json.loads(out[len("The result is:"):])
    assert guidance == {"columns": ["name"], "data": [{"name": "ann"}, {"name": "cy"}]}
    table = sql.execute_sql(db, "SELECT city FROM users", obs_format=sql.OBS_TABLE)
    assert sorted(table.split()[1:]) == ["oslo", "rome"]


def test_execute_errors_caps_and_is_read_only(db):
    assert sql.execute_sql(db, "SELECT nope FROM users").startswith("Error executing SQL:")
    long = sql.execute_sql(db, "SELECT name FROM users", max_chars=20)
    assert len(long) == 20 + len("... (truncated)")
    assert sql.execute_sql(db, "DELETE FROM users").startswith("Error executing SQL:")
    assert sql.fetch_rows(db, "SELECT COUNT(*) FROM users") == frozenset({(3,)})


def test_execute_timeout(db):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
            "SELECT COUNT(*) FROM c")
    assert sql.execute_sql(db, slow, timeout=0.5) == f"SQL Timeout:\n{slow}"


# -- reward -----------------------------------------------------------------------------------


def test_exact_match_is_set_equality(db):
    gold = "SELECT name FROM users WHERE city = 'oslo'"
    assert sql.grade("SELECT name FROM users WHERE city = 'oslo' ORDER BY id DESC", gold, db)["correct"]
    assert sql.grade("SELECT name FROM users WHERE city='oslo' UNION ALL SELECT 'ann'", gold, db)["correct"]
    assert not sql.grade("SELECT name, id FROM users WHERE city = 'oslo'", gold, db)["correct"]
    assert sql.grade("SELECT bad FROM users", gold, db) == {"submitted": True, "executed": False, "correct": False}
    assert sql.grade(None, gold, db) == {"submitted": False, "executed": False, "correct": False}


def test_refined_match_ignores_column_order_but_counts_rows(db):
    gold = "SELECT id, name FROM users"
    assert not sql.grade("SELECT name, id FROM users", gold, db)["correct"]
    assert sql.grade("SELECT name, id FROM users", gold, db, mode=sql.MATCH_REFINED)["correct"]
    dup = "SELECT id, name FROM users UNION ALL SELECT 1, 'ann'"
    assert sql.grade(dup, gold, db)["correct"]
    assert not sql.grade(dup, gold, db, mode=sql.MATCH_REFINED)["correct"]
    assert sql.results_match([(1, "a")], [(1.0, "a")], sql.MATCH_REFINED)


# -- environment ------------------------------------------------------------------------------


def test_episode(db):
    task = Task("q1", "How many users live in oslo?", {"db_path": db, "gold_sql":
                                                        "SELECT COUNT(*) FROM users WHERE city = 'oslo'"})
    env = sql.SqlDomain().make_env(task)
    head = env.reset()
    assert len(head) == 1 and "How many users live in oslo?" in head[0]["content"]
    assert "First, briefly explain" not in head[0]["content"]

    step = env.step({"role": "assistant", "content": "<action>SELECT city FROM users</action>"})
    assert not step.done
    msg = step.observations[0]
    assert msg["role"] == "user" and msg["content"].startswith("<observation>The result is:")
    assert msg["content"].endswith("<reminder>You have 5 turn(s) left to complete the task.</reminder></observation>")

    step = env.step({"role": "assistant", "content": "hmm"})
    assert sql.INVALID_ACTION_OBS in step.observations[0]["content"]
    assert "You have 4 turn(s) left" in step.observations[0]["content"]

    step = env.step({"role": "assistant", "content": "<answer>SELECT COUNT(id) FROM users WHERE city='oslo'</answer>"})
    assert step.done and step.observations == []
    assert env.reward() == 1.0
    assert env.final_output() == "SELECT COUNT(id) FROM users WHERE city='oslo'"
    view = env.gold_view()
    json.dumps(view)
    assert view["correct"] and view["f1"] == 1.0 and not view["broken_sql"]


def test_episode_wrong_and_unanswered(db):
    task = Task("q2", "?", {"db_path": db, "gold_sql": "SELECT 1"})
    env = sql.SqlEnv(task)
    env.reset()
    env.step({"role": "assistant", "content": "<answer>SELECT 2</answer>"})
    assert env.reward() == 0.0 and env.gold_view()["wrong_rows"]
    env2 = sql.SqlEnv(task)
    env2.reset()
    assert env2.reward() == 0.0 and env2.final_output() == "" and not env2.gold_view()["submitted"]


def test_load_tasks_resolves_relative_paths(tmp_path, db):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps({"task_id": "a", "prompt": "p", "data": {"db_path": "shop.sqlite",
                                                                        "gold_sql": "SELECT 1"}}) + "\n\n")
    tasks = sql.SqlDomain().load_tasks(str(path))
    assert len(tasks) == 1 and tasks[0].data["db_path"] == str((tmp_path / "shop.sqlite").resolve())


# -- stage-1 optimizer hooks ------------------------------------------------------------------


def test_hooks():
    turns = parse_state(sql.sample_messages(), sql)
    assert sql.label(turns[1]) == '[table] q="SELECT COUNT(*) FROM users"  rows=1'
    assert sql.turn_text(turns[0]) == "name\nusers\norders"
    assert sql.DESC and not hasattr(sql, "gold_strings")

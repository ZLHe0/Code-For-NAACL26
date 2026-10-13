"""Text-to-SQL: answer a question over a SQLite database by querying it, then submitting SQL.

Actions are free text: ``<action>SQL</action>`` runs an exploratory query and
``<answer>SQL</answer>`` submits the final query and ends the episode. Each query result comes
back as a user message ``<observation>RESULT\\n<reminder>You have N turn(s) left to complete
the task.</reminder></observation>``. The reward is binary execution match: 1.0 iff the
submitted SQL and the gold SQL both execute and return the same result set.

Every query opens a fresh read-only connection, runs inside a transaction that is always rolled
back, and is interrupted past a timeout, so the database file is never modified.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serving.compress import register_state_dsl
from ..serving.messages import Message, Messages, content_of, split_reasoning, strip_think
from .base import StepResult, Task, register_domain

NAME = "sql"
MAX_TURNS = 6


# ---------------------------------------------------------------------------------------------
# Prompt and action parsing
# ---------------------------------------------------------------------------------------------

SQL_NO_HISTORY = """
You are an expert SQL agent. You interact with a SQL database to answer questions by writing and executing queries.

{task_description}

{guidance}Now it's your turn to respond for the current step. Choose one of the following actions:
(1) Execute a SQL query to explore or verify using the format: <action>YOUR SQL QUERY HERE</action>. The query results will be returned to you as an observation.
(2) If no further exploration is needed, directly provide the final SQL in <answer>YOUR FINAL SQL QUERY HERE</answer>.
Only output the columns asked in the question -- nothing extra.
""".strip()

# Identical, except it asks for one sentence of plain-text reasoning before the action.
SQL_NO_HISTORY_REASON = """
You are an expert SQL agent. You interact with a SQL database to answer questions by writing and executing queries.

{task_description}

{guidance}Now it's your turn to respond for the current step.
First, briefly explain your reasoning in one sentence by identifying the relevant tables and columns. Then, choose one of the following actions:
(1) Execute a SQL query to explore or verify using the format: <action>YOUR SQL QUERY HERE</action>. The query results will be returned to you as an observation.
(2) If no further exploration is needed, directly provide the final SQL in <answer>YOUR FINAL SQL QUERY HERE</answer>.
Only output the columns asked in the question -- nothing extra.
""".strip()

OBSERVATION_BLOCK = (
    "<observation>{observation}\n"
    "<reminder>You have {turns_left} turn(s) left to complete the task.</reminder></observation>"
)

INVALID_ACTION_OBS = (
    "Your previous response was invalid. Use <action>YOUR SQL QUERY</action> to "
    "execute a query, or <answer>YOUR FINAL SQL QUERY</answer> to submit your answer."
)

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def build_prompt(task_description: str, *, reason: bool = False, guidance: str = "") -> str:
    template = SQL_NO_HISTORY_REASON if reason else SQL_NO_HISTORY
    return template.format(task_description=task_description, guidance=guidance)


def format_observation(observation: str, turns_left: int) -> str:
    return OBSERVATION_BLOCK.format(observation=observation, turns_left=turns_left)


def postprocess_action(text: str) -> str:
    """Trim everything after the first closing ``</action>``, else the first ``</answer>``."""
    lower = (text or "").lower()
    for tag in ("</action>", "</answer>"):
        if tag in lower:
            return text[: lower.index(tag) + len(tag)]
    return text or ""


def parse_sql_action(text: str) -> tuple[str, str | None, bool]:
    """``(kind, sql, valid)``: kind is ``'answer'`` (checked first), ``'action'`` or ``'none'``.

    Only what the turn committed to outside a ``<think>`` block is read.
    """
    raw = postprocess_action(strip_think(text or ""))
    answer = _ANSWER_RE.search(raw)
    if answer is not None:
        sql = answer.group(1).strip()
        return "answer", (sql or None), bool(sql)
    action = _ACTION_RE.search(raw)
    if action is not None:
        sql = action.group(1).strip()
        return "action", (sql or None), bool(sql)
    return "none", None, False


# ---------------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------------

DEFAULT_EXEC_TIMEOUT = 5
DEFAULT_MAX_OBS_CHARS = 2000
OBS_JSON = "json"      # column headers and cell values for up to 10 rows, as JSON
OBS_TABLE = "table"    # the de-duplicated row set as a plain table, without column names
JSON_MAX_ROWS = 10
_JSON_PREFIX = "The result is:"


def _connect(db_file: str) -> sqlite3.Connection:
    uri = f"{Path(db_file).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def _run_with_timeout(db_file: str, sql: str, timeout: float, fetch: Any) -> tuple[bool, Any]:
    """Run ``sql`` read-only in a worker thread; ``(timed_out, result_or_exception)``."""
    holder: dict[str, Any] = {"value": None, "conn": None}
    done = threading.Event()

    def worker() -> None:
        conn = None
        try:
            conn = _connect(db_file)
            holder["conn"] = conn
            cursor = conn.cursor()
            conn.execute("BEGIN TRANSACTION;")
            cursor.execute(sql)
            holder["value"] = fetch(cursor)
            conn.rollback()
        except Exception as exc:  # noqa: BLE001 -- reported as an observation
            holder["value"] = exc
        finally:
            if conn is not None:
                for close in (conn.rollback, conn.close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    if done.wait(timeout):
        return False, holder["value"]
    conn = holder.get("conn")
    if conn is not None:
        try:
            conn.interrupt()
        except Exception:  # noqa: BLE001
            pass
    done.wait()
    return True, None


def _cap(text: str, max_chars: int) -> str:
    return text[:max_chars] + "... (truncated)" if len(text) > max_chars else text


def _format_json(columns: list[str], rows: list[tuple[Any, ...]], max_chars: int) -> str:
    data = [dict(zip(columns, row)) for row in rows]
    out = _JSON_PREFIX + json.dumps({"columns": columns, "data": data}, indent=2, ensure_ascii=False, default=str)
    return _cap(out, max_chars)


def _plain_table(rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return "Empty DataFrame"
    width = max(len(r) for r in rows)
    cells = [[str(c) for c in r] + [""] * (width - len(r)) for r in rows]
    header = [str(i) for i in range(width)]
    sizes = [max(len(header[i]), *(len(r[i]) for r in cells)) for i in range(width)]
    lines = [" ".join(h.rjust(s) for h, s in zip(header, sizes))]
    lines += [" ".join(c.rjust(s) for c, s in zip(r, sizes)) for r in cells]
    return "\n".join(lines)


def _format_table(rows: frozenset[tuple[Any, ...]], max_chars: int) -> str:
    """The row set as a table; past the cap, the first 20 rows, cut to the cap."""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    listed = list(rows)

    def render(subset: list[tuple[Any, ...]]) -> str:
        return pd.DataFrame(subset).to_string(index=False) if pd is not None else _plain_table(subset)

    out = render(listed)
    if len(out) > max_chars:
        out = render(listed[:20])[:max_chars]
        out = "Truncated (showing first 20 rows):\n" + out
    return out


def execute_sql(db_file: str, sql: str, *, timeout: float = DEFAULT_EXEC_TIMEOUT, obs_format: str = OBS_JSON,
                max_chars: int = DEFAULT_MAX_OBS_CHARS) -> str:
    """Run an exploratory query and render the observation text.

    Returns the formatted result, ``Error executing SQL: ...`` on a SQL error, or
    ``SQL Timeout:`` followed by the query when it exceeds ``timeout``.
    """
    if obs_format not in (OBS_JSON, OBS_TABLE):
        raise ValueError(f"unknown observation format {obs_format!r}; want {OBS_JSON!r} or {OBS_TABLE!r}")

    def fetch(cursor: sqlite3.Cursor) -> Any:
        if obs_format == OBS_JSON:
            columns = [d[0] for d in cursor.description] if cursor.description else []
            return columns, cursor.fetchmany(JSON_MAX_ROWS)
        return frozenset(cursor.fetchall())

    timed_out, result = _run_with_timeout(db_file, sql, timeout, fetch)
    if timed_out:
        return f"SQL Timeout:\n{sql}"
    if isinstance(result, Exception):
        return _cap(f"Error executing SQL: {result}", max_chars)
    if isinstance(result, frozenset):
        return _format_table(result, max_chars)
    return _format_json(*result, max_chars)


# ---------------------------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------------------------

DEFAULT_GRADE_TIMEOUT = 30
MATCH_EXACT = "exact"      # result SETS equal: row order and duplicates ignored, column order matters
MATCH_REFINED = "refined"  # row counts equal, then cells sorted within rows and rows sorted


def _cell_key(value: Any) -> tuple[int, str]:
    """A total order over mixed SQLite cell values; ``1`` and ``1.0`` compare equal."""
    if value is None:
        return (0, "")
    if isinstance(value, (bool, int, float)):
        return (1, f"{float(value):.12g}")
    if isinstance(value, bytes):
        return (2, value.decode("utf-8", "replace"))
    return (3, str(value))


def _normalise_refined(rows: Iterable[tuple[Any, ...]]) -> list[tuple[tuple[int, str], ...]]:
    return sorted(tuple(sorted(_cell_key(c) for c in row)) for row in rows)


def results_match(pred_rows: Any, gold_rows: Any, mode: str = MATCH_EXACT) -> bool:
    """Exact mode expects frozensets; refined mode expects row lists."""
    if pred_rows is None or gold_rows is None:
        return False
    if mode == MATCH_EXACT:
        return bool(pred_rows == gold_rows)
    if len(pred_rows) != len(gold_rows):
        return False
    return _normalise_refined(pred_rows) == _normalise_refined(gold_rows)


def fetch_rows(db_file: str, sql: str, *, timeout: float = DEFAULT_GRADE_TIMEOUT, as_set: bool = True) -> Any:
    """All result rows (a frozenset, or a list when ``as_set`` is false); ``None`` on error or timeout."""
    def fetch(cursor: sqlite3.Cursor) -> Any:
        rows = cursor.fetchall()
        return frozenset(rows) if as_set else list(rows)

    timed_out, result = _run_with_timeout(db_file, sql, timeout, fetch)
    if timed_out or isinstance(result, Exception):
        return None
    return result


def grade(pred_sql: str | None, gold_sql: str, db_file: str, *, timeout: float = DEFAULT_GRADE_TIMEOUT,
          mode: str = MATCH_EXACT) -> dict[str, bool]:
    """``{submitted, executed, correct}`` for one submission."""
    if mode not in (MATCH_EXACT, MATCH_REFINED):
        raise ValueError(f"unknown match mode {mode!r}; want {MATCH_EXACT!r} or {MATCH_REFINED!r}")
    if not pred_sql:
        return {"submitted": False, "executed": False, "correct": False}
    as_set = mode == MATCH_EXACT
    pred_rows = fetch_rows(db_file, pred_sql, timeout=timeout, as_set=as_set)
    gold_rows = fetch_rows(db_file, gold_sql, timeout=timeout, as_set=as_set)
    executed = pred_rows is not None
    correct = executed and gold_rows is not None and results_match(pred_rows, gold_rows, mode)
    return {"submitted": True, "executed": executed, "correct": correct}


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------


class SqlEnv:
    def __init__(self, task: Task, *, max_turns: int = MAX_TURNS, reason: bool = False, obs_format: str = OBS_JSON,
                 max_obs_chars: int = DEFAULT_MAX_OBS_CHARS, exec_timeout: float = DEFAULT_EXEC_TIMEOUT,
                 match_mode: str = MATCH_EXACT) -> None:
        self.task = task
        self.db_file = str(task.data["db_path"])
        self.gold_sql = str(task.data["gold_sql"])
        self.max_turns = max_turns
        self.reason = reason
        self.obs_format = obs_format
        self.max_obs_chars = max_obs_chars
        self.exec_timeout = exec_timeout
        self.match_mode = match_mode
        self.turn = 0
        self.answer: str | None = None
        self._grade: dict[str, bool] | None = None

    def reset(self) -> Messages:
        return [{"role": "user", "content": build_prompt(self.task.prompt, reason=self.reason)}]

    def step(self, assistant: Message) -> StepResult:
        kind, sql, _valid = parse_sql_action(content_of(assistant))
        turns_left = self.max_turns - (self.turn + 1)
        self.turn += 1
        if kind == "answer":
            self.answer = sql or ""
            self._grade = None
            return StepResult([], done=True, info={"status": "answered"})
        if kind == "action" and sql:
            observation = execute_sql(self.db_file, sql, timeout=self.exec_timeout, obs_format=self.obs_format,
                                      max_chars=self.max_obs_chars)
        else:
            observation = INVALID_ACTION_OBS
        return StepResult([{"role": "user", "content": format_observation(observation, turns_left)}])

    def _graded(self) -> dict[str, bool]:
        if self._grade is None:
            self._grade = grade(self.answer, self.gold_sql, self.db_file, mode=self.match_mode)
        return self._grade

    def reward(self) -> float:
        return 1.0 if self._graded()["correct"] else 0.0

    def final_output(self) -> str:
        return self.answer or ""

    def gold_view(self) -> dict[str, Any]:
        g = self._graded()
        return {
            "gold_sql": self.gold_sql,
            "submitted": g["submitted"], "executed": g["executed"], "correct": g["correct"],
            "f1": 1.0 if g["correct"] else 0.0,
            "wrong_rows": g["submitted"] and g["executed"] and not g["correct"],
            "broken_sql": g["submitted"] and not g["executed"],
        }


def _resolve(base: Path, value: Any) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


@dataclass
class SqlDomain:
    name: str = NAME
    max_turns: int = MAX_TURNS
    tools: Any = None
    reason: bool = False
    obs_format: str = OBS_JSON
    max_obs_chars: int = DEFAULT_MAX_OBS_CHARS
    exec_timeout: float = DEFAULT_EXEC_TIMEOUT
    match_mode: str = MATCH_EXACT

    def load_tasks(self, path: str, *, split: str = "train", limit: int | None = None) -> list[Task]:
        """Read a JSONL of ``{task_id, prompt, data: {db_path, gold_sql}}``; see docs/domains.md.

        ``db_path`` is resolved against the JSONL's directory unless absolute. ``split`` is not
        used: one file holds one split.
        """
        del split
        base = Path(path).resolve().parent
        tasks = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            data = dict(row["data"])
            data["db_path"] = _resolve(base, data["db_path"])
            tasks.append(Task(task_id=str(row["task_id"]), prompt=row["prompt"], data=data))
        return tasks[:limit] if limit else tasks

    def task_key(self, task: Task) -> str:
        """The database plus the question."""
        return f"{Path(task.data['db_path']).stem}\n{task.prompt.strip()}"

    def make_env(self, task: Task) -> SqlEnv:
        return SqlEnv(task, max_turns=self.max_turns, reason=self.reason, obs_format=self.obs_format,
                      max_obs_chars=self.max_obs_chars, exec_timeout=self.exec_timeout, match_mode=self.match_mode)


# ---------------------------------------------------------------------------------------------
# State DSL
#
# One Turn per <observation> message: the SQL that produced it, a coarse kind, the result body
# and the rationale. The <action>/<answer> span lives on the assistant message and is never
# rewritten. The <reminder> turn counter is harness state: it is kept off .observation and
# re-attached on every render, so no program can drop it from the live turn.
# ---------------------------------------------------------------------------------------------

_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
_REMINDER_RE = re.compile(r"<reminder>.*?</reminder>", re.DOTALL)
EXEC_ERROR_RE = re.compile(r"Error executing SQL|SQL Timeout", re.IGNORECASE)
SCHEMA_RE = re.compile(r"\bsqlite_master\b|\bpragma\b", re.IGNORECASE)


@dataclass
class Turn:
    index: int                 # 0-based (0 = oldest)
    n_turns: int
    query: str                 # the SQL that produced this observation
    obs_kind: str              # 'table' | 'schema' | 'error'
    observation: str           # the observation body (result table / schema / error)
    row_count: int             # number of result rows (0 for non-table observations)
    truncated: bool            # True if the environment already truncated the result
    reasoning: str = ""        # rationale written before the query ('' on a no-reason run)
    digest: str | None = None  # if set, replaces the whole observation
    _msg_idx: int = -1
    _ai_idx: int = -1
    _raw: str = ""             # the original message text, re-emitted when untouched
    _orig: str = ""            # the body as parsed, to detect an edit
    _orig_reasoning: str = ""
    _reminder: str = ""        # the turn-count line, re-attached on every render


CONTRACT = '''\
You write a Python function:

    def compress(state):
        """Transform the observation state of one text-to-SQL episode and
        return the transformed list of turns."""
        ...
        return state

`state` is a list[Turn], oldest first. Each Turn has:
    .index        int    0-based turn position (0 = oldest)
    .n_turns      int    number of turns in the episode so far
    .query        str    the SQL that produced this observation
    .obs_kind     str    'table' (result rows) | 'schema' (PRAGMA/sqlite_master) | 'error'
    .observation  str    the observation body (result table / schema / error text)
    .row_count    int    number of result rows (0 for non-table observations)
    .truncated    bool   True if the environment already truncated the result
    .reasoning    str    the rationale the agent wrote before this query ('' if the run
                         did not ask for reasoning); shortening it does not affect the query
    .digest       str|None   set this to replace the whole observation with a short string

You MAY: rewrite .observation, rewrite or clear .reasoning, set .digest on older turns,
keep recent turns full. You MUST return a list[Turn]. You may not fabricate results.

How your output is served -- three facts about the runtime, not advice:

  * You are called ONCE per model turn, on the WHOLE episode so far. `.n_turns` is the
    total number of turns, not a window size. After you return, a second stage measures
    the result and, if it still exceeds the token budget, drops whole OLDEST turns until
    it fits. So the more you compress, the more turns survive that drop.
  * You cannot remove a turn. Leaving a Turn out of the list you return does not delete
    it -- that observation is served unchanged. Dropping history is the second stage's
    job. Your lever is what each turn COSTS.
  * The LAST turn in `state` is the observation the agent must answer RIGHT NOW. It is the
    live question, not history. Compressing it does not save history cost; it edits the
    thing being asked.

The task statement is not in `state` and cannot be modified.

    # naive: keep everything
    def compress(state):
        return state

    # syntax demo (NOT a strategy): stub the very first turn
    def compress(state):
        for t in state:
            if t.index == 0:
                t.digest = 'first turn'
        return state
'''


def _obs_body(content: str) -> str:
    match = _OBS_RE.search(content)
    body = match.group(1) if match else content
    return _REMINDER_RE.sub("", body).strip()


def _row_count(body: str) -> int:
    """Result rows, for either observation format.

    The table format is a header plus one line per row. The JSON format is parsed; a body cut
    at the character cap is no longer valid JSON, so there the rows shown are counted by their
    opening brace, a lower bound flagged by ``.truncated``.
    """
    _, marker, guidance = body.partition(_JSON_PREFIX)
    if not marker:
        return body.count("\n")
    try:
        data = json.loads(guidance).get("data")
        if isinstance(data, list):
            return len(data)
    except (ValueError, AttributeError):
        pass
    return guidance.count("\n    {")


def is_observation(message: Message) -> bool:
    return message.get("role") == "user" and "<observation>" in content_of(message)


def parse_turn(*, obs_message: Message, ai_message: Message | None, msg_idx: int, ai_idx: int,
               explicit_carrier: bool) -> Turn:
    raw = content_of(obs_message)
    body = _obs_body(raw)
    reminder = _REMINDER_RE.search(raw)
    query, reasoning = "", ""
    if ai_message is not None:
        reasoning, _ = split_reasoning(ai_message, explicit_carrier=explicit_carrier)
        query = parse_sql_action(content_of(ai_message))[1] or ""
    if EXEC_ERROR_RE.search(body):
        kind = "error"
    elif SCHEMA_RE.search(query):
        kind = "schema"
    else:
        kind = "table"
    return Turn(index=0, n_turns=0, query=query, obs_kind=kind, observation=body,
                row_count=_row_count(body) if kind == "table" else 0, truncated="truncated" in body.lower(),
                reasoning=reasoning, _msg_idx=msg_idx, _ai_idx=ai_idx, _raw=raw, _orig=body,
                _orig_reasoning=reasoning, _reminder=reminder.group(0) if reminder else "")


def render_turn(turn: Turn) -> str:
    """Verbatim when untouched; otherwise the new body with the reminder re-attached."""
    if turn.digest is None and turn._raw and turn.observation == turn._orig:
        return turn._raw
    body = turn.digest if turn.digest is not None else turn.observation
    tail = f"\n{turn._reminder}" if turn._reminder else ""
    return f"<observation>{body}{tail}</observation>"


def rebuild_obs(message: Message, text: str, turn: Turn) -> Message:
    del turn
    return {**message, "content": text}


def sample_messages() -> Messages:
    def obs(body: str, left: int) -> Message:
        return {"role": "user", "content": format_observation(body, left)}

    return [
        {"role": "user", "content": build_prompt("How many users are there?")},
        {"role": "assistant", "content": "First inspect the schema.\n<action>SELECT name FROM sqlite_master</action>"},
        obs("name\nusers\norders", 5),
        {"role": "assistant", "content": "Now count.\n<action>SELECT COUNT(*) FROM users</action>"},
        obs('The result is:{\n  "columns": [\n    "COUNT(*)"\n  ],\n  "data": [\n    {\n      "COUNT(*)": 42\n'
            "    }\n  ]\n}", 4),
    ]


# ---------------------------------------------------------------------------------------------
# Stage-1 optimizer hooks
# ---------------------------------------------------------------------------------------------

DESC = (
    "A text-to-SQL agent answers a question by writing SQL, executing it, and revising. Each "
    "observation is one query's result: a table of rows, a schema/PRAGMA listing, or an "
    "execution error. Result tables can be large; schema and error observations are short.")


def parse_actions(content: str) -> list[str]:
    kind, sql, valid = parse_sql_action(content)
    return [f"{kind}:{sql}"] if valid else []


def label(turn: Turn) -> str:
    return f'[{turn.obs_kind}] q="{(turn.query or "")[:34]}"  rows={turn.row_count}'


def turn_text(turn: Turn) -> str:
    return turn.observation


register_state_dsl(NAME, sys.modules[__name__])
register_domain(SqlDomain())

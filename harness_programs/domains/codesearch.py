"""Code localization: find the files, classes and functions an issue needs changed.

The agent explores a repository checkout with read-only bash commands through a ``terminal``
tool and submits its answer through ``localization_finish``. Observations are ``tool``
messages, one per command; one model turn may issue several commands. The reward is the sum of
three F1 scores (file, module, entity level) of the submitted locations against the gold edit
targets, so its scale is ``[0, 3]``.

The checkout under ``repo_dir`` must already be in the state the issue describes. Commands run
in it directly; nothing here copies, resets or patches it, so a policy that writes to the
checkout changes it for later episodes. Point ``repo_dir`` at a throwaway copy if that matters.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..serving.compress import parse_state, register_state_dsl
from ..serving.messages import Message, Messages, content_of, make_tool_call, split_reasoning, tool_calls_of
from .base import StepResult, Task, register_domain

NAME = "codesearch"
MAX_TURNS = 10
TERMINAL_TOOL = "terminal"
FINISH_TOOL = "localization_finish"


# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------

PROMPT_VARIANTS = ("full", "stripped")

_REASONING_INSTRUCTION = (
    "\n\nBefore each tool call, briefly explain your reasoning in one sentence, then make the tool call."
)

_SYSTEM_PROMPT = """\
You are a specialized code localization agent. Your sole objective is to identify and return the files in the codebase that are relevant to the user's query.
You are given access to the codebase in a linux file system.

## PRIMARY DIRECTIVE
- Find relevant files, do NOT answer the user's query directly
- Prioritize precision: every file you return should be relevant
- You have up to {max_turns} turns to explore and return your answer

## TOOL USAGE REQUIREMENTS

### bash tool (REQUIRED for search)
- You MUST use the bash tool to search and explore the codebase
- Execute bash commands like: rg, grep, find, ls, cat, head, tail, sed
- Use parallel tool calls: invoke bash tool up to {max_parallel} times concurrently in a single turn
- NEVER exceed {max_parallel} parallel tool calls per turn
- Common patterns:
  * `rg "pattern" -t py` - search for code patterns
  * `rg --files | grep "keyword"` - find files by name
  * `cat path/to/file.py` - read file contents
  * `find . -name "*.py" -type f` - locate files by extension
  * `wc -l path/to/file.py` - count lines in a file
  * `sed -n '1,100p' path/to/file.py` - read lines 1-100 of a file
  * `head -n 100 path/to/file.py` - read first 100 lines
  * `tail -n 100 path/to/file.py` - read last 100 lines

### Reading Files (CRITICAL for context management)
- NEVER read entire large files with `cat` - this will blow up your context window
- ALWAYS check file size first: `wc -l path/to/file.py`
- For files > 100 lines, read in chunks:
  * Use `sed -n '1,100p' file.py` to read lines 1-100
  * Use `sed -n '101,200p' file.py` to read lines 101-200
  * Continue with subsequent ranges as needed (201-300, 301-400, etc.)
- Strategic reading approach:
  * Read the first 50-100 lines to see imports and initial structure
  * Use `rg` to find specific patterns and their line numbers
  * Read targeted line ranges around matches using `sed -n 'START,ENDp'`
  * Only read additional chunks if the initial sections are relevant

### Submitting Your Answer (REQUIRED)

When you have identified all relevant locations, you MUST use the `localization_finish` tool to submit your results.

**When to include what:**
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class), provide the file path and class name.

## SEARCH STRATEGY

1. **Initial Exploration**: Cast a wide net
   - Search for keywords, function names, class names
   - Check file names and directory structure
   - Use up to 3 parallel bash calls to explore multiple angles
   - Check file sizes with `wc -l` before reading
   - Read promising files in chunks (lines 1-100) to verify relevance

2. **Deep Dive**: Follow the most promising leads
   - Use up to 3 parallel bash calls to investigate further
   - Read files in chunks to confirm they address the query
   - Use `rg` with line numbers to locate specific code, then read those ranges
   - Start eliminating false positives

3. **Final Verification**: Confirm your location list and terminate execution by calling the `localization_finish` tool

## CRITICAL RULES
- NEVER exceed {max_parallel} parallel bash tool calls in a single turn
- ALWAYS use the `localization_finish` tool after identifying all relevant locations
- ALWAYS use bash tool to search (do not guess file locations)
- NEVER read entire large files - always read in chunks (100-line ranges)
- Check file size with `wc -l` before reading
- Read file contents in chunks to verify relevance before including them
- Return file paths as they appear in the repository. Do not begin the path with "./"
- Aim for high precision (all files relevant) and high recall (no relevant files missed)
- Class and function names are OPTIONAL - only include when changes are at that level

## EXAMPLE OUTPUT BEHAVIOUR
Here are some examples of how to format your output when calling the `localization_finish` tool:
- src/parsers/parser.py requires changes to imports, a function parse_data which belongs to the class DataParser, and another function __str__ inside the same class. This should be represented as three separate entries: one with just the file path and one each for the two functions parse_data and __str__ with file path, class name, and function name.
- src/user.py requires changes to a global function get_user outside of any class. This should be represented as a single entry with file path and function name.
- utils/visualizer.py requires adding new function visualize inside the class Visualizer. This should be represented as a single entry with file path and class name.
- utils/configs/default_config.py requires adding a new global function and a new class. This should be represented as a single entry with just the file path. Do NOT include class or function names for this file since multiple implementations might be possible with different function names."""

# Keeps the task definition, the command menu, the parallel-call cap and the output contract;
# drops the hand-written reading policy and search strategy.
_SYSTEM_PROMPT_STRIPPED = """\
You are a specialized code localization agent. Your objective is to identify the files (and, where applicable, the classes and functions) in the codebase that need to be modified to resolve the user's issue.
You are given access to the codebase in a linux file system.

## Task
- Localize the relevant code. Do NOT attempt to fix the issue or answer it directly.
- You have up to {max_turns} turns to explore the repository and then submit your answer.

## Tools
### terminal (bash)
- Call the `terminal` tool with a bash command to search and read code.
- Available utilities include: rg, grep, find, ls, cat, head, tail, sed, wc.
- You may issue up to {max_parallel} terminal calls in parallel in a single turn; never more than {max_parallel}.
- Command syntax examples:
  * `rg "pattern" -t py`        search for a code pattern
  * `rg --files | grep name`    find files by name
  * `find . -name "*.py"`       locate files by extension
  * `wc -l path/to/file.py`     count the lines in a file
  * `sed -n '1,100p' file.py`   print lines 1-100 of a file
  * `head -n 100 file.py`       print the first 100 lines
  * `tail -n 100 file.py`       print the last 100 lines
  * `cat path/to/file.py`       print an entire file

### localization_finish (required to submit)
- When you have identified the relevant locations, call `localization_finish` with a structured list to end the task.
- Each location has: `file` (required -- path relative to the repository root, without a leading "./"), `class_name` (optional), and `function_name` (optional).
- Include `class_name` / `function_name` only when the change is at that granularity; a file-only entry is valid.

## Output format examples
- A change to method `parse_data` in class `DataParser` of `src/parsers/parser.py` -> one entry: file `src/parsers/parser.py`, class_name `DataParser`, function_name `parse_data`.
- A change to a module-level function `get_user` in `src/user.py` -> one entry: file `src/user.py`, function_name `get_user`.
- A change that adds a new class or global function to `utils/configs/default_config.py` -> one entry with just the file path."""


def _drop_parallel_promise(text: str) -> str:
    """Remove every line that promises concurrency."""
    return "\n".join(ln for ln in text.split("\n") if "parallel" not in ln.lower())


def build_system_prompt(max_turns: int, *, max_parallel: int = 1, variant: str = "full",
                        reason: bool = False) -> str:
    """The localization instructions, with the turn budget and the per-turn call cap filled in.

    At a cap of 1 every line about parallel calls is removed rather than rewritten to "up to 1".
    A cap of 2 is refused: the search-strategy section suggests "up to 3 parallel bash calls".
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown system prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    if max_parallel == 2:
        raise ValueError('max_parallel=2 contradicts the hint of "up to 3 parallel bash calls"; use 1 or >= 3')
    base = _SYSTEM_PROMPT if variant == "full" else _SYSTEM_PROMPT_STRIPPED
    if max_parallel == 1:
        base = _drop_parallel_promise(base)
    prompt = base.replace("{max_turns}", str(max_turns)).replace("{max_parallel}", str(max_parallel))
    return prompt + (_REASONING_INSTRUCTION if reason else "")


def build_user_prompt(working_dir: str, problem_statement: str) -> str:
    """The task statement: the working directory plus the issue to localize."""
    return (
        f"I have access to a python code repository in the directory {working_dir} . "
        "Consider the following issue description:\n\n"
        "<issue_description>\n"
        f"{problem_statement}\n"
        "</issue_description>\n\n"
        "Act as a code search agent and localize the specific files, classes or functions "
        "of code that need modification to resolve the issue in <issue_description>.\n\n"
        "NOTE: You do not need to solve the issue, all you need to do is localize "
        "relevant code from the repository. Your output will be used to guide another "
        "agent to solve the issue.\n\n"
        "IMPORTANT: Your output MUST follow the below rules:\n"
        '1. The final output must be a tool call to the "localization_finish" tool '
        "containing relevant code locations.\n"
        f"2. The locations of the file path must be RELATIVE to the {working_dir} "
        'directory WITHOUT any leading "./" in the output.\n'
        "3. Only include those locations in your output that need modification to "
        "resolve the issue in <issue_description>. Do NOT include any locations "
        "that do not need modification."
    )


# ---------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------

_TERMINAL_DESCRIPTION = (
    "Execute a bash command in the repository directory and return its combined "
    "stdout and stderr. Use standard Unix utilities (rg, grep, find, ls, cat, head, "
    "tail, sed, wc) to search and read code. Long output is truncated."
)

# Kept verbatim: rule 4 and the "must exist in the codebase" clause shape the entity strings the
# model emits, and the entity level is one of the three F1 terms of the reward.
_FINISH_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TERMINAL_TOOL,
            "description": _TERMINAL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINISH_TOOL,
            "description": _FINISH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "description": "List of code locations that need modification.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string",
                                         "description": "Path to the file relative to the repository root."},
                                "class_name": {"type": "string",
                                               "description": "Class name (omit for module-level changes)."},
                                "function_name": {"type": "string",
                                                  "description": "Function or method name (omit for file-level changes)."},
                            },
                            "required": ["file"],
                        },
                    },
                },
                "required": ["locations"],
            },
        },
    },
]


@dataclass
class ParsedCall:
    """One tool call off an assistant message. ``error`` is returned instead of running it."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_args: str = ""
    error: str | None = None


def parse_tool_calls(message: Message) -> list[ParsedCall]:
    """Every tool call on one assistant message, in order; malformed ones keep their id.

    A call whose arguments are not a JSON object is answered with an error rather than dropped,
    so every emitted call id gets exactly one response.
    """
    parsed: list[ParsedCall] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        call_id = call.get("id") or f"missing-tool-call-id-{index}"
        name = str(function.get("name") or "")
        raw = function.get("arguments") or ""
        if isinstance(raw, dict):
            args: Any = raw
        else:
            try:
                args = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
            except (json.JSONDecodeError, TypeError, ValueError):
                args = None
        if not isinstance(args, dict):
            parsed.append(ParsedCall(id=call_id, name=name, raw_args=str(raw),
                                     error=f"Tool error: arguments for {name or '<unknown>'} must be a JSON object."))
            continue
        parsed.append(ParsedCall(id=call_id, name=name, args=args, raw_args=str(raw)))
    return parsed


def finish_locations(parsed: Sequence[ParsedCall]) -> tuple[bool, list[Any] | None]:
    """``(finished, locations)``: a finish call with unusable arguments still ends the episode."""
    for call in parsed:
        if call.name == FINISH_TOOL:
            locations = call.args.get("locations")
            return True, (locations if isinstance(locations, list) else None)
    return False, None


def execute_tool_calls(parsed: Sequence[ParsedCall], runners: dict[str, Callable[..., str]], *,
                       limit: int) -> list[dict[str, str]]:
    """One response per call id, in emission order; calls past ``limit`` are answered, not run."""
    if limit < 1:
        raise ValueError(f"tool-call limit must be at least 1, got {limit}")
    results: list[dict[str, str]] = []
    for index, call in enumerate(parsed):
        if call.name == FINISH_TOOL:
            content = ""
        elif index >= limit:
            content = (f"Skipped: this turn issued more than {limit} tool calls. "
                       "Please issue remaining tool calls in separate turns.")
        elif call.error is not None:
            content = call.error
        elif call.name not in runners:
            content = f"Tool error: unknown tool {call.name or '<empty>'}."
        else:
            try:
                content = str(runners[call.name](**call.args))
            except Exception as exc:  # noqa: BLE001 -- a tool failure is an observation
                content = f"Tool error: {type(exc).__name__}: {exc}"
        results.append({"tool_call_id": call.id, "content": content})
    return results


# ---------------------------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_OUTPUT_BYTES = 10_000
DEFAULT_RLIMIT_AS_BYTES = 2 * 1024 ** 3
DEFAULT_RLIMIT_CPU_S = 60
DEFAULT_RLIMIT_FSIZE_BYTES = 64 * 1024 ** 2
DEFAULT_RLIMIT_NOFILE = 4096
_READ_CHUNK = 65536


def _rlimit_preexec(as_bytes: int | None, cpu_s: int | None, fsize_bytes: int | None,
                    nofile: int | None) -> Callable[[], None]:
    """The child's ``preexec_fn``; a limit the host refuses is skipped, the other guards hold."""

    def apply() -> None:
        for res_id, lim in ((resource.RLIMIT_AS, as_bytes), (resource.RLIMIT_CPU, cpu_s),
                            (resource.RLIMIT_FSIZE, fsize_bytes), (resource.RLIMIT_NOFILE, nofile)):
            if lim is None:
                continue
            try:
                resource.setrlimit(res_id, (lim, lim))
            except (ValueError, OSError):
                pass

    return apply


def run_bash(command: str, cwd: str, *, timeout_s: int = DEFAULT_TIMEOUT_S,
             max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
             rlimit_as_bytes: int | None = DEFAULT_RLIMIT_AS_BYTES,
             rlimit_cpu_s: int | None = DEFAULT_RLIMIT_CPU_S,
             rlimit_fsize_bytes: int | None = DEFAULT_RLIMIT_FSIZE_BYTES,
             rlimit_nofile: int | None = DEFAULT_RLIMIT_NOFILE) -> str:
    """Run ``command`` in ``cwd``; return combined stdout+stderr, capped, never raising.

    Four independent guards: a wall-clock timeout, a process-group kill, a byte cap applied
    while reading, and per-process resource limits. A trailing note marks a truncated or
    killed command, so the policy can tell "no output" from "too much output".
    """
    if not cwd or not os.path.isdir(cwd):
        return "Error: workspace directory not available"
    try:
        proc = subprocess.Popen(  # noqa: S602 -- a shell is the tool being offered to the policy
            ["bash", "-c", command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=_rlimit_preexec(rlimit_as_bytes, rlimit_cpu_s, rlimit_fsize_bytes, rlimit_nofile),
        )
    except Exception as exc:  # noqa: BLE001 -- a launch failure is an observation
        return f"Error launching command: {exc}"

    buf = bytearray()
    capped = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(_READ_CHUNK)
            if not chunk:
                break
            remaining = max_output_bytes - len(buf)
            if remaining > 0:
                buf.extend(chunk[:remaining])
            if len(buf) >= max_output_bytes:
                capped.set()
                break

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout_s)
    timed_out = thread.is_alive()
    if timed_out or capped.is_set() or proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except OSError:
        pass
    thread.join(timeout=1)

    out = buf.decode("utf-8", errors="replace")
    if timed_out:
        out += f"\n... (command timed out after {timeout_s}s and was killed)"
    elif capped.is_set():
        out += f"\n... (output truncated at {max_output_bytes} bytes)"
    return out


# ---------------------------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------------------------


def compute_f1(predicted: Any, true: Any, beta: float = 1.0) -> float:
    pred_set, true_set = set(predicted), set(true)
    if not true_set:
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set)
    denom = beta ** 2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta ** 2) * (precision * recall) / denom


def parse_structured_locations(locations: Sequence[Any]) -> tuple[list[str], list[str], list[str]]:
    """``(files, modules, entities)`` of a submission.

    module is ``file:Class`` or ``file:function``; entity is ``file:Class.function`` or
    ``file:function``. A location with no file makes the whole submission malformed and empty.
    """
    files: list[str] = []
    modules: list[str] = []
    entities: list[str] = []
    for loc in locations:
        if isinstance(loc, str):
            loc = {"file": loc}
        elif not isinstance(loc, dict):
            return [], [], []
        file_path = loc.get("file")
        class_name = loc.get("class_name")
        function_name = loc.get("function_name")
        if not file_path or not str(file_path).strip():
            return [], [], []
        files.append(file_path)
        if class_name:
            modules.append(f"{file_path}:{class_name}")
        elif function_name:
            modules.append(f"{file_path}:{function_name}")
        if class_name and function_name:
            entities.append(f"{file_path}:{class_name}.{function_name}")
        elif function_name:
            entities.append(f"{file_path}:{function_name}")
    return list(set(files)), list(set(modules)), list(set(entities))


def gold_sets(file_changes: Sequence[Any]) -> tuple[list[str], list[str], list[str]]:
    """``(files, modules, entities)`` named by the gold edit targets."""
    files: list[str] = []
    modules: list[str] = []
    entities: list[str] = []
    for change in file_changes:
        if not isinstance(change, dict):
            continue
        if "file" in change:
            files.append(change["file"])
        changes = change.get("changes")
        if isinstance(changes, dict):
            modules.extend(changes.get("edited_modules") or [])
            entities.extend(changes.get("edited_entities") or [])
    return files, modules, entities


def multilevel_localization_f1_reward(locations: Sequence[Any] | None, file_changes: Sequence[Any], *,
                                      file_level_weight: float = 1.0, module_level_weight: float = 1.0,
                                      entity_level_weight: float = 1.0) -> tuple[float, dict[str, float]]:
    """``(reward, breakdown)``; the reward is the weighted sum of the three F1s, in ``[0, 3]``."""
    if locations is None:
        return 0.0, {"file_reward": 0.0, "module_reward": 0.0, "entity_reward": 0.0}
    gt_files, gt_modules, gt_entities = gold_sets(file_changes)
    files, modules, entities = parse_structured_locations(locations)
    file_f1 = compute_f1(files, set(gt_files))
    module_f1 = compute_f1(modules, set(gt_modules))
    entity_f1 = compute_f1(entities, set(gt_entities))
    reward = file_f1 * file_level_weight + module_f1 * module_level_weight + entity_f1 * entity_level_weight
    return reward, {"file_reward": file_f1, "module_reward": module_f1, "entity_reward": entity_f1}


MAX_REWARD = 3.0


def _entities_submitted(locations: Sequence[Any] | None) -> dict[str, str]:
    """Entity-level names of a submission, lowercased key -> as written. File-only entries name none."""
    named: dict[str, str] = {}
    for loc in locations or []:
        if not isinstance(loc, dict):
            continue
        path = str(loc.get("file") or "").strip()
        function = str(loc.get("function_name") or "").strip()
        cls = str(loc.get("class_name") or "").strip()
        leaf = f"{cls}.{function}" if cls and function else (function or cls)
        if path and leaf:
            named.setdefault(f"{path}:{leaf}".lower(), f"{path}:{leaf}")
    return named


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------


def _file_changes(data: dict[str, Any]) -> list[Any]:
    gold = data.get("gold", data.get("file_changes", []))
    if isinstance(gold, str):
        gold = json.loads(gold)
    return list(gold or [])


class CodeSearchEnv:
    def __init__(self, task: Task, *, max_turns: int = MAX_TURNS, max_parallel: int = 1, variant: str = "full",
                 reason: bool = False, bash_timeout_s: int = DEFAULT_TIMEOUT_S,
                 max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        self.task = task
        self.repo_dir = str(task.data["repo_dir"])
        self.file_changes = _file_changes(task.data)
        self.max_turns = max_turns
        self.max_parallel = max_parallel
        self.variant = variant
        self.reason = reason
        self.bash_timeout_s = bash_timeout_s
        self.max_output_bytes = max_output_bytes
        self.finished = False
        self.locations: list[Any] | None = None
        self.status = "running"

    def reset(self) -> Messages:
        system = build_system_prompt(self.max_turns, max_parallel=self.max_parallel, variant=self.variant,
                                     reason=self.reason)
        return [{"role": "user", "content": system + "\n\n" + build_user_prompt(self.repo_dir, self.task.prompt)}]

    def _terminal(self, command: str) -> str:
        return run_bash(str(command), self.repo_dir, timeout_s=self.bash_timeout_s,
                        max_output_bytes=self.max_output_bytes)

    def step(self, assistant: Message) -> StepResult:
        parsed = parse_tool_calls(assistant)
        if not parsed:
            self.status = "no_tool_call"
            return StepResult([], done=True, info={"status": self.status})
        responses = execute_tool_calls(parsed, {TERMINAL_TOOL: self._terminal}, limit=self.max_parallel)
        finished, locations = finish_locations(parsed)
        if finished:
            self.finished, self.locations, self.status = True, locations, "answered"
        observations = [{"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]}
                        for r in responses]
        return StepResult(observations, done=finished, info={"status": self.status})

    def _score(self) -> tuple[float, dict[str, float]]:
        submitted = bool(self.locations)
        return multilevel_localization_f1_reward(self.locations if submitted else None, self.file_changes)

    def reward(self) -> float:
        return self._score()[0]

    def final_output(self) -> str:
        return json.dumps(self.locations) if self.locations is not None else ""

    def gold_view(self) -> dict[str, Any]:
        """The gold targets and how the submission relates to them, at entity level."""
        files, modules, entities = gold_sets(self.file_changes)
        reward, parts = self._score()
        wanted = {str(e).lower(): str(e) for e in set(entities)}
        named = _entities_submitted(self.locations)
        revealed = [orig for key, orig in sorted(wanted.items()) if key in named]
        return {
            "files": sorted({str(f) for f in files}),
            "modules": sorted({str(m) for m in modules}),
            "entities": sorted(wanted.values()),
            "submitted": bool(self.locations),
            "revealed": revealed,
            "missed": [orig for key, orig in sorted(wanted.items()) if key not in named],
            "wrong": sorted(name for key, name in named.items() if key not in wanted),
            "coverage": len(revealed) / len(wanted) if wanted else 0.0,
            "f1": reward / MAX_REWARD,
            "file_f1": parts["file_reward"],
            "module_f1": parts["module_reward"],
            "entity_f1": parts["entity_reward"],
        }


def _resolve(base: Path, value: Any) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


@dataclass
class CodeSearchDomain:
    name: str = NAME
    max_turns: int = MAX_TURNS
    max_reward: float = MAX_REWARD
    tools: Any = field(default_factory=lambda: TOOL_SPECS)
    max_parallel: int = 1
    variant: str = "full"
    reason: bool = False
    bash_timeout_s: int = DEFAULT_TIMEOUT_S
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def load_tasks(self, path: str, *, split: str = "train", limit: int | None = None) -> list[Task]:
        """Read a JSONL of ``{task_id, prompt, data: {repo_dir, gold}}``; see docs/domains.md.

        ``repo_dir`` is resolved against the JSONL's directory unless absolute. ``split`` is
        not used: one file holds one split.
        """
        del split
        base = Path(path).resolve().parent
        tasks = []
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            data = dict(row["data"])
            data["repo_dir"] = _resolve(base, data["repo_dir"])
            tasks.append(Task(task_id=str(row["task_id"]), prompt=row["prompt"], data=data))
        return tasks[:limit] if limit else tasks

    def task_key(self, task: Task) -> str:
        """The problem statement."""
        return task.prompt.strip()

    def make_env(self, task: Task) -> CodeSearchEnv:
        return CodeSearchEnv(task, max_turns=self.max_turns, max_parallel=self.max_parallel, variant=self.variant,
                             reason=self.reason, bash_timeout_s=self.bash_timeout_s,
                             max_output_bytes=self.max_output_bytes)


# ---------------------------------------------------------------------------------------------
# State DSL
#
# One Turn is one MODEL turn: the rationale it wrote and every command it issued, one Call per
# command. That is the unit the context budget drops in. parse_turn works per observation and
# returns a Call; finalize_turns groups Calls under their Turn, and expand_state flattens them
# back to one record per observation for re-serialization. The command itself is never
# rewritten: it rides on the assistant message's tool call.
# ---------------------------------------------------------------------------------------------

_SEARCH_CMDS = frozenset({"rg", "grep", "egrep", "fgrep", "ag", "ack"})
_READ_CMDS = frozenset({"cat", "sed", "head", "tail", "less", "more", "nl", "awk"})
_NAV_CMDS = frozenset({"ls", "find", "tree", "dir"})
_MEASURE_CMDS = frozenset({"wc", "stat", "file", "du"})

# The markers run_bash appends when a command floods its byte cap or hangs.
_TRUNC_RE = re.compile(r"output truncated at \d+ bytes|command timed out after \d+s", re.IGNORECASE)
_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(slots=True)
class Call:
    """One bash command issued in a turn, and its output.

    Slotted on purpose: the rationale belongs to the Turn, so ``call.reasoning = ...`` raises
    instead of silently doing nothing.
    """

    command: str               # the bash command (read-only: it rides on the tool call)
    cmd_kind: str              # 'search' | 'read' | 'nav' | 'measure' | 'other'
    output: str                # the command's stdout -- this is what costs tokens
    truncated: bool = False    # the executor already cut this output at its byte cap
    digest: str | None = None  # if set, replaces this call's output entirely
    keep: bool = False         # serve verbatim even when the enclosing Turn has a digest
    _msg_idx: int = -1
    _ai_idx: int = -1
    _turn_reasoning: str = ""  # parked here by parse_turn until finalize_turns moves it to the Turn


@dataclass(slots=True)
class Turn:
    """One model turn: the rationale it wrote, and every command it issued."""

    index: int                 # 0-based turn (0 = oldest)
    n_turns: int               # model turns in the episode so far
    reasoning: str             # the rationale written before this turn's commands
    calls: list[Call] = field(default_factory=list)
    digest: str | None = None  # if set, replaces every call in this turn
    _ai_idx: int = -1
    _orig_reasoning: str = ""


@dataclass(slots=True)
class _Served:
    """One observation as it is written back; not part of the CONTRACT."""

    output: str
    digest: str | None
    reasoning: str
    _msg_idx: int
    _ai_idx: int
    _orig_reasoning: str


CONTRACT = '''\
You write a Python function:

    def compress(state):
        """Transform the observation state of one code-localization episode and
        return the transformed list of turns."""
        ...
        return state

`state` is a list[Turn], oldest first. One Turn is one model turn. Each Turn has:
    .index      int    0-based turn position (0 = oldest)
    .n_turns    int    number of turns in the episode so far
    .reasoning  str    the rationale the agent wrote before this turn's commands ('' if the
                       run did not ask for reasoning); shortening it does not affect the
                       commands
    .calls      list[Call]   the commands this turn issued, in order
    .digest     str|None     set this to replace every call in the turn with a short string.
                             It applies only to calls that did not express their own intent:
                             a Call with .digest set, or with .keep True, is left alone.

A turn may issue several commands at once -- see "Commands per turn" in the measured evidence
for how often that happens in THIS run, because a cap on parallel calls makes it vary from
almost never to most turns. Where it is common, sibling commands in one turn tend to read
overlapping regions of the same files, and `.calls` often has several entries. Each Call has:
    .command    str    the bash command that produced this output (read-only)
    .cmd_kind   str    one of 'search','read','nav','measure','other'
    .output     str    the command's stdout (this is what costs tokens)
    .truncated  bool   True if the executor already cut this output at its byte cap
    .digest     str|None   set this to replace this one call's output
    .keep       bool       set True to serve this call verbatim even if you also set a
                           digest on its Turn -- the way to say "collapse this turn EXCEPT
                           the command that mattered"

You MAY: rewrite .output, rewrite or clear .reasoning, set .digest on a Call or a whole Turn,
mark a Call .keep, keep recent turns full. You MUST return a list[Turn]. You may not fabricate
output.

.command is read-only: it rides on the assistant message and is not yours to change. The
rationale lives on the Turn, not on a Call -- a Call has no .reasoning to set.

How your output is served -- three facts about the runtime, not advice:

  * You are called ONCE per model turn, on the WHOLE episode so far. `.n_turns` is the total
    number of turns, not a window size. After you return, a second stage measures the result
    and, if it still exceeds the token budget, drops whole OLDEST turns until it fits. So the
    more you compress, the more turns survive that drop.
  * You cannot remove a turn or a call. Leaving one out of the list you return does not
    delete it -- it is served unchanged. Dropping history is the second stage's job. Your
    lever is what each turn COSTS.
  * The LAST turn in `state` is what the agent must answer RIGHT NOW. It is the live
    question, not history. Compressing it does not save history cost; it edits the thing
    being asked.

The issue statement is not in `state` and cannot be modified.

    # naive: keep everything
    def compress(state):
        return state

    # syntax demo (NOT a strategy): stub the very first turn
    def compress(state):
        for t in state:
            if t.index == 0:
                t.digest = 'first turn'
        return state

    # syntax demo (NOT a strategy): dispatch on what a command was doing
    def compress(state):
        for t in state[:-2]:
            for c in t.calls:
                if c.cmd_kind == 'search':
                    c.output = '\\n'.join(c.output.splitlines()[:5])
                elif c.cmd_kind in ('nav', 'measure'):
                    c.digest = ''
        return state

    # syntax demo (NOT a strategy): collapse a wide turn but spare one sibling
    def compress(state):
        for t in state[:-1]:
            for c in t.calls:
                if c.cmd_kind == 'read':
                    c.keep = True
            t.digest = '(%d commands)' % len(t.calls)
        return state
'''


def _tokenize(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _command_words(command: str) -> tuple[str, str]:
    """The ``(primary, secondary)`` command words, skipping ``VAR=val`` prefixes."""
    toks = _tokenize(command)
    i = 0
    while i < len(toks) and _ENV_PREFIX_RE.match(toks[i]):
        i += 1
    primary = os.path.basename(toks[i]) if i < len(toks) else ""
    secondary = os.path.basename(toks[i + 1]) if i + 1 < len(toks) else ""
    return primary, secondary


def classify_command(command: str) -> str:
    """Classify a command by its leading utility; a pipeline is classified by its first command."""
    primary, secondary = _command_words(command)
    if primary == "git":
        if secondary == "grep":
            return "search"
        if secondary in ("ls-files", "ls-tree"):
            return "nav"
    if primary in _SEARCH_CMDS:
        return "search"
    if primary in _READ_CMDS:
        return "read"
    if primary in _NAV_CMDS:
        return "nav"
    if primary in _MEASURE_CMDS:
        return "measure"
    return "other"


def _command_for(obs_message: Message, ai_message: Message) -> str:
    """The command behind an observation, matched by ``tool_call_id`` rather than position."""
    tcid = obs_message.get("tool_call_id") or ""
    for call in tool_calls_of(ai_message):
        if call["name"] == TERMINAL_TOOL and call["id"] == tcid:
            return str(call["args"].get("command") or "")
    return ""


def is_observation(message: Message) -> bool:
    return message.get("role") == "tool"


def parse_turn(*, obs_message: Message, ai_message: Message | None, msg_idx: int, ai_idx: int,
               explicit_carrier: bool) -> Call:
    """One observation -> one :class:`Call`; ``finalize_turns`` does the grouping."""
    command, reasoning = "", ""
    if ai_message is not None:
        command = _command_for(obs_message, ai_message)
        reasoning, _ = split_reasoning(ai_message, explicit_carrier=explicit_carrier)
    output = content_of(obs_message)
    return Call(command=command, cmd_kind=classify_command(command), output=output,
                truncated=bool(_TRUNC_RE.search(output)), _msg_idx=msg_idx, _ai_idx=ai_idx,
                _turn_reasoning=reasoning)


def finalize_turns(calls: list[Call]) -> list[Turn]:
    """Group calls under the assistant message that issued them, keeping transcript order."""
    turns: list[Turn] = []
    by_ai: dict[int, Turn] = {}
    for call in calls:
        turn = by_ai.get(call._ai_idx) if call._ai_idx >= 0 else None
        if turn is None:
            turn = Turn(index=len(turns), n_turns=0, reasoning=call._turn_reasoning, _ai_idx=call._ai_idx,
                        _orig_reasoning=call._turn_reasoning)
            turns.append(turn)
            if call._ai_idx >= 0:
                by_ai[call._ai_idx] = turn
        turn.calls.append(call)
    for turn in turns:
        turn.n_turns = len(turns)
    return turns


def expand_state(turns: list[Turn]) -> list[_Served]:
    """Flatten to one record per observation. Precedence: keep > call digest > turn digest."""
    out: list[_Served] = []
    for turn in turns:
        for call in getattr(turn, "calls", None) or []:
            if call.keep:
                digest = None
            elif call.digest is not None:
                digest = call.digest
            else:
                digest = turn.digest
            out.append(_Served(output=call.output, digest=digest, reasoning=turn.reasoning,
                               _msg_idx=call._msg_idx, _ai_idx=turn._ai_idx,
                               _orig_reasoning=turn._orig_reasoning))
    return out


def render_turn(turn: _Served | Call) -> str:
    return turn.digest if turn.digest is not None else turn.output


def rebuild_obs(message: Message, text: str, turn: Any) -> Message:
    """Keep the ``tool_call_id``: it pairs this observation with the call that produced it."""
    del turn
    return {**message, "content": text}


def sample_messages() -> Messages:
    """Two turns, the second issuing two commands, so a one-observation-per-turn program fails at load."""
    return [
        {"role": "user", "content": "Localize the bug described in the issue."},
        {"role": "assistant", "content": "Search for the symbol.",
         "tool_calls": [make_tool_call("a1", TERMINAL_TOOL, {"command": 'rg -n "parse_config" src/'})]},
        {"role": "tool", "tool_call_id": "a1", "content": "src/config.py:42:def parse_config(path):"},
        {"role": "assistant", "content": "Read it and check the tests.",
         "tool_calls": [make_tool_call("b1", TERMINAL_TOOL, {"command": "sed -n 40,60p src/config.py"}),
                        make_tool_call("b2", TERMINAL_TOOL, {"command": "ls tests/"})]},
        {"role": "tool", "tool_call_id": "b1",
         "content": "def parse_config(path):\n    return json.load(open(path))"},
        {"role": "tool", "tool_call_id": "b2", "content": "test_config.py\ntest_cli.py"},
    ]


# ---------------------------------------------------------------------------------------------
# Stage-1 optimizer hooks
# ---------------------------------------------------------------------------------------------

DESC = (
    "A code-localization agent runs bash commands (grep/find to search, cat/sed to read "
    "files, ls to navigate) across many turns to find which file(s) contain a described "
    "bug. One turn may issue several commands at once, and each command contributes its own "
    "stdout block: read-command outputs are whole file bodies (large), while search/nav "
    "outputs are short hit-lists of paths.")

_STRUCTURAL = re.compile(r"\s*(import |from |def |class |@|async def )")


def gold_strings(gold: Any) -> list[str]:
    """The gold file path(s) of an episode, from its ``gold_view()`` (or a raw gold list)."""
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(gold, dict):
        if "files" in gold:
            return [str(f) for f in gold.get("files") or [] if f]
        gold = gold.get("gold", gold.get("file_changes", []))
    if isinstance(gold, list):
        return [str(c["file"]) for c in gold if isinstance(c, dict) and c.get("file")]
    return []


def label(turn: Turn) -> str:
    if not turn.calls:
        return "(no commands)"
    shown = "  ".join(f"[{c.cmd_kind}] `{c.command[:40]}` ({len(c.output)}c)" for c in turn.calls[:3])
    rest = len(turn.calls) - 3
    return f"{shown}  (+{rest} more)" if rest > 0 else shown


def turn_text(turn: Turn) -> str:
    return " ".join(f"{c.output} {c.command}" for c in turn.calls)


def _percentile(values: list[int], fraction: float) -> int:
    return values[min(len(values) - 1, int(fraction * len(values)))] if values else 0


def evidence(episodes: Sequence[Any]) -> tuple[dict[str, Any], str]:
    """Where the gold path first surfaces, read-output sizes, and turn/command counts."""
    dsl = sys.modules[__name__]
    with_gold = 0
    first_kind: dict[str, int] = {}
    read_sizes: list[int] = []
    structural = body = 0
    turn_counts: list[int] = []
    calls_per_turn: list[int] = []
    for ep in episodes:
        golds = gold_strings(ep.gold)
        found_kind = None
        turns = parse_state(ep.messages, dsl)
        if turns:
            turn_counts.append(len(turns))
        for turn in turns:
            calls_per_turn.append(len(turn.calls))
            for call in turn.calls:
                if call.cmd_kind == "read":
                    read_sizes.append(len(call.output))
                    for line in call.output.splitlines():
                        if _STRUCTURAL.match(line):
                            structural += 1
                        else:
                            body += 1
                # Attributed to the command that surfaced it: one turn mixes command kinds.
                if found_kind is None and golds and any(g in f"{call.output} {call.command}" for g in golds):
                    found_kind = call.cmd_kind
        if found_kind:
            with_gold += 1
            first_kind[found_kind] = first_kind.get(found_kind, 0) + 1
    read_sizes.sort()
    turn_counts.sort()
    calls_per_turn.sort()
    multi_call = sum(1 for c in calls_per_turn if c > 1)
    stats: dict[str, Any] = {
        "episodes_total": len(episodes), "episodes_with_gold": with_gold,
        "gold_first_seen_in": first_kind,
        "read_output_chars": {"p50": _percentile(read_sizes, .5), "p90": _percentile(read_sizes, .9),
                              "max": read_sizes[-1] if read_sizes else 0},
        "read_structural_line_frac": round(structural / max(structural + body, 1), 3),
        "turns_per_episode": {"p50": _percentile(turn_counts, .5), "max": turn_counts[-1] if turn_counts else 0},
        "commands_per_turn": {"p50": _percentile(calls_per_turn, .5),
                              "max": calls_per_turn[-1] if calls_per_turn else 0,
                              "multi_frac": round(multi_call / max(len(calls_per_turn), 1), 3)},
    }
    kinds = ", ".join(f"{k}: {v}" for k, v in first_kind.items()) or "(none)"
    text = (
        f"Measured over {stats['episodes_total']} rollouts "
        f"({stats['episodes_with_gold']} with a localizable gold file).\n"
        f"Command kind where the gold file path first surfaces: {kinds}\n"
        f"Read-output size (chars): p50={stats['read_output_chars']['p50']}, "
        f"p90={stats['read_output_chars']['p90']}, max={stats['read_output_chars']['max']}\n"
        f"Structural lines (import/def/class/decorator) are "
        f"{stats['read_structural_line_frac']:.0%} of read-output lines.\n"
        f"Turns per episode: median={stats['turns_per_episode']['p50']}, "
        f"max={stats['turns_per_episode']['max']} -- a rule that keeps the last N turns "
        f"full leaves nothing to compress once N reaches the median.\n"
        f"Commands per turn: median={stats['commands_per_turn']['p50']}, "
        f"max={stats['commands_per_turn']['max']}; "
        f"{stats['commands_per_turn']['multi_frac']:.0%} of turns issue more than one, "
        f"and sibling commands in one turn often read overlapping regions of the same files.")
    return stats, text


register_state_dsl(NAME, sys.modules[__name__])
register_domain(CodeSearchDomain())

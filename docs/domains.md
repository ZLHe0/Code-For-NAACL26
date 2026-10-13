# Domains

A domain is one module under `harness_programs/domains/` that provides:

1. an **environment** (`reset`, `step`, `reward`, `final_output`, `gold_view`);
2. a **`Domain`** dataclass (`name`, `max_turns`, `tools`, `load_tasks`, `task_key`, `make_env`);
3. a **state DSL**, the typed view of a transcript that compression programs edit (`CONTRACT`, `Turn`,
   `is_observation`, `parse_turn`, `render_turn`, `rebuild_obs`, `sample_messages`);
4. optional **stage-1 optimizer hooks** (see below).

Importing the module registers it. `harness_programs.domains.load(name)` imports and returns it.

| domain | observation message | action | reward | `max_turns` | extra dependency |
|---|---|---|---|---|---|
| `codesearch` | `role: "tool"`, one per tool call | `terminal{command}`, `localization_finish{locations}` tool calls | file + module + entity localization F1, in [0, 3] | 10 | `bash` (`rg` optional) |
| `sql` | `role: "user"`, `<observation>…<reminder>…</reminder></observation>` | `<action>SQL</action>`, `<answer>SQL</answer>` | execution match with the gold SQL, 0/1 | 6 | none |
| `alfworld` | `role: "user"`, the full step prompt (feedback + admissible menu + footer) | `<action>text</action>` | game won, 0/1 | 50 | `alfworld` (+ its game data) |
| `toy` | `role: "user"`, `<observation>…</observation>` | `<action>…</action>` | exact answer, 0/1 | 8 | none |

CodeSearch and SQL ask for no free-text reasoning by default (`reason=False`). ALFWorld asks for
one sentence of reasoning (`reason=True`). Each domain dataclass exposes its knobs as fields,
for example `SqlDomain(obs_format="table", match_mode="refined")`.

## Task files

`Domain.load_tasks(path, split=..., limit=...)` reads a JSONL file with one task per line:

```json
{"task_id": "...", "prompt": "...", "data": {...}}
```

A relative path inside `data` is resolved against the JSONL file's directory, and an absolute
path is used as is. The release contains no download or curation code: build the JSONL from your
own copy of a benchmark.

`Domain.task_key(task)` is the task's identity, compared by content: the prompt for CodeSearch,
the database and question for SQL, the game file for ALFWorld. `harness_programs.splits` uses it to carve
D_hold out of the training file and to check that D_hold shares no task with D_train or with an
evaluation set (`docs/overview.md`).

### `codesearch`

```json
{"task_id": "repo__issue-1",
 "prompt": "<issue text>",
 "data": {"repo_dir": "repos/repo__issue-1",
          "gold": [{"file": "src/pkg/config.py",
                    "changes": {"edited_modules": ["src/pkg/config.py:Loader"],
                                "edited_entities": ["src/pkg/config.py:Loader.parse"]}}]}}
```

- `repo_dir`: a checkout of the repository at the issue's base commit.
  - Commands run with this directory as the working directory, behind a timeout, an output byte
    cap, a process-group kill and per-process resource limits.
  - The checkout is not isolated or restored, so give concurrent episodes separate copies if the
    policy may write.
- `gold`: the edited files. It may be a JSON string, and `file_changes` is accepted as an alias.
  - A module is named `file:Class`, or `file:function` for a top-level function.
  - An entity is named `file:Class.method` or `file:function`.
- Reward: the sum of F1 at the file, module and entity levels, in [0, 3]. Evaluation reports
  `mean_score`, their mean in [0, 1], which is the metric the paper reports.
  - The policy submits a list of `{file, class_name?, function_name?}`.
  - If any submitted location has no `file`, the whole submission counts as empty.
  - `gold_view()["f1"]` reports the reward divided by 3.

### `sql`

```json
{"task_id": "q-17", "prompt": "<question, schema hints, evidence>",
 "data": {"db_path": "dbs/shop.sqlite", "gold_sql": "SELECT COUNT(*) FROM users WHERE city = 'oslo'"}}
```

- `prompt` is inserted into the instruction template as the task description.
- The database is opened read-only, and every query runs inside a transaction that is rolled back.
- Observations show up to 10 rows as JSON (`obs_format="json"`) or the row set as a table
  (`"table"`). Either way they are capped at `max_obs_chars`.
- Reward: 1 if the submitted `<answer>` SQL executes and its result **set** equals the gold
  result set (`match_mode="exact"`), else 0.
  - `"refined"` mode additionally requires equal row counts and ignores column order.

### `alfworld`

```json
{"task_id": "valid_seen-0", "prompt": "",
 "data": {"game_file": "json_2.1.1/valid_seen/<task>/<trial>/game.tw-pddl",
          "split": "eval_in_distribution"}}
```

- `game_file`: a `game.tw-pddl` file from the ALFWorld data tree.
- `split`: `train`, `eval_in_distribution` or `eval_out_of_distribution`. If omitted, it is taken
  from the `split=` argument of `load_tasks`.
- `data_dir` (optional): the ALFWorld data root (`json_2.1.1/`, `logic/`). It falls back to
  `AlfworldDomain(data_dir=...)` and then to `$ALFWORLD_DATA`.
- `AlfworldDomain(config_path=...)` replaces the built-in text-environment config with your own
  YAML file. Environment variables in that file are expanded.
- `prompt` is unused, because the game states the task in its first observation.
- Install `alfworld` separately. It is imported only when an episode starts.

## Adding a domain

1. Copy `harness_programs/domains/toy.py` to `harness_programs/domains/<name>.py` and add `<name>` to `AVAILABLE` in
   `harness_programs/domains/__init__.py`.
2. Write the environment. `reset()` returns the opening messages, which become the pinned head.
   `step(assistant)` returns a `StepResult` with the new observation messages and `done`.
   `gold_view()` must be JSON-serializable.
3. Write the state DSL. One `Turn` per observation, holding:
   - the editable handles (text that costs tokens, plus `reasoning` and `digest`);
   - private provenance (`_msg_idx`, `_ai_idx`, `_orig_reasoning`).

   `render_turn` must return the original text when nothing changed. The no-op program must then
   be an exact identity. Programs may never change the transcript length, the pinned head or any
   tool call.
4. Optional:
   - `finalize_turns(turns)` to regroup or annotate the parsed turns. CodeSearch uses it to group
     several tool outputs under one model turn.
   - `expand_state(turns)` to flatten them back to one record per observation.
5. Make `sample_messages()` exercise every structural case: several calls in one turn, a reminder
   line, a pinned turn 0, and so on. Each program is smoke-run on it before it is served.
6. Define `task_key(task)` so that two rows of the same task get the same key, whatever their
   `task_id`. The D_hold check compares these keys.
7. At the bottom of the module, call `register_state_dsl(NAME, sys.modules[__name__])` and
   `register_domain(<Domain>())`.
8. Test it the way `tests/test_domain_*.py` do:
   - the no-op program is an identity on `sample_messages()`;
   - a digest program preserves actions;
   - unit tests cover the reward;
   - one environment step runs on a tiny fixture.

## Stage-1 optimizer hooks

The compression optimizer (`python -m harness_programs.compression`) reads these optional module-level
names from the state-DSL module with `getattr`. A domain without them still works and falls back
to generic behaviour.

| hook | domains | used for |
|---|---|---|
| `DESC: str` | all | 2-3 sentences describing what an observation *is* (structure only, no strategy). It opens the proposer prompt. |
| `label(turn) -> str` | all | a one-line label per turn in the worked-episode overlay |
| `turn_text(turn) -> str` | all | the text searched for gold strings when marking which turn revealed the answer |
| `gold_strings(gold) -> list[str]` | `codesearch`, `toy` | the gold string(s) to look for. `gold` is the episode's `gold_view()`. |
| `parse_actions(content) -> list[str]` | `sql`, `alfworld`, `toy` (free-text actions) | the actions an assistant message took, `[]` if none. The optimizer checks that a program leaves every action intact. Tool-call domains omit it, because tool calls are compared directly. |
| `PINS_TURN0 = True` | `alfworld` | turn 0's observation lives in the pinned head, which is never evicted |
| `evidence(episodes) -> (dict, str)` | `codesearch` | a domain-specific evidence block that replaces the generic one: where the gold file first surfaces, read-output sizes, structural-line fraction, and turns and commands per turn. `episodes` have `.messages` and `.gold`. Parse them with `harness_programs.serving.compress.parse_state(messages, dsl)`. |

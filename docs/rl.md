# RL: the trainer bridge (`harness_programs.rl`)

Stage 2 trains the policy with GRPO while the served context comes from the harness
pipeline. The policy sees exactly the context it will see at evaluation: compressed, budgeted,
and steered. `harness_programs.rl` is a [verl](https://github.com/volcengine/verl) **agent loop** that
plays one multi-turn episode of a domain per dataset row, with
`harness_programs.serving.ContextPipeline` between the environment and the model.

It targets **verl 0.8.x**: the `verl.experimental.agent_loop` API, the async vLLM rollout, and
the PPO ray trainer launched through `python -m verl.trainer.main_ppo`. Some 0.8 builds also ship
a newer trainer entry point. The changes described below apply to the ray-trainer path.

| module | what it holds | needs verl |
|---|---|---|
| `agent_loop.py` | `EpisodeRunner` (the episode loop), `HarnessAgentLoop` (the verl adapter), dataset-row helpers, settings | adapter only |
| `traces.py` | episode → training rows, instance-level GRPO advantage, batch bookkeeping | no |
| `parsing.py` | Hermes `<tool_call>{json}</tool_call>` parser and text-action passthrough | no |
| `step_marker.py` | training-step marker and the per-step episode dump | no |

`import harness_programs.rl` works without verl. Constructing `HarnessAgentLoop` without verl raises
an `ImportError` that says what is missing.

## One episode

`HarnessAgentLoop.run(sampling_params, **row)` does the following:

1. Rebuilds the row's `Task` from `extra_info.task` (`task_from_row`) and creates the
   environment with `harness_programs.domains.load(name).make_env(task)`. The domain's dataclass gets the
   entry's `domain_options` and the run's turn cap, so the "turns left" text the environment shows
   matches the loop.
2. Starts the episode. With the pipeline on, it opens one `EpisodeContext` from
   `ContextPipeline.new_episode()`. The pipeline is shared by every episode of a worker and re-reads
   `program.json` when the file changes, so a trigger installed mid-run applies from the next episode.
3. Runs each turn:
   1. `EpisodeContext.serve(transcript)` produces the served messages: the compression, the budgeted
      window, then the guidance tail.
   2. The tokenizer's chat template renders them in an executor thread, using
      `data.apply_chat_template_kwargs`, plus the domain's tool schemas when it has any.
   3. The policy is sampled token-in/token-out with
      `server_manager.generate(request_id=..., prompt_ids=..., sampling_params=...)`. The token ids
      and log-probs are kept as sampled; they are never re-tokenized. All turns of an episode share
      one `request_id`, so the server routes them to the same replica and its prefix cache.
   4. The decoded text becomes an assistant message. Tool-call domains parse Hermes blocks into
      OpenAI `tool_calls` with ids `call_<turn>_<i>`, capped at `multi_turn.max_parallel_calls`.
      Text-action domains pass the text through.
   5. `env.step(message)` runs in an executor thread and its observations extend the transcript.
4. Ends the episode when the environment says done (`answered`), at the turn cap (`max_turns`), or
   on an error (`error`).
5. Writes the result:
   - `env.reward()` becomes the reward;
   - the episode is appended to the step's dump (below);
   - the turns become training rows (next section).

Rewards come from the environment. Every row carries a `reward_score`, so verl builds its reward
tensor from them and never calls a reward function.

Failures are contained to the sample:
- An environment exception, a prompt that cannot be served, or a prompt longer than
  `data.max_prompt_length` ends the episode with status `error`.
- The episode is still dumped, with an extra `error` column.
- It returns one placeholder row with `response_mask=[0]`, which trains on nothing.
- A row that cannot even be parsed degrades the same way.

### Dataset rows

```python
import pandas as pd
from harness_programs.rl import make_rows

# the training split minus D_hold (python -m harness_programs.splits carve); D_hold is never trained on
pd.DataFrame(make_rows("sql", "tasks/sql_train_policy.jsonl", split="train")).to_parquet("data/sql/train.parquet")
pd.DataFrame(make_rows("sql", "tasks/sql_val.jsonl", split="val")).to_parquet("data/sql/val.parquet")
```

Each row has these columns:
- `data_source`: the domain name.
- `prompt`: what verl's overlong-prompt filter measures.
- `reward_model`: `{"style": "environment"}`.
- `extra_info`: `{index, split, validate, task_id, task}`, where `task` holds the JSON of
  `{task_id, prompt, data}`.
- `agent_name` (optional): selects the agent-loop entry for that row.

A row may carry only `extra_info.task_id` if `tasks_path` points at the task file.
`docs/domains.md` gives each domain's task schema.

## Traces: one row per turn

Stock multi-turn RL concatenates an episode into one row: the first prompt, then each generation
followed by the observation tokens after it, which are masked out of the loss. That is correct
only when every turn's prompt extends the previous prompt plus response plus observation.

Under a context budget, that does not hold. Each turn's served prompt is rebuilt from scratch: the
pinned head, a compressed and partly evicted history, and an ephemeral guidance tail. Training on
the concatenation would score each generation under a prompt it was never sampled from.

With the pipeline on, an episode therefore becomes **one row per turn** (`trace_mode: per_turn`):

* `prompt_ids` is the prompt served that turn, `response_ids` is that turn's generation, and
  every response token is trainable (`response_mask` all ones).
* The episode's reward sits on its last row, the **reward trace** (`extra_fields.is_reward`).
  The other rows carry score 0, plus the episode's outcome in `extra_fields.episode_reward` for
  logging.
* The trainer computes an **instance-level GRPO advantage**:
  1. Reduce each episode (instance) to its reward trace's score.
  2. Group the instances by the dataset row they were sampled for (`uid`), and take the group's
     mean and std over instances, not rows.
  3. Broadcast the episode's normalized advantage to all of its rows.

  Stock per-row GRPO would count an episode once per turn in its group's baseline, and would give
  the non-final turns an advantage computed from a score of 0.
  `harness_programs.rl.traces.instance_grpo_advantages` is the reference implementation. When every
  episode is a single row, it reduces to stock GRPO (tested).

Without the pipeline, `trace_mode: merged` (or `auto`) emits the stock single row. This is valid
because the transcript is then append-only (`traces.is_append_only`). `trace_mode: auto` picks
`per_turn` exactly when `context_budget.enable` is set. Per-turn traces are also valid without a
budget. The paper's runs used them by default, including for the unmanaged baseline, so that all
arms share one trace layout, and the shipped configs set `per_turn` explicitly.

A per-turn step holds a varying number of rows, which constrains the trainer. The adapter checks
these at construction (`per_turn_config_errors`):

- `algorithm.adv_estimator=grpo`;
- `actor.loss_agg_mode=token-mean`, because sequence-mean modes divide by a fixed batch size;
- `actor.use_dynamic_bsz=true` or a micro-batch of 1;
- `data.train_batch_size` a multiple of `actor.ppo_mini_batch_size`.

## The verl change this needs

Stock verl 0.8.x accepts **one** `AgentLoopOutput` per sample, and the trainer unions the
generated batch with the prompt batch row-for-row. Per-turn traces return a list, so verl needs
the change below. We do not ship verl source. The change is small, and `harness_programs/rl/traces.py`
contains the pure-Python reference for every computation it adds.

**`verl/experimental/agent_loop/agent_loop.py`**

1. Add a module-level `SUPPORTS_MULTI_OUTPUT = True`. The adapter refuses per-turn mode unless
   this flag is present (`require_multi_output_patch: false` skips the check).
2. In `AgentLoopWorker._run_agent_loop`, `agent_loop.run(...)` may now return a list. Run
   `_agent_loop_postprocess` on each element, which pads prompt and response to the configured
   widths, and return the list. Also set `agent_loop.trajectory_info = dict(trajectory)` before
   `run`; this hands the loop its step, replicate index and validation flag (see "Step plumbing").
3. In `AgentLoopWorker.generate_sequences`, flatten the gathered results before `_postprocess`
   into `(rows, parent_index, is_reward)` (reference: `traces.flatten_outputs`).
4. In `AgentLoopWorker._postprocess`, the input batch's per-sample columns (`uid`, `index`,
   `data_source`, `extra_info`, …) currently have one entry per *sample*. Gather them by
   `parent_index` so each row inherits its sample's values. Also add two columns:
   `__instance_id__` (the sample's position) and `__is_reward_trace__`.

**`verl/trainer/ppo/ray_trainer.py`**

1. `RayPPOTrainer.fit`, after generation: `batch.repeat(n, interleave=True)` has one row per
   *sample*. Index it with the output's `__instance_id__` (`batch.select_idxs(...)`) before
   `batch.union(gen_output)`, instead of assuming equal lengths.
2. `RayPPOTrainer.fit`, before balancing and the update: pad the row count to a multiple of
   `dp_size × num_mini_batch` (`traces.pad_size`). Pad with clones of real rows (as verl's
   `pad_dataproto_to_divisor` makes them) that have a zeroed `response_mask` and the flag
   `__is_pad__=True`, so the clones carry every column but no gradient and no advantage.
3. Module-level `compute_advantage`, GRPO branch: when `__instance_id__` is present, use the
   instance-level advantage above (`traces.instance_grpo_advantages`), excluding pad rows.
4. The actor update: split the step's rows into `num_mini_batch = train_batch_size //
   ppo_mini_batch_size` mini-batches (`traces.num_mini_batch`) instead of fixed mini-batches of
   `ppo_mini_batch_size × n` rows. Each step then makes the same number of optimizer updates as the
   single-row baseline, whatever its row count. The paper's configs set
   `ppo_mini_batch_size = train_batch_size`, i.e. one update per step.
5. `RayPPOTrainer._validate`: collapse each instance to its reward trace before the union and
   before computing validation metrics.
6. Reward and score metrics: compute them over reward traces only. Log
   `traces.traces_per_instance_stats` for visibility.

With per-turn mode off, none of this is exercised. The merged mode runs on stock verl.

## Step plumbing and episode dumps

The steering supervisor (`python -m harness_programs.steering supervise`) reads training rollouts by
training step. It reads them from `<run_dir>/rollouts/step_<N>.jsonl`, the files this loop
appends to.

* **Step.** If the runtime sets `trajectory_info` (verl change 2 above), the loop takes `step`,
  `rollout_n` and `validate` from it. Otherwise it reads `<run_dir>/global_step`, which the trainer
  writes at the top of every training step. That needs one line at the start of the
  `for batch_dict in self.train_dataloader:` body of `RayPPOTrainer.fit`:

  ```python
  from harness_programs.rl.step_marker import write_global_step
  write_global_step(run_dir, self.global_steps)   # run_dir = config.harness_programs.run_dir
  ```

  The write is atomic (temp file plus rename). An episode that finds no marker, such as the
  pre-training validation, goes to `rollouts/step_unknown.jsonl`, which the supervisor ignores.
* **Dump.** Each finished episode is appended as one JSON line in the
  `harness_programs.rollout.EpisodeRecord` format:
  - `messages` is the full, unmanaged transcript;
  - `served` holds the per-turn budget stats plus `fired` trigger names;
  - `triggers` holds `setup` and per-turn `firings`;
  - `context_budget` is the history budget;
  - also `reward`, `status`, `final_output` and `gold`;
  - `rollout = {step, group_key, replicate, validate}`.

  `group_key` is the task id. `replicate` is the trajectory's `rollout_n`, or a per-process counter
  without it. Lines are written with `O_APPEND` under an advisory lock, so concurrent workers never
  interleave. `harness_programs.rollout.read_jsonl` and `harness_programs.rl.step_marker.read_step` read them back.

## Configuration

`configs/agent_loop.yaml` registers one entry per domain. verl passes each entry's keys to the
constructor:
- `domain` and `domain_options` (e.g. `{reason: true}`);
- `max_turns`;
- `per_turn_max_tokens`.

The run-level settings live in a top-level `harness_programs:` block of the trainer config, which
overrides the entry:

```yaml
harness_programs:
  trace_mode: per_turn            # per_turn | merged | auto
  context_budget: {enable: true, history_budget: 1024, trim_overflow: false}
  program_path: ${RUN_DIR}/program.json
  program_dynamic: true           # serve without a program until the file exists
  run_dir: ${RUN_DIR}             # step marker + rollouts/
  max_per_episode: 0              # trigger-firing caps (0 = none)
  max_per_trigger: 0
```

Limits come from verl's rollout config:

| verl setting | used as |
|---|---|
| `data.max_prompt_length` | the stored prompt width; a longer served prompt is an error |
| `data.max_response_length` | the response width |
| `rollout.max_model_len` | the model window |
| `rollout.multi_turn.max_assistant_turns` | the turn cap (it wins over the entry and the domain) |
| `per_turn_max_tokens` | caps each turn's generation |

The pipeline's serving limit is `serving_limit_from(prompt_length, max_model_len,
per_turn_max_tokens)`.

`configs/grpo_{codesearch,sql,alfworld}.yaml` are complete trainer configs. They read
`${MODEL_PATH}`, `${RUN_DIR}` and `${DATA_DIR}` from the environment:

```bash
export MODEL_PATH=<base model> RUN_DIR=<run dir> DATA_DIR=<data dir>
cp <stage-1 out>/program.json "$RUN_DIR/program.json"        # H_0; optional
python -m verl.trainer.main_ppo --config-path="$PWD/configs" --config-name=grpo_sql
python -m harness_programs.steering supervise --run-dir "$RUN_DIR" --program "$RUN_DIR/program.json" \
    --domain sql --total-steps 200 --tasks <train_policy.jsonl> --holdout <train_holdout.jsonl> ...
```

`scripts/run_pipeline.sh` does the same and also copies stage 1's rollouts to
`$RUN_DIR/rollouts/step_0.jsonl`, so that R_0 is part of the reservoir that Replay reads.

### Paper hyperparameters

All runs used:
- 200 steps on one node of 8 GPUs;
- GRPO with no KL in the reward, lr 1e-6, and one optimizer update per step
  (`ppo_mini_batch_size = train_batch_size`);
- vLLM async rollout at TP 1 with rollout log-probs and eager mode;
- dynamic batch size, gradient checkpointing, and padding removal;
- thinking off in the chat template;
- greedy validation every `test_freq` steps, plus once before training.

| | CodeSearch | SQL | ALFWorld |
|---|---|---|---|
| models | Qwen3-4B, Qwen3-8B | Qwen3.5-2B, Qwen3.5-4B | Qwen2.5-3B/7B-Instruct |
| tasks per step (`train_batch_size`) | 8 | 128 | 16 |
| group size (`rollout.n`) | 8 | 5 | 8 |
| rollout temperature | 1.0 | 1.0 | 0.6 |
| turn cap | 10 | 6 | 50 |
| `per_turn_max_tokens` | 1024 | 512 | 512 |
| `max_prompt_length` (E=1024 / larger E) | 4096 / 12288 (E=8192) | 4096 / 12288 (E=8192) | 2048 / 12288 (E=null) |
| `max_response_length` | 1024 | 2048 | 2048 |
| `max_model_len` (E=1024 / larger E) | 40960 | 16384 / 40960 | 16384 / 32768 |
| KL loss (low-variance) | 0.001 | off | 0.01 |
| entropy bonus | 0 | 0 | 0.001 |
| FSDP param + optimizer offload | yes | yes | no |
| vLLM `gpu_memory_utilization` | 0.6 | 0.5 | 0.4 |
| `test_freq` / `save_freq` | 5 / 50 | 20 / 50 | 5 (10 at E=null) / 25 |
| domain options | `reason: true` | `reason: true` | (default `reason: true`) |

E is `context_budget.history_budget`, the history tokens allowed above the pinned head. `E=null`
keeps the pipeline on but caps only at the serving limit.

## Simplifications

* **Reward.** Rewards come from `env.reward()` for every domain. They are not a separate verl
  reward function graded on the transcript.
* **No per-episode wall-clock timeout.** An environment call that hangs holds its sample. Set
  the domain's own timeouts (e.g. `CodeSearchDomain.bash_timeout_s`).
* **Failed episodes stay in their group as zero-gradient reward traces with score 0.** They
  lower the group baseline the way a failed episode would. Dropping them instead is a trainer-side
  choice.
* **Each per-turn row carries the whole episode's timing metrics.** verl aggregates them per row.
  Max and min are unaffected, but the means weight long episodes more.

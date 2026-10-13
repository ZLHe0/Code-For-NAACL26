# Harness Programs for Agentic RL

Code for **learning and evolving the agent harness as programs during agentic RL**.

The harness is represented as two programs over the agent's typed state (the task plus every
turn so far):

* **Compress** — re-renders earlier turns so the evidence the task needs fits the context
  budget, offloading what is no longer needed.
* **Steer** — a set of *triggers*. Each trigger is a small program that detects one
  failure mode in the trace, paired with a short piece of guidance. The guidance is appended to
  the context only on turns where it fires.

Both programs are deterministic Python. They make no model calls, so they run on every turn of
every rollout. They are learned from the policy's own rollouts in two stages:

1. **Stage 1, compression optimization** (before training).
   *Rollout*: the base policy plays the training tasks under the append-only context.
   *ProposeCompress*: a creator LLM contrasts the successful and failed attempts on each task and
   writes candidate programs; each is repaired until it runs, applied to stored episodes, and
   accepted or revised after seeing the contexts before and after it, with two measurements
   (is it *unfaithful*, is it *inert*). *Gate*: each accepted program gets one trial on D_hold,
   the held-out training tasks, under the frozen base policy; the best becomes H0 if it beats no compression, and is
   held fixed.
2. **Stage 2, RL with steering** (during training), every round:
   *UpdatePolicy*: GRPO under the current harness. *ProposeSteer*: the creator reads
   mixed-outcome groups and proposes failure hypotheses, each a trigger plus its guidance.
   *Replay*: every candidate and installed trigger runs over all stored rollouts, giving each
   failure's rate per training step: a curve that is falling, holding steady or rising.
   *Admit*: a judge reads those curves and returns intervene, monitor or continue.
   *Gate*: the admitted triggers are accepted together only if they score strictly above the
   incumbent on D_hold. Accepted triggers are installed into `program.json`, which the
   trainer re-reads every episode.

This code accompanies the paper and is meant to be read beside it: each operator of the paper's
Algorithm 1 is one function, listed in `docs/overview.md`. It ships no data, checkpoints or
programs from the paper's runs.

## Install

```bash
pip install -e .              # core: serving, both optimizers, toy domain (needs only pyyaml)
pip install -e ".[dev]"       # + pytest
pip install -e ".[rl]"        # + verl, torch, transformers, pandas for stage-2 training
pip install -e ".[alfworld]"  # + the ALFWorld environment
```

Stage 2 needs a verl build with the multi-output agent-loop change described in `docs/rl.md`.
Without it, training runs with one merged trace per episode.

## Quick start (offline, no GPU, no API)

```bash
python -m pytest                      # the full suite runs on CPU without network access
python examples/toy_pipeline.py       # stage 1 + a trigger on a toy domain, scripted agents
```

The toy example exercises the mechanics end to end: stage 1 (rollout, propose, gate), then
serving with a trigger. The toy domain is built to be solvable, so its numbers
demonstrate the plumbing, not the method.

## Running the pipeline

Both stages talk to models through OpenAI-compatible endpoints:

```bash
export HARNESS_LLM_BASE_URL=... HARNESS_LLM_MODEL=... HARNESS_LLM_API_KEY=...   # creator LLM
export HARNESS_POLICY_BASE_URL=http://127.0.0.1:8000/v1 HARNESS_POLICY_MODEL=policy
MODEL_PATH=<base model> scripts/serve_policy.sh &                                      # vLLM
```

Then, with task files in the schema of `docs/domains.md`:

```bash
DOMAIN=codesearch MODEL_PATH=<base model> WORK=<work dir> \
TRAIN_TASKS=<train.jsonl> VAL_TASKS=<val.jsonl> SERVING_LIMIT=4096 HISTORY_BUDGET=1024 \
  scripts/run_pipeline.sh
```

Both gates score on **D_hold**, a set of held-out *training* tasks. `run_pipeline.sh` first carves
`HOLDOUT_N` (default 128) tasks out of `TRAIN_TASKS` into `<train>_holdout.jsonl`, and the policy
trains on the rest, `<train>_policy.jsonl`. `VAL_TASKS` is used only for evaluation. It is never
D_hold, and every entry point refuses a D_hold that shares a task with D_train or with an
evaluation file (`python -m harness_programs.splits`, see `docs/overview.md`).

Or run each step on its own:

| step | script | output |
|---|---|---|
| verl rows | `scripts/prepare_data.sh` | `$DATA_DIR/<domain>/{train,val}.parquet` |
| stage 1 | `scripts/run_compression.sh` | `$OUT/program.json`, plus one resumable artifact per phase |
| stage 2 | `scripts/run_steering_rl.sh` | checkpoints, `rollouts/step_<N>.jsonl`, `steering/round_<N>/`, the evolving `program.json` |
| evaluation | `python -m harness_programs.eval` | per-task records and a summary |

Every script documents its environment variables at the top and passes extra arguments through
to the underlying command. `run_pipeline.sh` also runs the single-program arms: `STEERING=0`
trains under the stage-1 compression only, and `COMPRESSION=0` skips stage 1.

## Layout

```
harness_programs/
  serving/      the served context: sandbox, compression contract, budget, triggers, program.json
  domains/      codesearch, sql, alfworld, toy: environments + state DSLs
  compression/  stage 1
  steering/     stage 2 optimizer
  rl/           stage 2 trainer bridge (verl agent loop, per-turn traces, episode dumps)
  rollout/      plain multi-turn rollouts against an endpoint
  llm/          OpenAI-compatible client, offline mock
  eval.py
configs/        verl trainer configs and the agent-loop registry
scripts/        the pipeline
docs/           overview, compression, steering, rl, domains
examples/       offline toy pipeline
tests/
```

## Scope of this release

* No datasets, checkpoints or generated programs are included. Task-file schemas are in
  `docs/domains.md`.
* Generated code is screened, with an import allow-list, banned builtins and a time limit,
  against accidental misuse. This is not a security sandbox. Run untrusted programs in an
  isolated environment.

## License

To be determined.

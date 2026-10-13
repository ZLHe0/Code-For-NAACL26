# Overview

This code treats the agent harness as two programs over a typed **state**
`s_t = (task, u_0, …, u_{t-1})`, where each turn `u_i` is an action and its observation:

* **Compress** reads the state and returns a new view of it that fits the context budget,
  re-rendering earlier turns so the evidence the task needs is kept and the rest is offloaded.
* **Steer** is a set of *triggers*. Each trigger is a small program over the state
  that fires on one detectable failure mode, paired with a short piece of guidance. On a turn
  where a trigger fires, its guidance is appended after the compressed turns.

The policy acts on `flatten(Compress(s_t) ⊕ Steer(s_t))`. When that still exceeds the budget,
the oldest turns are dropped until it fits. Both programs are deterministic and run without a
model call, so they run on every turn of every rollout. Because they read the full state, a turn
that one view offloads can still be used to build later views.

## The two stages: Algorithm 1

```
            training tasks + base policy pi_0
                     │
     ┌───────────────▼────────────────┐
     │ Stage 1  compression           │  python -m harness_programs.compression
     │ Rollout → ProposeCompress →    │  scripts/run_compression.sh
     │ Gate                           │
     └───────────────┬────────────────┘
                     │  program.json = H_0 {compression, triggers: []}
                     │  rollouts      = R_0 (step_0.jsonl)
     ┌───────────────▼────────────────┐
     │ Stage 2  for k = 1..K          │  scripts/run_steering_rl.sh
     │  UpdatePolicy (GRPO under H)   │   trainer side:   harness_programs.rl
     │  ProposeSteer → Replay →       │   optimizer side: harness_programs.steering
     │  Admit → Gate                  │
     └───────────────┬────────────────┘
                     │  final checkpoint pi_K, program.json = H_K
                     ▼
            python -m harness_programs.eval
```

Each line of the paper's Algorithm 1 is one function:

| line | operator | code |
|---|---|---|
| 1 | `R_0 ← Rollout(pi_0, D_train)` | `compression.campaign.Campaign.rollout` → `compression.rollout.run` |
| 2 | `P ← ProposeCompress(R_0)` | `Campaign.propose` → `compression.propose.run`: `propose.draft` (contrast successes and failures per task, draft candidates), `refine.refine_one` (repair until it runs; accept or revise), `measure.measure` (unfaithful, inert, before/after contexts) |
| 3 | `H_0 ← Gate(pi_0, ∅, P, D_hold)` | `Campaign.gate` → `compression.gate.run` (one held-out trial per program and for no compression), `gate.select` (best, strictly above no compression), `emit.run` |
| 5 | `(pi_k, R_k) ← UpdatePolicy(pi_{k-1}, H_{k-1}, D_train)` | the GRPO trainer (`configs/grpo_<domain>.yaml`) with `rl.agent_loop.HarnessAgentLoop`, which serves every turn through `program.json` and dumps `rollouts/step_<N>.jsonl` |
| 6 | `P ← ProposeSteer(R_k)` | `steering.rounds.Round.propose` → `steering.propose.propose_steer` over the mixed-outcome groups of `steering.reservoir` |
| 7 | `E ← Replay(P, H_{k-1}, R_{0:k})` | `Round.replay` → `steering.replay.replay_evidence` (each trigger's failure rate per training step) |
| 8 | `P ← Admit(P, H_{k-1}, E)` | `Round.admit` → `steering.admit.judge`, `admit.arbitrate` |
| 9 | `H_k ← Gate(pi_k, H_{k-1}, P, D_hold)` | `Round.gate` → `steering.gate.measure` (one A/B), `gate.decide` (strictly above), `gate.accept` → `install.install` |

`steering.supervisor.supervise` runs a round every `--round-every` training steps, beside the
trainer, so the trainer never waits.

`program.json` is the only interface between the stages. Stage 1 writes its `compression` field
and stage 2 appends to `triggers`. Serving re-reads the file at the start of every episode, so
a trigger installed mid-training takes effect from the next episode on. `scripts/run_pipeline.sh`
chains the stages and places stage 1's rollouts at `rollouts/step_0.jsonl` as R_0.

## Package layout

| package | role | docs |
|---|---|---|
| `harness_programs.serving` | builds the served context from the state and `program.json`: sandbox, compression contract, budgeted window, triggers, guidance tail | this page |
| `harness_programs.domains` | per-environment tools, state DSL, task loading, reward; `toy` is offline | `domains.md` |
| `harness_programs.compression` | stage 1 | `compression.md` |
| `harness_programs.steering` | stage 2 optimizer: reservoir, propose (with program admission), replay, admit, gate, install | `steering.md` |
| `harness_programs.rl` | stage 2 trainer bridge: an agent loop for a GRPO trainer that serves through `harness_programs.serving` | `rl.md` |
| `harness_programs.rollout` | plain multi-turn rollouts against an OpenAI-compatible endpoint (stage-1 rollout and gate, stage-2 gate, eval) | — |
| `harness_programs.llm` | OpenAI-compatible chat client and an offline mock | — |
| `harness_programs.splits` | carves D_hold out of the training split and refuses any D_hold that overlaps D_train or an evaluation set | below |
| `harness_programs.eval` | scores a checkpoint on the evaluation tasks: score in [0, 1], reward, turns, eviction rate, trigger firings | — |

## D_hold: held-out training tasks

Algorithm 1 uses two task sets. The policy trains on D_train, and both Gates (lines 3 and 9)
score on D_hold. D_hold is held out **from the training split**: it is disjoint from D_train,
so a program is not rewarded for helping the policy recite tasks it trains on, and disjoint from
every evaluation set, so the reported numbers are never a selection set. The validation file is
never D_hold, and nothing defaults to it.

```bash
python -m harness_programs.splits carve --domain sql --tasks data/sql/train.jsonl --n 128 \
    --eval-tasks data/sql/val.jsonl
# -> data/sql/train_policy.jsonl   D_train: the policy trains on it (TRAIN_TASKS, --tasks)
#    data/sql/train_holdout.jsonl  D_hold: both gates score on it (HOLDOUT_TASKS, --holdout)
```

The carve draws whole tasks by identity (each domain's `task_key`, which compares content,
because `task_id`s are reassigned per file). `splits.check` runs before either stage starts:
the compression and steering CLIs and `scripts/run_pipeline.sh` all refuse a D_hold that *is* an
evaluation file, or that shares any task with `--tasks` or `--eval-tasks`.

## What this release leaves out

* **Only free-function programs.** A compression is a `compress(state)` function and a trigger
  a `check(ep)` predicate; no other program form is included.
* **No datasets, checkpoints or programs from the paper's runs.** `docs/domains.md` gives the
  task-file schema for each domain.

## Serving, in one paragraph

`ContextPipeline` (`serving/serve.py`) is what both stages and the trainer call. It loads
`program.json` (reloading it whenever the file changes), checks that the compression was written
against the current state contract, and builds a fresh `BudgetedContext` per episode. Each turn,
the context runs the compression once over the full transcript and then fits the largest window of
recent turns that stays within the budget, always keeping the pinned head (system prompt and
task). Firing triggers are rendered into one ephemeral user message,
`Guidance to follow for your next step: …`, placed after the window. Its tokens are reserved out
of the budget, so guidance can evict old turns but never truncates itself. Per-turn stats
(`raw_tokens`, `tokens_before`, `evicted`, `turns_kept`, `turns_total`, `overflow`, `fired`)
are logged with every rollout.

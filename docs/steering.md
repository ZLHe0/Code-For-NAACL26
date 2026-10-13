# Stage 2 — steering optimization during RL

Stage 2 evolves the **steering harness** with the policy. The steering harness is the
`triggers` list of the served program document `program.json`; stage 2 holds the stage-1
`compression` fixed and only ever adds triggers. A trigger (`harness_programs/serving/steer.py`) has:

* `check(ep)`: a small program over the episode so far. Serving runs it before every turn, and
  a check that raises evaluates as false;
* `guidance`: the text appended to the prompt when the check fires. An optional `render(ep)`
  writes it from the prefix instead;
* `fault(ep)` (optional): runs offline on a complete episode and may read `ep.gold`. It says
  whether the failure the trigger targets occurred. Without one, the check run on the complete
  episode stands in.

Serving re-reads `program.json` when it changes, so an installed trigger is live from the next
episode.

The code follows Algorithm 1, lines 4-10. The trainer is UpdatePolicy. Every `--round-every`
steps the supervisor runs a round at step k, and `Round.run()` (`harness_programs/steering/rounds.py`)
calls the remaining operators in order:

```python
proposals = self.propose()                   # P   <- ProposeSteer(R_k)
evidence = self.replay(proposals)            # E   <- Replay(P, H_{k-1}, R_{0:k})
admitted = self.admit(proposals, evidence)   # P   <- Admit(P, H_{k-1}, E)
verdict, installed = self.gate(admitted)     # H_k <- Gate(pi_k, H_{k-1}, P, D_hold)
```

## UpdatePolicy — `harness_programs.rl`, `supervisor.supervise`

GRPO on the training tasks under the incumbent harness H_{k-1} is the trainer (`docs/rl.md`).
It serves `program.json` and dumps each step's rollouts R_k to
`<run-dir>/rollouts/step_<N>.jsonl`. Together these files are the reservoir R_{0:k}; when
present, `step_0.jsonl` holds the stage-1 rollouts R_0.

`supervisor.supervise` runs beside the trainer. Whenever the newest dump reaches a multiple of
`--round-every` that has no completed round, it runs (or resumes) that round. A round that
raises is logged and retried on the next poll. The supervisor stops after the round at
`--total-steps`.

## ProposeSteer — `propose.propose_steer`

`reservoir.load_records` reads R_k: the dumps after the previous completed round, up to step k.
`reservoir.build_groups` keeps only the **mixed-outcome** groups, meaning rollouts of one task
at one step whose rewards differ. `reservoir.sample_groups` then draws up to `--max-groups` of
them. The creator gets one call per group (`propose.propose`), which shows the group's lower-
and higher-reward episodes. It replies with one failure hypothesis:

```json
{"insight": "the corrective insight",
 "name": "short_snake_case",
 "check": "def check(ep): ...",
 "fault": "def fault(ep): ...",
 "guidance": "If <condition>, <instruction>.",
 "render": "def render(ep): ..."}
```

The guidance is written from the insight. `fault` and `render` are optional. The admitted
hypotheses' triggers form the proposal set P.

**Program admission** runs inside ProposeSteer, while the exchange with the creator is still
open (`propose.screened`):

* **Static screen** (`screen.static_screen`). Each program needs its entry point and must pass
  the sandbox policy: allowed imports, no forbidden names, no unsafe constructs. It must also
  follow the read rules:
  * the check reads the episode, and does not read `final_output`/`status` (always empty in
    flight), the reward, or the gold view;
  * the fault consults the oracle only through `ep.gold`, and reads what the agent did;
  * the guidance is 40–300 characters, states a condition, leads a clause with an
    instruction, and quotes nothing from the gold view.
* **Compile and smoke** (`propose.smoke`). Each program is compiled in the restricted scope and
  executed once on sample episodes.

Complaints are shown to the creator for up to `HARNESS_STEERING_MAX_REVISIONS` (default 2)
repairs. A hypothesis that is still refused after that is dropped. `screen.dedup` drops a
proposal whose check is byte-identical to an installed, held or earlier one. A repeated name is
suffixed `__<group>`.

## Replay — `replay.replay_evidence`

Replay executes three kinds of trigger over **all** stored rollouts R_{0:k} (every
`step_<N>.jsonl` up to k, including `step_0.jsonl`):

* every candidate in P;
* every trigger held under MONITOR in the previous round;
* every installed trigger, whose fault is kept in the install ledger.

The check runs on each turn's prefix, exactly where serving would run it. The fault runs once on
the complete episode. For each training step the trigger's series records
(`replay.StepPoint`):

* `episodes`: the episodes replayed;
* `faulted`: the episodes in which the failure occurred, so `fault_rate = faulted / episodes`;
* `turns` and `fires`: the fire rate, recorded as secondary information.

Replay fails closed: a program that raises is dropped whole. The behaviour screens
(`screen.behaviour_screen`) drop a degenerate candidate or held trigger: one that never fires,
fires on every turn, or whose failure never occurred. Installed triggers are only recorded. The
reasons go into `replay.json` under `dropped`.

Each trigger's series is the curve of its failure over training, falling, holding steady or
rising. The series of all triggers (`replay.Evidence`) together make up the replay evidence E.
Replay ends there: it reads the curves as they are and does not extrapolate them.

## Admit — `admit.judge`, `admit.arbitrate`

For every candidate with evidence, `admit.report` builds the judge's report. It contains the
trigger's guidance and its per-step fault-rate table, read from E and never recomputed, along with the installed triggers and their current rates. The judge replies
`DECISION: INTERVENE | MONITOR | CONTINUE` with a rationale:

| verdict | when | effect (`admit.arbitrate`) |
|---|---|---|
| INTERVENE | the failure holds steady or rises | trigger and guidance go to the gate |
| MONITOR | the failure looks real but is not settled | trigger held: replayed and judged again next round, guidance not shown |
| CONTINUE | the failure keeps falling | dropped; left for the policy to learn |

A reply with no parseable decision is unresolved and is held like MONITOR. `--admit-all` is the
ablation that skips the judge and admits every replayed candidate.

## Gate — `gate.measure`, `gate.decide`, `gate.accept`

Each round makes one comparison under the current policy π_k, served by the policy endpoint.
Both arms play the same D_hold, the held-out training tasks (`--holdout`, optionally limited
by `--holdout-tasks`). D_hold is carved out of the training split, never the validation file,
and `config_from` refuses one that shares a task with `--tasks` or `--eval-tasks`:

* **arm A** serves the incumbent H_{k-1};
* **arm B** serves H_{k-1} plus **all** admitted triggers together, because guidance interacts.

`gate.decide` accepts only if arm B's mean reward is strictly above arm A's, so a tie rejects.
On acceptance, `gate.accept` calls `install.install`, which:

1. preflights the whole new document with the serving loader;
2. writes it atomically as H_k;
3. appends a row to the ledger `<program>.installs.jsonl`.

An installed name is never replaced, and nothing removes a trigger. If nothing is admitted, the
gate is skipped and H_k = H_{k-1}.

## Round artifacts

Each round writes to `<run-dir>/steering/round_<step>/`, atomically:

| file | operator | contents |
|---|---|---|
| `proposals.json` | ProposeSteer | the R_k files read, the sampled groups, the hypotheses, the dropped duplicates |
| `replay.json` | Replay | `end_step`, the R_{0:k} files read, the evidence E per trigger, the dropped triggers and reasons |
| `admit.json` | Admit | each report, verdict and rationale, plus `admitted`, `held`, `dropped` and `unresolved` |
| `gate.json` | Gate | both arms' scores, the triggers gated, the held-out task count, accepted or not |
| `gate/{incumbent,candidate}.{program.json,rollouts.jsonl}` | Gate | each arm's program and held-out rollouts |
| `round.json` | — | the round summary, written last; it marks the round complete |

A stage whose artifact exists is loaded rather than recomputed, so a stopped round resumes
without repeating a model call.

## CLI

```bash
python -m harness_programs.steering supervise --run-dir runs/rl --program runs/rl/program.json \
    --domain sql --total-steps 200 --tasks data/sql/train_policy.jsonl \
    --holdout data/sql/train_holdout.jsonl --eval-tasks data/sql/val.jsonl \
    --policy-base-url "$POLICY_URL" --policy-model latest
python -m harness_programs.steering round --run-dir runs/rl --program runs/rl/program.json \
    --domain sql --total-steps 200 --tasks data/sql/train_policy.jsonl \
    --holdout data/sql/train_holdout.jsonl --step 40                          # one round, or resume
```

The creator LLM (proposer and judge) is set with `--llm-base-url`/`--llm-model` or with
`HARNESS_LLM_{BASE_URL,MODEL,API_KEY}`. The policy endpoint (`--policy-*` or
`HARNESS_POLICY_*`) must serve the run's latest checkpoint π_k. `scripts/run_steering_rl.sh`
starts the trainer with the supervisor beside it.

| flag | default | meaning |
|---|---|---|
| `--total-steps` | required | last training step; the supervisor stops after the round there |
| `--tasks` | required | D_train, the training tasks the policy trains on; D_hold is checked against it |
| `--holdout` / `--holdout-tasks` | required / all | D_hold for the gate (held-out training tasks), and an optional limit on it |
| `--eval-tasks` | | evaluation task files; D_hold is checked to be disjoint from them as well |
| `--round-every` | 10 | training steps between rounds (`supervise`) |
| `--max-groups` | 20 | mixed-outcome groups the creator reads per round |
| `--admit-all` | off | ablation: admit every replayed candidate without the judge |
| `--history-budget`, `--serving-limit` | none / 1e6 | context sizing for gate rollouts; match training |
| `--workers`, `--seed` | 4 / none | parallel creator calls and gate rollouts; group sampling seed |
| `--max-rounds`, `--poll-seconds` | none / 60 | supervisor stopping and polling |
| `--step` | newest dump | the round to run or resume (`round`) |

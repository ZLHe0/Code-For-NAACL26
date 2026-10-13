# Stage 1 — compression optimization

Stage 1 learns the **compression program** H0: a deterministic Python function
`compress(state)` that re-renders earlier turns so the evidence a task needs survives the
context limit. It runs once, against the frozen base policy π0, before RL starts, and its
output is the `compression` field of `program.json`, which stage 2 holds fixed.

The code follows Algorithm 1, lines 1-3. `Campaign.run()` (`harness_programs/compression/campaign.py`)
calls the three operators in order:

```python
reservoir = self.rollout()            # R0 <- Rollout(pi_0, D_train)
proposals = self.propose(reservoir)   # P  <- ProposeCompress(R0)
return self.gate(proposals)           # H0 <- Gate(pi_0, {}, P, D_hold)
```

## Rollout — `rollout.run`

The base policy is rolled out on the training tasks under the append-only context, with no
compression program. Each task is rolled out `group_size` times (default 4) through the same
`ContextPipeline`, agent loop and context limit that training uses. Every episode keeps its
reward, and `episodes.by_task` groups them by task. Together they form the reservoir R0.
Rollouts collected elsewhere, in the same `EpisodeRecord` JSONL format, can be passed with
`--episodes` instead.

## ProposeCompress — `propose.run`

**Draft** (`propose.draft`, prompt from `evidence.creator_prompt`). The creator LLM's prompt
leads with the contrast. For several mixed-outcome tasks of R0, it shows one successful and one
failed attempt as compact turn digests (`evidence.contrast_block`). The creator identifies the
evidence the success used and the failure lost, then writes several candidate `compress(state)`
programs.

Supporting material follows so that the programs are written against the real format:
- observation sizes;
- what the context limit costs;
- the largest observations, verbatim.

The prompts never name a domain strategy. Domains contribute only optional structural hooks,
listed in `docs/domains.md`.

**Refine** (`refine.refine_one`). Each candidate goes through a loop on stored episodes of R0:

1. *Repair until it runs* (`refine.until_runs`). The candidate is compiled and applied to the
   episodes. A violation of the serving contract is sent back to the creator as an error to
   repair. Violations are:
   - it does not compile;
   - it raises;
   - it changes or drops an action;
   - it edits the live turn.
2. *Measure* (`measure.measure`). The creator is shown sampled contexts before and after
   compression (`measure.render_samples`), together with two measurements:
   - **unfaithful**: does the program remove information the policy needs later? Up to four
     decision points are taken per episode.
     - The needed items at a decision point are the domain's gold strings plus the words of
       every action the agent takes after it.
     - Only items readable in the append-only context are counted, excluding the pinned head,
       the actions and the live turn, which the contract already keeps.
     - The measurement is the share of counted items no longer readable in the compressed
       context. It is shown with examples of lost items.
   - **inert**: does the program leave the context largely unchanged? Only episodes whose
     final context binds the budget or fills at least half of it are considered. The
     measurement is the share of those episodes on which the served context is identical to
     the uncompressed one, or the program saves less than 5% of the tokens. It is shown with
     the mean token saving.
3. *Accept or revise*. The creator replies `accept`, or `revise` with new code, which goes back
   to step 1. A candidate not accepted within `--rounds` rounds, or that never runs, is
   dropped.

The accepted programs form the proposal set P.

## Gate — `gate.run`

Each program in P gets one trial on D_hold, the held-out training tasks: `gate.trial` runs it under the frozen
base policy with the same serving pipeline, tools and context limit as training. The
incumbent, no compression (`naive_baseline`), is trialled the same way. A trial's score is the
mean reward over D_hold.

`gate.select` makes H0 the best-scoring program if it scores strictly above no compression.
Otherwise H0 is no compression. `emit.run` writes H0 to `program.json`, which gets
`"compression": null` when no compression wins. Triggers already in the document are kept.

## Artifacts and resuming

| file | written by | content |
|---|---|---|
| `rollout/episodes.jsonl`, `rollout.json` | Rollout | R0 and a summary: tasks, mixed-outcome tasks, mean reward |
| `propose.json` | ProposeCompress | the creator's findings, every candidate with its rounds and measurements, and P |
| `gate/<arm>/program.json`, `gate/<arm>/episodes.jsonl` | Gate | each arm's program and held-out episodes |
| `gate.json` | Gate | every score and the winner |
| `program.json` | Gate | H0 |

Each phase is skipped on a re-run when its artifact exists. Delete an artifact to re-run that
phase and everything after it.

## Running it

```
python -m harness_programs.compression --domain codesearch --out runs/stage1 \
    --tasks data/codesearch/train_policy.jsonl --holdout data/codesearch/train_holdout.jsonl \
    --eval-tasks data/codesearch/val.jsonl \
    --serving-limit 32768 --history-budget 8192 --tokenizer "$TOKENIZER"
```

| flag | default | meaning |
|---|---|---|
| `--tasks` | required | D_train: the training tasks the policy trains on, rolled out for R0 and checked against D_hold |
| `--episodes` | | pre-collected rollouts to use as R0 instead of rolling out |
| `--holdout` | required | D_hold, for the gate: held-out training tasks, carved out of the training split (`python -m harness_programs.splits carve`) |
| `--eval-tasks` | | evaluation task files; D_hold is checked to be disjoint from them as well |
| `--serving-limit`, `--history-budget`, `--tokenizer` | | the context limit training serves under |
| `--rollout-tasks`, `--group-size` | 64, 4 | training tasks rolled out, and rollouts per task |
| `--candidates`, `--rounds` | 3, 3 | programs drafted, and accept-or-revise rounds per program |
| `--holdout-tasks` | 64 | D_hold tasks per trial |
| `--workers` | 8 | parallel episodes |

The creator LLM and the policy are any two OpenAI-compatible endpoints. They are configured
with the `--llm-*` / `--policy-*` flags, or with `HARNESS_LLM_{BASE_URL,MODEL,API_KEY}` and
`HARNESS_POLICY_{BASE_URL,MODEL,API_KEY}`. `scripts/run_compression.sh` wraps the call.
`examples/toy_pipeline.py` runs the whole stage offline on the toy domain, with a scripted
creator and policy.

## Writing a compression program

A program is a function `compress(state)` over the domain's state DSL (see `docs/domains.md`).
It returns the state with some turns re-rendered. The contract:

- a turn it omits is served unchanged;
- serving rejects it if it changes the number of messages, rewrites the pinned head, or alters
  an action;
- it must leave the live turn (the newest observation) alone;
- it is screened before it runs:
  - imports are limited to `re`, `json`, `math`, `itertools` and `collections`;
  - there are no I/O or reflection builtins and no dunder access;
  - each call has a time limit.

  The screen guards against accidental misuse by a code-writing model. It is not a security
  boundary.

`program.json` records the hash of the state contract the program was written against. Serving
refuses a program whose hash no longer matches.

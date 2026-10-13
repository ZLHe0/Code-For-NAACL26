#!/usr/bin/env bash
# Algorithm 1 end to end: rows -> stage 1 (compression) -> stage 2 (RL + steering) -> evaluation.
#
#   DOMAIN=codesearch MODEL_PATH=<base model> WORK=<work dir> \
#   TRAIN_TASKS=<train.jsonl> VAL_TASKS=<val.jsonl> SERVING_LIMIT=4096 HISTORY_BUDGET=1024 \
#     scripts/run_pipeline.sh
#
# Three disjoint task sets. TRAIN_TASKS is the full training split. First, HOLDOUT_N of its
# tasks are carved out as D_hold, the held-out training tasks both gates score on
# (<train>_holdout.jsonl); the policy trains on the rest (<train>_policy.jsonl). VAL_TASKS is the
# evaluation split: only the final eval reads it, and it is never D_hold.
#
# Stage 1 writes $WORK/stage1/program.json (H_0) and its base-policy rollouts become R_0, the
# first file of the stage-2 reservoir ($WORK/stage2/rollouts/step_0.jsonl).
#
# Endpoints, configured once in the environment:
#   HARNESS_LLM_{BASE_URL,MODEL,API_KEY}   creator LLM, used by both stages
#   HARNESS_POLICY_{BASE_URL,MODEL}        stage 1: the base policy pi_0; stage 2: the latest
#                                            checkpoint pi_k, for the gate (scripts/serve_policy.sh)
# Optional:
#   HOLDOUT_N        training tasks carved out as D_hold (default 128)
#   HOLDOUT_TASKS    an already-carved D_hold; TRAIN_TASKS must then be the tasks the policy
#                    trains on, without it. Checked against TRAIN_TASKS and VAL_TASKS either way
#   COMPRESSION=0    skip stage 1 and train with no compression program (the "+ Steering" arm)
#   STEERING=0       train under H_0 only (the "+ Compression" arm)
# Each stage is resumable; re-running skips work whose artifacts exist.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DOMAIN:?}" "${MODEL_PATH:?}" "${WORK:?}" "${TRAIN_TASKS:?}" "${VAL_TASKS:?}" "${SERVING_LIMIT:?}"
export DATA_DIR="$WORK/data" OUT="$WORK/stage1" RUN_DIR="$WORK/stage2"

echo "== split: D_hold is carved out of the training tasks, never taken from VAL_TASKS"
if [[ -z "${HOLDOUT_TASKS:-}" ]]; then
  stem="${TRAIN_TASKS%.jsonl}"
  [[ -f "${stem}_holdout.jsonl" ]] || python -m harness_programs.splits carve --domain "$DOMAIN" \
    --tasks "$TRAIN_TASKS" --n "${HOLDOUT_N:-128}" --eval-tasks "$VAL_TASKS"
  TRAIN_TASKS="${stem}_policy.jsonl" HOLDOUT_TASKS="${stem}_holdout.jsonl"
fi
if [[ "$(readlink -f "$HOLDOUT_TASKS")" == "$(readlink -f "$VAL_TASKS")" ]]; then
  echo "refusing: HOLDOUT_TASKS is VAL_TASKS; D_hold must be held-out training tasks" >&2
  exit 2
fi
python -m harness_programs.splits check --domain "$DOMAIN" --tasks "$TRAIN_TASKS" --holdout "$HOLDOUT_TASKS" \
  --eval-tasks "$VAL_TASKS"
export DOMAIN MODEL_PATH TRAIN_TASKS VAL_TASKS HOLDOUT_TASKS SERVING_LIMIT

echo "== rows"
[[ -f "$DATA_DIR/$DOMAIN/train.parquet" ]] || scripts/prepare_data.sh

echo "== stage 1"
if [[ "${COMPRESSION:-1}" == 0 ]]; then
  mkdir -p "$OUT"
  [[ -f "$OUT/program.json" ]] || printf '{"harness": "%s", "compression": null, "triggers": []}\n' \
    "$DOMAIN" >"$OUT/program.json"
else
  [[ -f "$OUT/program.json" ]] || scripts/run_compression.sh
  mkdir -p "$RUN_DIR/rollouts"
  if [[ ! -f "$RUN_DIR/rollouts/step_0.jsonl" ]]; then
    python -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["episode_files"]))' \
      "$OUT/rollout.json" | xargs cat >"$RUN_DIR/rollouts/step_0.jsonl"
  fi
fi

echo "== stage 2"
PROGRAM="$OUT/program.json" scripts/run_steering_rl.sh "$@"

echo "== eval (serve the final checkpoint at HARNESS_POLICY_BASE_URL first)"
eargs=(--domain "$DOMAIN" --tasks "$VAL_TASKS" --program "$RUN_DIR/program.json"
       --serving-limit "$SERVING_LIMIT" --out "$WORK/eval")
[[ -n "${HISTORY_BUDGET:-}" ]] && eargs+=(--history-budget "$HISTORY_BUDGET")
python -m harness_programs.eval "${eargs[@]}"

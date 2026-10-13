#!/usr/bin/env bash
# Stage 2: GRPO under the stage-1 program, with the steering supervisor running beside it.
#
# The trainer serves $RUN_DIR/program.json and re-reads it every episode (UpdatePolicy). The
# supervisor watches $RUN_DIR/rollouts/step_<N>.jsonl, runs a steering round every ROUND_EVERY
# steps (propose -> replay -> admit -> gate) and installs accepted triggers into the same
# program.json. Replay reads every step file; put the stage-1 rollouts R_0 at
# $RUN_DIR/rollouts/step_0.jsonl to include them.
#
# Required:
#   DOMAIN MODEL_PATH RUN_DIR DATA_DIR       as in configs/grpo_<domain>.yaml (scripts/prepare_data.sh)
#   PROGRAM          stage-1 program.json (copied to $RUN_DIR/program.json if that does not exist)
#   TRAIN_TASKS      the training tasks the policy trains on (D_train), for the D_hold check
#   HOLDOUT_TASKS    held-out TRAINING tasks D_hold (the gate), carved out of the training split
#                    by `python -m harness_programs.splits carve`; never the validation file
#   HARNESS_LLM_{BASE_URL,MODEL,API_KEY}        creator LLM (proposer and judge)
#   HARNESS_POLICY_{BASE_URL,MODEL}             endpoint serving the LATEST checkpoint (the gate)
# Optional:
#   VAL_TASKS        evaluation tasks; D_hold is also checked to be disjoint from them
#   STEERING=0       train with the fixed stage-1 program only (no supervisor)
#   ROUND_EVERY      steps between steering rounds (default 10)
#   STEPS            training steps (default 200, as in the configs); the supervisor stops there
#   HOLDOUT_LIMIT    held-out tasks per gate arm (default: all of HOLDOUT_TASKS)
#   ADMIT_ALL=1      ablation: admit every replayed trigger without the judge
#   HISTORY_BUDGET   history tokens above the pinned head; passed to the trainer and the gate
#   SERVING_LIMIT    gate rollouts' context limit; match the trainer's served prompt width
# Extra arguments are Hydra overrides for the trainer.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DOMAIN:?}" "${MODEL_PATH:?}" "${RUN_DIR:?}" "${DATA_DIR:?}" "${PROGRAM:?}"
export MODEL_PATH RUN_DIR DATA_DIR
STEPS="${STEPS:-200}"

mkdir -p "$RUN_DIR"
[[ -f "$RUN_DIR/program.json" ]] || cp "$PROGRAM" "$RUN_DIR/program.json"

supervisor=""
if [[ "${STEERING:-1}" != 0 ]]; then
  : "${TRAIN_TASKS:?D_hold is checked against the training tasks in TRAIN_TASKS}"
  : "${HOLDOUT_TASKS:?the steering gate plays the held-out training tasks in HOLDOUT_TASKS}"
  sargs=(supervise --run-dir "$RUN_DIR" --program "$RUN_DIR/program.json" --domain "$DOMAIN"
         --total-steps "$STEPS" --tasks "$TRAIN_TASKS" --holdout "$HOLDOUT_TASKS"
         --round-every "${ROUND_EVERY:-10}")
  [[ -n "${VAL_TASKS:-}" ]] && sargs+=(--eval-tasks "$VAL_TASKS")
  [[ -n "${HOLDOUT_LIMIT:-}" ]] && sargs+=(--holdout-tasks "$HOLDOUT_LIMIT")
  [[ "${ADMIT_ALL:-0}" == 1 ]] && sargs+=(--admit-all)
  [[ -n "${HISTORY_BUDGET:-}" ]] && sargs+=(--history-budget "$HISTORY_BUDGET")
  [[ -n "${SERVING_LIMIT:-}" ]] && sargs+=(--serving-limit "$SERVING_LIMIT")
  python -m harness_programs.steering "${sargs[@]}" >"$RUN_DIR/steering.log" 2>&1 &
  supervisor=$!
  trap '[[ -n "$supervisor" ]] && kill "$supervisor" 2>/dev/null || true' EXIT
  echo "steering supervisor pid $supervisor, log $RUN_DIR/steering.log"
fi

targs=()
[[ -n "${HISTORY_BUDGET:-}" ]] && targs+=("harness_programs.context_budget.history_budget=$HISTORY_BUDGET")

python -m verl.trainer.main_ppo --config-path="$PWD/configs" --config-name="grpo_$DOMAIN" \
  trainer.total_training_steps="$STEPS" "${targs[@]}" "$@"

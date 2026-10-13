#!/usr/bin/env bash
# Stage 1: Rollout -> ProposeCompress -> Gate against the base policy; writes $OUT/program.json.
#
# Required:
#   DOMAIN          codesearch | sql | alfworld | toy
#   TRAIN_TASKS     training tasks D_train the policy trains on (the rollout)
#   HOLDOUT_TASKS   held-out TRAINING tasks D_hold (the gate), carved out of the training split
#                   by `python -m harness_programs.splits carve`; never the validation file
#   OUT             artifact directory (resumable: every phase writes $OUT/<phase>.json)
#   SERVING_LIMIT   the policy's context limit in tokens
#   HARNESS_LLM_{BASE_URL,MODEL,API_KEY}     creator LLM
#   HARNESS_POLICY_{BASE_URL,MODEL}          base policy (scripts/serve_policy.sh)
# Optional:
#   HISTORY_BUDGET  history tokens allowed above the pinned head (the limited-context regime)
#   TOKENIZER       HF tokenizer for exact token counts (default: ~4 chars per token)
#   VAL_TASKS       evaluation tasks; D_hold is also checked to be disjoint from them
# Extra arguments go to `python -m harness_programs.compression`
# (e.g. --group-size 8 --candidates 4 --holdout-tasks 128).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DOMAIN:?}" "${TRAIN_TASKS:?}" "${HOLDOUT_TASKS:?}" "${OUT:?}" "${SERVING_LIMIT:?}"

args=(--domain "$DOMAIN" --out "$OUT" --tasks "$TRAIN_TASKS" --holdout "$HOLDOUT_TASKS"
      --serving-limit "$SERVING_LIMIT")
[[ -n "${HISTORY_BUDGET:-}" ]] && args+=(--history-budget "$HISTORY_BUDGET")
[[ -n "${TOKENIZER:-}" ]] && args+=(--tokenizer "$TOKENIZER")
[[ -n "${VAL_TASKS:-}" ]] && args+=(--eval-tasks "$VAL_TASKS")

python -m harness_programs.compression "${args[@]}" "$@"

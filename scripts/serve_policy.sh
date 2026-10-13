#!/usr/bin/env bash
# Serve a policy checkpoint as an OpenAI-compatible endpoint with vLLM.
#
# Stage 1 needs the BASE policy pi_0 served (its rollouts and its gate). The stage-2 gate needs
# the run's LATEST checkpoint pi_k served; re-run this script on each new HF-format checkpoint.
#
#   MODEL_PATH=<model or checkpoint dir> scripts/serve_policy.sh
#   export HARNESS_POLICY_BASE_URL=http://127.0.0.1:8000/v1 HARNESS_POLICY_MODEL=policy
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH to a Hugging Face model id or a local checkpoint directory}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-policy}" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --tensor-parallel-size "${TP:-1}" \
  "$@"

#!/usr/bin/env bash
# Turn task files (JSONL, schema in docs/domains.md) into the parquet rows verl reads.
#
#   DOMAIN=codesearch TRAIN_TASKS=<train.jsonl> VAL_TASKS=<val.jsonl> DATA_DIR=<dir> scripts/prepare_data.sh
#   -> $DATA_DIR/$DOMAIN/train.parquet, $DATA_DIR/$DOMAIN/val.parquet  (as configs/grpo_*.yaml expect)
#
# TRAIN_TASKS is what the policy trains on: the training split with D_hold already carved out
# (the <train>_policy.jsonl from `python -m harness_programs.splits carve`). VAL_TASKS is evaluation only.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DOMAIN:?}" "${TRAIN_TASKS:?}" "${VAL_TASKS:?}" "${DATA_DIR:?}"

mkdir -p "$DATA_DIR/$DOMAIN"
python - "$DOMAIN" "$TRAIN_TASKS" "$VAL_TASKS" "$DATA_DIR/$DOMAIN" <<'PY'
import sys
import pandas as pd
from harness_programs.rl import make_rows

domain, train, val, out = sys.argv[1:]
for split, path in (("train", train), ("val", val)):
    rows = make_rows(domain, path, split=split)
    pd.DataFrame(rows).to_parquet(f"{out}/{split}.parquet")
    print(f"{domain}/{split}: {len(rows)} rows")
PY

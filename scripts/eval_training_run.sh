#!/usr/bin/env bash
# Evaluate one finished training run with the archived main-table protocol.
#
#   bash scripts/eval_training_run.sh <run-dir> <out-json>
#
# With no arguments it evaluates the default finished run, which is what a
# detached launch needs (remote.py --detached cannot pass arguments).
#
# The protocol (docs/TGRS_EXPERIMENTS.md §3.3) scores the whole 301x301 grid
# with a 64x64 patch slid at stride 64 on the clean test split. --config must
# match the training run or the model constructor gets different kwargs.
# --out-json is mandatory: evaluate.py defaults it to None and writes nothing.
set -u

RUN_DIR="${1:-/home/wf/radarNWP/runs/mocast_1794069242}"
OUT_JSON="${2:-/home/wf/radarNWPQML/artifacts/mocast_1794069242.eval.json}"

LIVE_ROOT="${RADAR_NWP_LIVE_ROOT:-/home/wf/radarNWP}"
PY="${QMLBENCH_PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
DATA_ROOT="${RADAR_NWP_DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"

cd "$LIVE_ROOT" || exit 1
"$PY" scripts/evaluate.py \
    --model mocast --config mocast \
    --checkpoint "$RUN_DIR/last.pt" \
    --data-root "$DATA_ROOT" --packed \
    --split test --patch-size 64 --slide-stride 64 \
    --out-json "$OUT_JSON"
rc=$?
echo "EVAL_EXIT=$rc"
exit "$rc"

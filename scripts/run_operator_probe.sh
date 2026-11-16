#!/usr/bin/env bash
# Run the radar -> rain operator probe on the remote host.
#
# Usage: bash scripts/run_operator_probe.sh [n_cases] [n_px] [issue_stride] [folds] [max_iter]
#
# The probe only reads the archive.  The JSON lands in
# tools/_evals/operator_probe.json and is pulled back with:
#
#   python tools/remote.py --download \
#       /home/wf/radarNWPQML/tools/_evals/operator_probe.json \
#       --out tools/_evals/operator_probe.json
set -euo pipefail

cd /home/wf/radarNWPQML

PY=/home/wf/.conda/envs/radar_nwp/bin/python
N_CASES="${1:-62}"
N_PX="${2:-500}"
STRIDE="${3:-4}"
FOLDS="${4:-5}"
MAX_ITER="${5:-80}"
OUT=/home/wf/radarNWPQML/tools/_evals/operator_probe.json

mkdir -p tools/_evals run_logs

echo "[operator_probe] start $(date -Is)  cases=${N_CASES} px=${N_PX} stride=${STRIDE} folds=${FOLDS} max_iter=${MAX_ITER}"
"$PY" tools/_operator_probe.py \
  --out "${OUT}" \
  --n-cases "${N_CASES}" \
  --n-px "${N_PX}" \
  --issue-stride "${STRIDE}" \
  --folds "${FOLDS}" \
  --max-iter "${MAX_ITER}"
echo "[operator_probe] done  $(date -Is)  -> ${OUT}"

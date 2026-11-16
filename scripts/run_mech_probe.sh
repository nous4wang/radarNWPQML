#!/usr/bin/env bash
# Run the read-only mechanism probe on the remote host.
#
# Usage: bash scripts/run_mech_probe.sh [n_cases]
#
# The probe only reads the archive; it never trains and never writes to
# the data tree.  The JSON lands in tools/_evals/mechanism_probe.json and
# is pulled back with:
#
#   python tools/remote.py --download \
#       /home/wf/radarNWPQML/tools/_evals/mechanism_probe.json \
#       --out tools/_evals/mechanism_probe.json
set -euo pipefail

cd /home/wf/radarNWPQML

PY=/home/wf/.conda/envs/radar_nwp/bin/python
N_CASES="${1:-8}"
OUT=/home/wf/radarNWPQML/tools/_evals/mechanism_probe.json

mkdir -p tools/_evals run_logs

echo "[mech_probe] start $(date -Is)  n_cases=${N_CASES}"
"$PY" tools/_mechanism_probe.py --out "${OUT}" --n-cases "${N_CASES}"
echo "[mech_probe] done  $(date -Is)  -> ${OUT}"

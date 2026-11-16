#!/usr/bin/env bash
# Read-only status of the full-model sweep. Safe to run while the queue is live.
#
#   bash /home/wf/radarNWPQML/scripts/full_sweep_status.sh
#   OUT_DIR=/path/to/other/sweep QUEUE_LOG=/path/to/queue.log bash ...

set -uo pipefail

OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/full_sweep_20261108}"
QUEUE_LOG="${QUEUE_LOG:-/home/wf/radarNWPQML/runs/full_sweep_20261108_queue.log}"
PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"

echo "=== full-model sweep status  $(date -Is) ==="
echo "out_dir   : ${OUT_DIR}"
echo "queue_log : ${QUEUE_LOG}"
echo

if [[ -f "${OUT_DIR}/sweep_config.txt" ]]; then
    echo "--- provenance ---"
    cat "${OUT_DIR}/sweep_config.txt"
    echo
else
    echo "!! no sweep_config.txt yet"
    echo
fi

n_models=$(wc -w < <(sed -n 's/^models=//p' "${OUT_DIR}/sweep_config.txt" 2>/dev/null) || echo 0)
echo "--- progress ---"
printf 'models configured : %s\n' "${n_models}"
printf 'train logs        : %s\n' "$(find "${OUT_DIR}/logs" -name 'train_*.log' 2>/dev/null | wc -l)"
printf 'eval logs         : %s\n' "$(find "${OUT_DIR}/logs" -name 'eval_*.log' 2>/dev/null | wc -l)"
printf 'checkpoints       : %s\n' "$(find "${OUT_DIR}/train" -name 'last.pt' 2>/dev/null | wc -l)"
printf 'eval reports      : %s\n' "$(find "${OUT_DIR}/evals" -name '*.json' 2>/dev/null | wc -l)"
printf 'status rows       : %s\n' "$(wc -l < "${OUT_DIR}/status.tsv" 2>/dev/null || echo 0)"
echo

echo "--- running jobs ---"
ps -eo pid,ppid,etime,cmd | grep -E 'train\.py|evaluate\.py|run_full_model_sweep' \
    | grep -v grep | head -20 || true
echo

echo "--- GPUs ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true
echo

echo "--- queue log tail ---"
if [[ -f "${QUEUE_LOG}" ]]; then
    tail -12 "${QUEUE_LOG}"
else
    echo "(no queue log)"
fi
echo

if [[ -f "${OUT_DIR}/summary.json" ]]; then
    echo "--- summary ---"
    "${PYTHON}" - <<'PY' "${OUT_DIR}/summary.json"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("generated       :", d["generated"])
print("expected/present:", d["n_models_expected"], "/", d["n_reports_present"])
print("batch_complete  :", d["batch_complete"])
print("caveat          :", d["caveat"])
rows = [r for r in d["rows"] if r["present"] and r.get("csi_2.0") is not None]
print("top 5 by CSI@2.0 (reference only, NOT a claim):")
for r in sorted(rows, key=lambda x: -x["csi_2.0"])[:5]:
    print(f"  {r['model']:<26} {r['csi_2.0']:.6f}")
PY
else
    echo "(no summary yet)"
fi

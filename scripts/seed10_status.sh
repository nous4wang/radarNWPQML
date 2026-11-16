#!/usr/bin/env bash
# Read-only progress report for a detached two-arm seed batch.
#
# Usage (server side, archived 0..9 batch):
#   bash /home/wf/radarNWPQML/scripts/seed10_status.sh
# Fresh-seed batch (or any other seed set):
#   OUT_DIR=... QUEUE_LOG=... SEEDS="10 11 ... 19" \
#       bash /home/wf/radarNWPQML/scripts/seed10_status.sh
# A thin wrapper scripts/newseed_status.sh sets those for seeds 10..19.
#
# Writes nothing. Safe to run while the queue is training.

set -uo pipefail

OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/seeds10}"
QUEUE_LOG="${QUEUE_LOG:-/home/wf/radarNWPQML/runs/seeds10_queue.log}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-qpe_0 base_concat}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/evals"
LOG_DIR="${OUT_DIR}/logs"

n_jobs=$(( $(wc -w <<< "${ARMS}") * $(wc -w <<< "${SEEDS}") ))

echo "=== two-arm seed batch status  $(date -Is) ==="
echo "out dir     : ${OUT_DIR}"
echo "arms        : ${ARMS}"
echo "seeds       : ${SEEDS}"

# The detached launcher runs a /tmp script whose cmdline does not contain the
# queue name, so infer the state from its workers and its terminal marker.
workers="$(ps -eo cmd | grep -E 'train_keep_epochs.py|scripts/evaluate.py' \
    | grep -v grep | wc -l)"
if [[ -f "${QUEUE_LOG}" ]] && grep -q "QUEUE_DONE" "${QUEUE_LOG}" 2>/dev/null; then
    queue_state="DONE"
elif [[ "${workers}" -gt 0 ]]; then
    queue_state="RUNNING (${workers} workers)"
elif [[ -f "${QUEUE_LOG}" ]]; then
    queue_state="no live workers (between stages, finishing, or stopped)"
else
    queue_state="NOT STARTED"
fi
echo "queue       : ${queue_state}"

n_train=0
n_eval=0
n_train_log=0
n_eval_log=0
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        job="${arm}_s${seed}"
        [[ -n "$(ls "${TRAIN_DIR}/${job}"/*/epoch_02.pt 2>/dev/null)" ]] \
            && n_train=$(( n_train + 1 ))
        [[ -f "${EVAL_DIR}/${job}_ep00.eval.json" ]] \
            && n_eval=$(( n_eval + 1 ))
        [[ -f "${LOG_DIR}/train_${job}.log" ]] && n_train_log=$(( n_train_log + 1 ))
        [[ -f "${LOG_DIR}/eval_${job}_ep00.log" ]] && n_eval_log=$(( n_eval_log + 1 ))
    done
done
echo "training    : ${n_train} / ${n_jobs} complete (logs ${n_train_log})"
echo "scoring ep00: ${n_eval} / ${n_jobs} reports (logs ${n_eval_log})"

echo -n "live procs  : "
ps -eo cmd | grep -E "train_keep_epochs.py|scripts/evaluate.py" \
    | grep -v grep | wc -l
echo "              (counts DataLoader worker processes too, ~7 per training)"
echo "GPU         :"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader | sed 's/^/  /'

if [[ -f "${OUT_DIR}/queue_config.txt" ]]; then
    echo "config      : ${OUT_DIR}/queue_config.txt"
    sed 's/^/  /' "${OUT_DIR}/queue_config.txt"
fi

if [[ -f "${OUT_DIR}/summary.json" ]]; then
    echo "summary     : ${OUT_DIR}/summary.json"
    /home/wf/.conda/envs/radar_nwp/bin/python -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); print('  expected seeds:',p['protocol']['expected_seeds']); [print('  %-12s %2d/%2d  ratio=%s' % (a,r['pass_count'],r['n_seeds'],r['peak_end_ratio'])) for a,r in p['arms'].items()]; print('  fisher p  :',p['fisher_one_sided_p']); print('  complete  :',p['batch_complete'])" "${OUT_DIR}/summary.json" 2>/dev/null || echo "  (summary not parseable yet)"
else
    echo "summary     : not written yet (appears after stage 3)"
fi

echo "--- queue log tail ---"
tail -n 18 "${QUEUE_LOG}" 2>/dev/null || echo "(no queue log)"

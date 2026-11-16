#!/usr/bin/env bash
# Full-model sweep: train and evaluate every registered upstream baseline.
#
# Run detached on the GPU server:
#   bash /home/wf/radarNWPQML/scripts/run_full_model_sweep.sh
#
# Scope
# -----
# The target list is discovered at run time from
# ``radar_nwp_bench.models.list_models()`` (25 registered models at the time
# of writing). QLag-FlowNet / QLFN is deliberately NOT included: it is not in
# the upstream registry, and docs/archive/QLFN_SPEC.md R14 (已撤销) formerly forbade training, evaluating or
# modifying the upstream checkout for this reference implementation. Adding it
# here requires an explicit decision that overturns R14 and the ARIS verdict.
#
# Provenance and isolation
# ------------------------
# Nothing is written into the upstream ``runs/`` tree. All weights, reports
# and logs go under OUT_DIR. The parameter set is UNIFORM across all 25 models
# (the two upstream sweep scripts use two different parameter sets, so mixing
# them would produce a table whose rows are not comparable with each other).
# Because this uniform set is not the exact native configuration of either
# upstream script, the resulting table is NOT_COMPARABLE with the authoritative
# MASTER.md tables: it is a diagnostic sweep, not a new result claim.
#
# The judge is the file count, not the exit code. A model whose evaluation
# report already parses as JSON is skipped, so an interrupted run resumes.
#
# Overrides: MODELS, EPOCHS, PARALLEL, OUT_DIR, REPO_ROOT, DATA_ROOT,
#            PYTHON, BATCH_SIZE, WORKERS, PATCH_SIZE, SLIDE_STRIDE, LR,
#            SEED, EVAL_MAX_BATCHES, MAX_STEPS, SUMMARISER.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/full_sweep_20261108}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/evals"
LOG_DIR="${OUT_DIR}/logs"
SUMMARISER="${SUMMARISER:-/home/wf/radarNWPQML/scripts/summarize_full_sweep.py}"

EPOCHS="${EPOCHS:-10}"
PARALLEL="${PARALLEL:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

# Model list comes from the live registry, never from a hard-coded copy.
MODELS="${MODELS:-$(
    cd "${REPO_ROOT}" && "${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from radar_nwp_bench.models import list_models
print(" ".join(list_models()))
PY
)}"

if [[ -z "${MODELS// /}" ]]; then
    echo "FATAL: model discovery returned nothing" >&2
    exit 2
fi

# Provenance snapshot: a reader must be able to reconstruct the exact sweep
# without trusting the launcher's command line or the first lines of the log.
{
    echo "models=${MODELS}"
    echo "epochs=${EPOCHS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "workers=${WORKERS}"
    echo "patch_size=${PATCH_SIZE}"
    echo "slide_stride=${SLIDE_STRIDE}"
    echo "lr=${LR}"
    echo "seed=${SEED}"
    echo "loss=standard (train.py default; NOT the balanced/alpha-csi P2 batch)"
    echo "eval_split=test"
    echo "eval_max_batches=${EVAL_MAX_BATCHES}"
    echo "max_steps=${MAX_STEPS} (0 = full epochs)"
    echo "parallel=${PARALLEL}"
    echo "repo_root=${REPO_ROOT}"
    echo "data_root=${DATA_ROOT}"
    echo "out_dir=${OUT_DIR}"
    echo "comparability=NOT_COMPARABLE with MASTER.md authoritative tables"
    echo "qlfn_scope=EXCLUDED per docs/archive/QLFN_SPEC.md R14 (已撤销)"
    echo "written=$(date -Is)"
} > "${OUT_DIR}/sweep_config.txt"

echo "=========================================================="
echo " full-model sweep (train + evaluate)"
echo " models     : ${MODELS}"
echo " count      : $(wc -w <<< "${MODELS}")"
echo " epochs     : ${EPOCHS}"
echo " parallel   : ${PARALLEL} (8 cards, round robin)"
echo " out        : ${OUT_DIR}"
echo " started    : $(date -Is)"
echo "=========================================================="

latest_checkpoint() {
    ls -dt "$1"/*/last.pt 2>/dev/null | head -1 || true
}

run_model() {
    local model="$1" gpu="$2"
    local rundir="${TRAIN_DIR}/${model}"
    local report="${EVAL_DIR}/${model}.json"
    local tlog="${LOG_DIR}/train_${model}.log"
    local elog="${LOG_DIR}/eval_${model}.log"
    local train_rc=99 eval_rc=99 ckpt=""

    mkdir -p "${rundir}"

    ckpt="$(latest_checkpoint "${rundir}")"
    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] train on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
        "${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
            --model "${model}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split train \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${WORKERS}" \
            --patch-size "${PATCH_SIZE}" \
            --slide-stride "${SLIDE_STRIDE}" \
            --lr "${LR}" \
            --log-every 200 \
            --max-steps "${MAX_STEPS}" \
            --seed "${SEED}" \
            --device cuda \
            --out-dir "${rundir}" \
            > "${tlog}" 2>&1
        train_rc=$?
        ckpt="$(latest_checkpoint "${rundir}")"
    else
        echo "  [${model}] already trained (${ckpt}), training skipped"
        train_rc=0
    fi

    if [[ -z "${ckpt}" ]]; then
        echo "  [${model}] NO CHECKPOINT (train_rc=${train_rc}), eval skipped" >&2
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${model}" "${train_rc}" "no-checkpoint" "-" "${report}" \
            >> "${OUT_DIR}/status.tsv"
        return
    fi

    echo "  [${model}] eval on GPU ${gpu} (ckpt=${ckpt})"
    CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    "${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
        --model "${model}" \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${WORKERS}" \
        --max-batches "${EVAL_MAX_BATCHES}" \
        --seed "${SEED}" \
        --device cuda \
        --out-json "${report}" \
        > "${elog}" 2>&1
    eval_rc=$?

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${model}" "${train_rc}" "${eval_rc}" "${ckpt}" "${report}" \
        >> "${OUT_DIR}/status.tsv"
    echo "  [${model}] done train_rc=${train_rc} eval_rc=${eval_rc} $(date -Is)"
}

# Phase 1: train + eval, PARALLEL jobs at a time, one card each.
: > "${OUT_DIR}/status.tsv"
slot=0
for model in ${MODELS}; do
    if [[ -f "${EVAL_DIR}/${model}.json" ]] && \
       "${PYTHON}" -c 'import json,sys; json.load(open(sys.argv[1]))' \
            "${EVAL_DIR}/${model}.json" >/dev/null 2>&1; then
        echo "  [${model}] report already valid, skipped"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${model}" "skip" "skip" "cached" "${EVAL_DIR}/${model}.json" \
            >> "${OUT_DIR}/status.tsv"
        slot=$(( slot + 1 ))
        continue
    fi
    gpu=$(( slot % 8 ))
    slot=$(( slot + 1 ))
    run_model "${model}" "${gpu}" &
    while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
done
wait
echo "[phase 1 done] $(date -Is)"

# Phase 2: summarise. Failures are recorded, never silently dropped.
"${PYTHON}" "${SUMMARISER}" \
    --evals-dir "${EVAL_DIR}" \
    --models ${MODELS} \
    --config "${OUT_DIR}/sweep_config.txt" \
    --out "${OUT_DIR}/summary.json" \
    --tsv "${OUT_DIR}/summary.tsv" \
    > "${LOG_DIR}/summarise.log" 2>&1 || true

echo "[phase 2 done] $(date -Is)"
echo "SWEEP_DONE exit=0 $(date -Is)"

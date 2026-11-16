#!/usr/bin/env bash
# Independent 10-seed baseline batch, run detached on the GPU server.
#
# Why two arms
# ------------
# docs/MASTER.md 11.9 P2 requires *both* qpe_0 and base_concat, each with
# >=10 independent training seeds, in the same batch. A single-arm 10-seed
# run cannot clear the gate, so the queue is 2 x 10 trainings.
#
# Why three epochs
# ----------------
# scripts/run_route_seeds_train.sh documents the measurement: there is no LR
# schedule, batch order is fixed by the seed, and a 3-epoch run agrees byte
# for byte with a 20-epoch run on epoch_00..02. P3 binds the pass rate to
# ep00, so three epochs are sufficient and roughly 3x cheaper.
#
# Provenance and isolation
# ------------------------
# Every hyper-parameter below is copied from the upstream official scripts
# scripts/run_route_seeds_train.sh (training) and scripts/run_route_p2_seeds.sh
# (evaluation). All output goes under OUT_DIR, which is outside the running
# benchmark's runs/ tree: no archived checkpoint or report is overwritten.
#
# The judge is the file count, not the exit code. A run that already has its
# three kept epochs is skipped; an eval report that parses as JSON is skipped.
# Re-running this script after an interruption resumes the missing jobs.
#
# Usage (server side, no arguments needed):
#   bash /home/wf/radarNWPQML/scripts/run_seed10_queue.sh
# Fresh seeds 10..19: use scripts/run_newseed_queue.sh, which sets SEEDS,
# OUT_DIR and SUMMARISER_ARGS so the batch stays isolated and is judged
# against the right expected-seed set.
# Overrides: ARMS, SEEDS, PARALLEL, EPOCHS, OUT_DIR, REPO_ROOT, DATA_ROOT,
# SUMMARISER, SUMMARISER_ARGS.

set -uo pipefail

PYTHON="${PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/seeds10}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/evals"
LOG_DIR="${OUT_DIR}/logs"
SUMMARISER="${SUMMARISER:-/home/wf/radarNWPQML/scripts/summarize_seed10.py}"
# Extra summariser flags for a non-default seed set, e.g.
#   SUMMARISER_ARGS="--expected-seeds 10 11 ... 19"
# Left unquoted at the call site on purpose: it is a list of arguments.
SUMMARISER_ARGS="${SUMMARISER_ARGS:-}"

ARMS="${ARMS:-qpe_0 base_concat}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-3}"
PARALLEL="${PARALLEL:-4}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-6}"
TIERS="${TIERS:-0.1 0.5 2.0 4.0}"
ALPHA_CSI="${ALPHA_CSI:-0.5}"
SCORE_EPOCHS="${SCORE_EPOCHS:-00}"

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

# Provenance snapshot. The seed set is part of the measurement: a reader
# must be able to tell a fresh-seed batch from a deterministic replay
# without trusting the launcher's command line or the log's first lines.
{
    echo "arms=${ARMS}"
    echo "seeds=${SEEDS}"
    echo "epochs=${EPOCHS}"
    echo "score_epochs=${SCORE_EPOCHS}"
    echo "summariser_args=${SUMMARISER_ARGS}"
    echo "out_dir=${OUT_DIR}"
    echo "written=$(date -Is)"
} > "${OUT_DIR}/queue_config.txt"

# arm -> registered model name (upstream scripts/run_route_seeds_train.sh)
model_for() {
    case "$1" in
        qpe_0)       echo "simvp_v2_qpe_guided" ;;
        base_concat) echo "simvp_v2_nwp_concat" ;;
        *)           echo "" ;;
    esac
}

echo "=========================================================="
echo " independent 10-seed baseline batch"
echo " arms       : ${ARMS}"
echo " seeds      : ${SEEDS}"
echo " epochs     : ${EPOCHS} (kept per-epoch weights; score ep00)"
echo " parallel   : ${PARALLEL}"
echo " out        : ${OUT_DIR}"
echo " started    : $(date -Is)"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Stage 1: train every (arm, seed), four at a time, one card each.
# ---------------------------------------------------------------------------
echo "[stage 1/3] training"
slot=0
for arm in ${ARMS}; do
    model="$(model_for "${arm}")"
    if [[ -z "${model}" ]]; then
        echo "  !! unknown arm '${arm}', skipped"
        continue
    fi
    for seed in ${SEEDS}; do
        job="${arm}_s${seed}"
        run_root="${TRAIN_DIR}/${job}"
        if [[ -n "$(ls "${run_root}"/*/epoch_02.pt 2>/dev/null)" ]]; then
            echo "  ${job}: already trained, skipped"
            slot=$(( slot + 1 ))
            continue
        fi
        gpu=$(( slot % 8 ))
        slot=$(( slot + 1 ))
        echo "  -> ${job} (${model}) on GPU ${gpu}"
        (
            cd "${REPO_ROOT}" || exit 1
            CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 \
            "${PYTHON}" scripts/train_keep_epochs.py \
                --model "${model}" \
                --data-root "${DATA_ROOT}" --packed \
                --split train \
                --epochs "${EPOCHS}" \
                --batch-size "${BATCH_SIZE}" \
                --num-workers "${WORKERS}" \
                --patch-size "${PATCH_SIZE}" \
                --slide-stride "${SLIDE_STRIDE}" \
                --device cuda \
                --seed "${seed}" \
                --out-dir "${run_root}" \
                --loss balanced \
                --alpha-csi "${ALPHA_CSI}" \
                --tier-thresholds ${TIERS} \
                --qpe-weight 0.0 \
                --model-kwargs "{}" \
                > "${LOG_DIR}/train_${job}.log" 2>&1
            echo "TRAIN_EXIT=$? ${job}" >> "${OUT_DIR}/train_status.txt"
        ) &
        while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
    done
done
wait
echo "[stage 1/3] training finished $(date -Is)"

missing_train=0
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        job="${arm}_s${seed}"
        n=0
        for ep in 00 01 02; do
            [[ -n "$(ls "${TRAIN_DIR}/${job}"/*/epoch_${ep}.pt 2>/dev/null)" ]] \
                && n=$(( n + 1 ))
        done
        echo "  ${job}: ${n} / 3 kept epochs"
        [[ "${n}" -eq 3 ]] || missing_train=$(( missing_train + 1 ))
    done
done

# ---------------------------------------------------------------------------
# Stage 2: score ep00 for every (arm, seed) with the whole-grid protocol.
# --out-json is mandatory (evaluate.py defaults it to None); --per-window-out
# keeps the per-window table that P4's two-level bootstrap needs.
# ---------------------------------------------------------------------------
echo "[stage 2/3] scoring ep00"
slot=0
for arm in ${ARMS}; do
    model="$(model_for "${arm}")"
    [[ -n "${model}" ]] || continue
    for seed in ${SEEDS}; do
        job="${arm}_s${seed}"
        run_dir="$(ls -dt "${TRAIN_DIR}/${job}"/*/ 2>/dev/null | head -1)"
        if [[ -z "${run_dir}" ]]; then
            echo "  ${job}: no run directory, cannot score"
            continue
        fi
        for ep in ${SCORE_EPOCHS}; do
            ck="${run_dir}epoch_${ep}.pt"
            [[ -f "${ck}" ]] || { echo "  ${job}: epoch_${ep}.pt missing"; continue; }
            out="${EVAL_DIR}/${job}_ep${ep}.eval.json"
            windows="${EVAL_DIR}/${job}_ep${ep}.windows.json"
            if [[ -f "${out}" ]] && "${PYTHON}" -c \
                "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
                echo "  ${job} ep${ep}: already scored, skipped"
                slot=$(( slot + 1 ))
                continue
            fi
            gpu=$(( slot % 8 ))
            slot=$(( slot + 1 ))
            echo "  -> scoring ${job} ep${ep} on GPU ${gpu}"
            (
                cd "${REPO_ROOT}" || exit 1
                CUDA_VISIBLE_DEVICES="${gpu}" \
                "${PYTHON}" scripts/evaluate.py \
                    --model "${model}" \
                    --checkpoint "${ck}" \
                    --data-root "${DATA_ROOT}" --packed --split test \
                    --batch-size "${BATCH_SIZE}" --num-workers "${WORKERS}" \
                    --patch-size "${PATCH_SIZE}" --slide-stride "${SLIDE_STRIDE}" \
                    --thresholds ${TIERS} \
                    --device cuda \
                    --out-json "${out}" \
                    --per-window-out "${windows}" \
                    > "${LOG_DIR}/eval_${job}_ep${ep}.log" 2>&1
                echo "EVAL_EXIT=$? ${job} ep${ep}" >> "${OUT_DIR}/eval_status.txt"
            ) &
            while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]]; do sleep 5; done
        done
    done
done
wait
echo "[stage 2/3] scoring finished $(date -Is)"

missing_eval=0
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        job="${arm}_s${seed}"
        out="${EVAL_DIR}/${job}_ep00.eval.json"
        ok=0
        [[ -f "${out}" ]] && "${PYTHON}" -c \
            "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null \
            && ok=1
        echo "  ${job} ep00 report: ${ok} / 1"
        [[ "${ok}" -eq 1 ]] || missing_eval=$(( missing_eval + 1 ))
    done
done

# ---------------------------------------------------------------------------
# Stage 3: diagnostic summary. This is a P2 anchor, not an improvement claim.
# ---------------------------------------------------------------------------
echo "[stage 3/3] summarising"
if [[ -f "${SUMMARISER}" ]]; then
    "${PYTHON}" "${SUMMARISER}" \
        --evals-dir "${EVAL_DIR}" \
        ${SUMMARISER_ARGS} \
        --out "${OUT_DIR}/summary.json" || true
    echo "  summary -> ${OUT_DIR}/summary.json"
else
    echo "  !! summariser missing: ${SUMMARISER}"
fi

total=$(( missing_train + missing_eval ))
echo "=========================================================="
echo " finished    : $(date -Is)"
echo " missing kept-epoch runs : ${missing_train}"
echo " missing ep00 reports    : ${missing_eval}"
echo " supports_improvement_claim = False (baseline arms only)"
echo "QUEUE_DONE exit=${total}"
echo "=========================================================="

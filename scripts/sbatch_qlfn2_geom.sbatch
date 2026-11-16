#!/bin/bash
#SBATCH --job-name=qlfn2-geom
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# Measure the evaluation geometry on the checkpoints that already exist.
#
# Why this job exists.
#
# The qlfn2 sweep scored 0.1743 at CSI@2.0, above the benchmark's oracle
# ceiling of 0.1653.  A deployable model cannot beat an oracle that is
# handed the answer, so either the number is wrong or the two were measured
# with different instruments.  They were:
#
#   the sweep        patch 128 / stride 64   16 tiles   8608 samples
#   the oracle       patch  64 / stride 64   25 tiles  13450 samples
#
# The sweep's own protocol was confirmed to shift the score: the
# benchmark's physical_qpe epoch_00 scores 0.1226 under 64/64 and 0.1395
# under 128/128 -- the same weights, the same verifier, +13.8 %.
#
# This job closes the question without waiting for a retrain: it scores the
# checkpoints the sweep already produced under EVERY geometry, including the
# upstream authority one.  If qlfn2's 0.1743 falls to something below the
# oracle under 64/64, the anomaly is a measurement artefact and the sweep's
# absolute values are simply not comparable with the upstream numbers.  If it
# stays above, the anomaly is real and needs a different explanation.
#
# It is read-only: it loads checkpoints that already exist and writes
# reports under /tmp.

set -eo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-/tmp/geometry}"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_ROOT}/runs/qlfn2_sweep}"

# The three geometries in play, with the one the upstream authority uses
# first so its number is the one that can be compared with 0.1226 / 0.1653.
GEOMETRIES=("64 64" "128 64" "128 128")

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " geometry comparison  $(date -Is)"
echo " out : ${OUT_DIR}"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

for ARM_DIR in qlfn2 qlfn2_no_nwp qlfn2_no_motion physical_qpe; do
    CKPT="$(ls -1t "${SWEEP_ROOT}"/"${ARM_DIR}"/seed10/train/*/last.pt 2>/dev/null | head -1 || true)"
    [[ -f "${CKPT}" ]] || continue
    MODEL="${ARM_DIR}"

    # The arm's own flag has to travel to the scoring stage.  An arm trained
    # without the NWP but scored with it measures a different model than the
    # one that was trained, and an arm whose advection was switched off at
    # training time but switched back on at scoring time is scored with the
    # mechanism it was supposed to have removed.  Both would look like a
    # result.  The physical_qpe control is the benchmark's own model, so it
    # is scored through the benchmark's own evaluator.
    case "${ARM_DIR}" in
        qlfn2)         EVAL_CMD=("${REPO_ROOT}/scripts/eval_qlfn2.py"); EVAL_EXTRA=() ;;
        qlfn2_no_nwp)  EVAL_CMD=("${REPO_ROOT}/scripts/eval_qlfn2.py"); EVAL_EXTRA=(--no-nwp) ;;
        qlfn2_no_motion)
            EVAL_CMD=("${REPO_ROOT}/scripts/eval_qlfn2.py")
            EVAL_EXTRA=(--model-kwargs '{"motion_mode": "none"}')
            ;;
        *)             EVAL_CMD=("${BENCH_ROOT}/scripts/evaluate.py"); EVAL_EXTRA=() ;;
    esac

    echo "----------------------------------------------------------"
    echo " arm ${ARM_DIR}   checkpoint ${CKPT}"
    echo "   evaluator ${EVAL_CMD[*]}   flags ${EVAL_EXTRA[*]:-<none>}"
    echo "----------------------------------------------------------"
    for GEO in "${GEOMETRIES[@]}"; do
        PATCH="${GEO% *}"
        STRIDE="${GEO#* }"
        JSON="${OUT_DIR}/${ARM_DIR}_p${PATCH}_s${STRIDE}.json"
        echo "  geometry patch=${PATCH} stride=${STRIDE} -> ${JSON}"
        python -u "${EVAL_CMD[@]}" \
            --model "${MODEL}" \
            --checkpoint "${CKPT}" \
            --data-root "${DATA_ROOT}" \
            --packed \
            --split test \
            --patch-size "${PATCH}" \
            --slide-stride "${STRIDE}" \
            --batch-size 16 \
            --num-workers 4 \
            --thresholds 0.1 0.5 1 2 4 8 16 \
            --fss-radii 1 2 4 \
            --seed 10 \
            --device cuda \
            --out-json "${JSON}" \
            ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"} || echo "   evaluate rc=$?"
        if [[ -f "${JSON}" ]]; then
            python "${REPO_ROOT}/tools/check_eval_report.py" "${JSON}" || true
        else
            echo "   FATAL: no report at ${JSON}" >&2
        fi
        echo
    done
done

echo "=========================================================="
echo " geometry comparison finished $(date -Is)"
echo "=========================================================="

#!/bin/bash
#SBATCH --job-name=qlfn2-repro
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# Reproduce the benchmark's own published number, from the benchmark's own
# checkpoint, with the benchmark's own geometry.
#
# Why this job exists.
#
# The whole comparison this work depends on is against three upstream
# numbers: the persistence floor 0.0924, the oracle ceiling 0.1653, and
# physical_qpe's peak 0.1226.  All three are measured under patch 64 /
# stride 64.
#
# Before this work's own numbers can be read against them, the measuring
# apparatus has to be shown to be the same one.  The benchmark's own
# physical_qpe checkpoint is still on disk, with the epoch copies its
# epoch-curve experiment kept.  Scoring epoch_00 under 64/64 must return
# 0.1226.  If it does, the evaluator, the verifier, the stitching and the
# threshold units in this environment are demonstrably the benchmark's, and
# a difference between our numbers and the upstream ones is a difference in
# the models rather than in the instrument.
#
# If it does NOT return 0.1226, then the instrument is not the benchmark's,
# and every comparison this work makes is void until that is fixed.  Either
# outcome is informative, which is why it is worth a job.
#
# This is read-only: it loads checkpoints that already exist and writes
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

BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
CKPT_DIR="${CKPT_DIR:-/home/wf/radarNWP/runs/physical_qpe/seed0/physical_qpe_1793928770}"
OUT_DIR="${OUT_DIR:-/tmp/repro}"
# The upstream authority geometry.  The oracle ceiling and the persistence
# floor were measured under it, so it is the only geometry in which a
# comparison with them is a comparison.
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " reproduction  $(date -Is)"
echo "   checkpoints : ${CKPT_DIR}"
echo "   geometry    : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   expected    : epoch_00 -> CSI@2.0 = 0.1226 (upstream E19)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

for CKPT in "${CKPT_DIR}"/epoch_00.pt "${CKPT_DIR}"/epoch_01.pt \
            "${CKPT_DIR}"/epoch_02.pt "${CKPT_DIR}"/last.pt; do
    [[ -f "${CKPT}" ]] || continue
    NAME="$(basename "${CKPT}" .pt)"
    JSON="${OUT_DIR}/physical_qpe_${NAME}.json"

    echo "----------------------------------------------------------"
    echo " checkpoint ${NAME}"
    echo "----------------------------------------------------------"
    python -u "${BENCH_ROOT}/scripts/evaluate.py" \
        --model physical_qpe \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --packed \
        --split test \
        --patch-size "${PATCH_SIZE}" \
        --slide-stride "${SLIDE_STRIDE}" \
        --batch-size 16 \
        --num-workers 4 \
        --thresholds 0.1 0.5 1 2 4 8 16 \
        --fss-radii 1 2 4 \
        --device cuda \
        --out-json "${JSON}"
    eval_rc=$?
    echo "evaluate rc=${eval_rc}"

    if [[ -f "${JSON}" ]]; then
        python "${REPO_ROOT}/tools/check_eval_report.py" "${JSON}" || true
    else
        echo "FATAL: no report at ${JSON}" >&2
        exit 5
    fi
    echo
done

echo "=========================================================="
echo " reproduction check $(date -Is)"
echo "=========================================================="

# The verdict is printed by the summariser so that the comparison against
# the published value is made in one place, from the stored reports.
python "${REPO_ROOT}/tools/summarize_geometry.py" \
    --root "${OUT_DIR}" || true

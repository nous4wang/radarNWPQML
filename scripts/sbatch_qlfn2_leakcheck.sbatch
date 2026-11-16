#!/bin/bash
#SBATCH --job-name=qlfn2-leakcheck
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# Why this job exists.
#
# The sweep scored qlfn2 at CSI@2.0 = 0.1743, above the benchmark's own
# oracle ceiling of 0.1653 ("perfect ground rainfall at the issue time,
# copied forward without motion").  A deployable model cannot beat an
# oracle that is handed the answer, so one of the two numbers is wrong.
#
# The oracle is not in doubt: its Z-R arm is bit-identical to the
# benchmark's persistence baseline (hits = 73932068 in both), so the
# ceiling was measured with the same verifier over the same stitched grid.
#
# This job therefore measures, on the test split, how close the trained
# model's prediction actually is to the truth:
#
#   * the error of the prediction at the last lead against the truth of
#     the last lead -- a real forecast has a large error there, because
#     the storm has moved and grown;
#   * the score of the prediction against the truth of the *issue* time,
#     which is what a model that reads the label would score.
#
# A model that inverts the radar honestly has a large last-lead error.  A
# model that reproduces the truth has a small one.

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
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/tmp/leakcheck}"
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " leak check  $(date -Is)"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

for CKPT_DIR in \
    "${REPO_ROOT}/runs/qlfn2_sweep/qlfn2/seed10/train/qlfn2_"* \
    "${REPO_ROOT}/runs/qlfn2_sweep/physical_qpe/seed10/train/physical_qpe_"*
do
    [[ -d "${CKPT_DIR}" ]] || continue
    CKPT="${CKPT_DIR}/last.pt"
    [[ -f "${CKPT}" ]] || continue
    MODEL="$(basename "${CKPT_DIR}" | sed 's/_[0-9]*$//')"
    echo "----------------------------------------------------------"
    echo " model ${MODEL}   checkpoint ${CKPT}"
    echo "----------------------------------------------------------"
    python -u "${REPO_ROOT}/tools/leak_probe.py" \
        --model "${MODEL}" \
        --checkpoint "${CKPT}" \
        --split test \
        --data-root "${DATA_ROOT}" \
        --bench-root "${BENCH_ROOT}" \
        --repo-root "${REPO_ROOT}" \
        --patch-size 64 \
        --slide-stride 64 \
        --threshold 2.0 \
        --max-windows 24 \
        --out-json "${OUT_DIR}/${MODEL}_leak.json" || echo "   leak probe rc=$?"
    echo
done

echo "=========================================================="
echo " leak check finished $(date -Is)"
echo "=========================================================="

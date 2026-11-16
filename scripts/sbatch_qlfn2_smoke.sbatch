#!/bin/bash
#SBATCH --job-name=qlfn2-smoke
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# A smoke test of the training step at the batch size the sweep will use.
#
# The sweep runs sixteen 128x128 windows per batch, which is sixteen times
# eleven history frames of ten radar channels -- a large activation for a
# 24 GiB card.  Rather than discover that as an out-of-memory failure
# halfway through a six-hour job, this job runs a handful of steps at that
# batch size and reports the peak memory it reached.

set -eo pipefail

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${REPO_ROOT}/runs/qlfn2_smoke"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " qlfn2 smoke   $(date -Is)"
echo "   batch size : ${BATCH_SIZE}"
echo "   out        : ${OUT_DIR}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device         :", props.name)
    print(f"device memory  : {props.total_memory / 1024 ** 3:.1f} GiB")
PY

echo
echo "--- training steps at the sweep's batch size --------------------"
# The wrapper runs the benchmark's own training loop and then reports the
# allocator's peak from the same process.  A second python afterwards would
# start with an empty allocator and report a peak of zero, which is why the
# training is not run here directly.
python -u "${REPO_ROOT}/scripts/train_qlfn2_mem.py" \
    --model qlfn2 \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs 1 \
    --max-steps 5 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 2 \
    --patch-size 128 \
    --slide-stride 64 \
    --lr 2e-4 \
    --loss balanced \
    --qpe-weight 0.5 \
    --log-every 1 \
    --seed 10 \
    --device cuda \
    --out-dir "${OUT_DIR}/train"
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "=========================================================="
echo " smoke finished $(date -Is)   train rc=${train_rc}"
echo "=========================================================="

exit "${train_rc}"

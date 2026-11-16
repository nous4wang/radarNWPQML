#!/bin/bash
#SBATCH --job-name=qlfn2-peakcheck
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# Score the upstream benchmark's own physical_qpe checkpoints under the same
# protocol the sweep uses.
#
# Why this job exists: the sweep scored one checkpoint per run -- the last
# one -- and reported physical_qpe at CSI@2.0 = 0.0163, far below the
# persistence floor, while qlfn2 scored 0.1554.  The benchmark's own
# protocol keeps a copy of every epoch and reads the score off the peak,
# because for these models the last checkpoint is the collapsed end.  This
# job scores the benchmark's own physical_qpe checkpoints epoch by epoch, so
# the two numbers can be compared on the same footing: if epoch_00 lands
# near the published 0.1226 and last.pt lands near 0.0163, the sweep was
# reading the wrong checkpoint and the model comparison has to be redone on
# per-epoch scores.
#
# This is a read-only evaluation: it loads checkpoints that already exist
# and writes reports under /tmp.

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
CKPT_DIR="${CKPT_DIR:-/home/wf/radarNWP/runs/physical_qpe/seed0/physical_qpe_1793928770}"
OUT_DIR="${OUT_DIR:-/tmp/peakcheck}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " peak check  $(date -Is)"
echo " checkpoints : ${CKPT_DIR}"
echo " output      : ${OUT_DIR}"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

for CKPT in "${CKPT_DIR}"/epoch_*.pt "${CKPT_DIR}/last.pt"; do
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
echo " peak check finished $(date -Is)"
echo "=========================================================="

#!/bin/bash
#SBATCH --job-name=qlfn2-pilot
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# The qlfn2 pilot on the Slurm cluster.
#
# Why sbatch and not a background process: the GPU is invisible to a login
# session on this platform.  The device cgroup of the login slice refuses
# /dev/nvidiactl with EPERM, so ``cuInit`` returns 100 there no matter how
# healthy the driver is.  Only a job that Slurm has given a GRES allocation
# can open the device.  Rule 3 of the platform rules also forbids running
# heavy compute outside Slurm.
#
# What this job proves
# --------------------
# 1. the qlfn2 self-checks (13) still pass on the cluster's interpreter;
# 2. the **benchmark's own** ``scripts/train.py`` trains qlfn2 here, with the
#    benchmark's loss, split, interpolation policy and checkpoint layout;
# 3. the **benchmark's own** ``scripts/evaluate.py`` scores the resulting
#    checkpoint, with the same full-grid Hann stitching and the same
#    verifier.
#
# The point of the pilot is not a good score.  It is that the number the
# benchmark would report for qlfn2 comes out of the benchmark's own harness,
# with no special case for qlfn2 anywhere in it.
#
# The CUDA visibility is Slurm's: the device may be renumbered inside the
# job, so the script uses ``cuda:0`` and never touches the host indices.

set -eo pipefail

# The conda activation scripts reference variables that are not set, so
# ``set -u`` has to be off while conda is being activated.  With it on, the
# job dies in the activation script before python is ever reached:
#
#     libblas_mkl_activate.sh: line 1: MKL_INTERFACE_LAYER: unbound variable
#
# The nounset protection is restored once the environment is ready.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

cd "${SLURM_SUBMIT_DIR}"

# 8 logical CPUs are 4 physical cores with two threads each, so the compute
# threads are set to 4.  The dataloader workers are counted separately.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/qlfn2_pilot}"
# The patch size and the stride have to tile the 301x301 grid exactly: the
# benchmark's symmetric origins for a 128-pixel patch are (0, 58, 115, 173),
# which covers the field from the first pixel to the last with no gap.  A
# smaller patch leaves the far edge uncovered, and the upstream verifier
# then reports "0 samples" instead of a score -- a silent failure, which is
# why the pilot checks the sample count below.
PATCH_SIZE="${PATCH_SIZE:-128}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
# The dataset enumerates its items as ``sample_index * n_positions +
# position_index``, so sixteen consecutive items are exactly the sixteen
# tiles of one window.  A batch of sixteen therefore completes a window in
# a single batch, which is what the stitcher needs: a window is only
# scored once all of its tiles have been seen.
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-3}"
# eight batches of sixteen tiles = eight complete windows, enough for the
# verifier to produce a score without evaluating the whole test split
MAX_BATCHES="${MAX_BATCHES:-8}"

mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " qlfn2 pilot   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   host       : $(hostname)"
echo "   repo       : ${REPO_ROOT}"
echo "   data       : ${DATA_ROOT}"
echo "   out        : ${OUT_DIR}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
print("device count   :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name    :", torch.cuda.get_device_name(0))
    print("capability     :", torch.cuda.get_device_capability(0))
PY

echo
echo "--- 1/3  self-checks -------------------------------------------------"
python -m qlfn2.checks
checks_rc=$?
echo "self-checks rc=${checks_rc}"

echo
echo "--- 2/3  training through the benchmark's train.py -------------------"
python -u "${REPO_ROOT}/scripts/train_qlfn2.py" \
    --model qlfn2 \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs 1 \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --lr 2e-4 \
    --loss balanced \
    --qpe-weight 0.5 \
    --log-every 1 \
    --seed 0 \
    --device cuda \
    --out-dir "${OUT_DIR}/train"
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "--- 3/3  scoring through the benchmark's evaluate.py -----------------"
# The benchmark writes the checkpoint into a timestamped subdirectory of
# the output directory (``<out-dir>/<model>_<unix-time>/last.pt``), so the
# newest one is found rather than assumed.
CKPT="$(ls -1t "${OUT_DIR}"/train/*/last.pt 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
    echo "FATAL: no checkpoint under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint: ${CKPT}"
python -u "${REPO_ROOT}/scripts/eval_qlfn2.py" \
    --model qlfn2 \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split test \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --thresholds 0.5 1 2 4 \
    --fss-radii 1 2 \
    --seed 0 \
    --device cuda \
    --out-json "${OUT_DIR}/eval_test.json"
eval_rc=$?
echo "eval rc=${eval_rc}"

# The verifier reports "0 samples" when the sliding walk leaves a gap in the
# grid, and the score table is then all NaN.  That is a silent failure: the
# job completes, the JSON is written, and the file looks like a result.  The
# pilot therefore checks that the report actually carries a score, with the
# same checker the experiment script uses, so the two cannot drift apart.
REPORT="${OUT_DIR}/eval_test.json"
python "${REPO_ROOT}/tools/check_eval_report.py" "${REPORT}"
report_rc=$?
echo "report rc=${report_rc}"

echo
echo "=========================================================="
echo " pilot finished $(date -Is)"
echo "   self-checks rc = ${checks_rc}"
echo "   train rc       = ${train_rc}"
echo "   eval rc        = ${eval_rc}"
echo "   report rc      = ${report_rc}"
echo "   report         = ${REPORT}"
echo "=========================================================="

exit $(( checks_rc | train_rc | eval_rc | report_rc ))

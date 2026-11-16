#!/bin/bash
#SBATCH --job-name=qlfn2-exp
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
#SBATCH --error=run_logs/slurm/%x-%j.err

# One qlfn2 experiment: train one arm with one seed, then score it.
#
# The job is submitted as an array (see scripts/run_qlfn2_sweep.sh), so the
# array index selects the arm and the seed.  Every arm goes through the same
# two upstream entry points:
#
#     scripts/train_qlfn2.py  ->  the benchmark's scripts/train.py
#     scripts/eval_qlfn2.py   ->  the benchmark's scripts/evaluate.py
#
# and every arm is scored on the same split with the same thresholds and the
# same stitching.  The arms differ only in the flags, so a difference in the
# score is a difference in the mechanism.
#
# Arm table (the array index picks one row):
#
#   0  qlfn2                       motion=corr, CAPPI on, NWP on
#   1  qlfn2  --no-nwp             the NWP is withheld
#   2  qlfn2  motion=none          the advection is switched off
#   3  physical_qpe                the upstream control, same harness
#
# Arm 3 is the control that decides whether qlfn2 beats the benchmark's own
# single-frame QPE arm (CSI@2.0 = 0.1226 upstream) under an identical
# protocol.  It runs the benchmark's own scripts/train.py, not the wrapper,
# because physical_qpe is registered upstream.
#
# The GPU is only visible inside the job: the login slice's device cgroup
# refuses /dev/nvidiactl with EPERM, so ``cuInit`` returns 100 there.  The
# device may be renumbered, so the script uses ``cuda:0`` and never touches
# the host indices.

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qlfn2_sweep}"

ARM="${ARM:-0}"
# The array index picks the seed, so that one array job is one arm with
# several seeds.  The seed base is exported by the submit script, so the
# seeds can be moved away from the ones an earlier experiment used.
SEED_BASE="${SEED_BASE:-10}"
SEED="${SEED:-$((SEED_BASE + ${SLURM_ARRAY_TASK_ID:-0}))}"
# Ten epochs is the benchmark's own default, so the control arm is trained
# under the recipe the upstream configuration documents for it
# (``--loss balanced --qpe-single-frame``).  Every arm gets the same budget,
# which is what makes the difference between the arms a difference in the
# mechanism.
EPOCHS="${EPOCHS:-10}"
# A 128-pixel patch with a stride of 64 gives the benchmark's symmetric
# origins (0, 58, 115, 173) -- sixteen tiles that cover the 301x301 grid
# from the first pixel to the last.  A stride that does not tile the grid
# makes the verifier score nothing, which is a silent failure.
PATCH_SIZE="${PATCH_SIZE:-128}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
# The dataset enumerates its items as ``sample_index * n_positions +
# position_index``, so sixteen consecutive items are exactly the sixteen
# tiles of one window.  The batch size must therefore be a multiple of the
# number of tiles, or the stitcher never completes a window and the
# verifier scores nothing.
#
# Sixteen 128x128 windows with eleven history frames is a large batch for a
# 24 GiB card; the memory is checked in the log right below, and the
# training falls back to a smaller batch if the card cannot hold it.
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-2}"
QPE_WEIGHT="${QPE_WEIGHT:-0.5}"

# The history supervision weight is what lets the model learn the t0
# inversion at all.  physical_qpe carries it at 0.5 as well, so the two arms
# are supervised on the same quantity with the same weight; the difference
# between them is the mechanism (a learned QPE plus a measured advection
# against a learned QPE plus a copy), not the objective.
#
# physical_qpe additionally takes --qpe-single-frame, because its ten
# output frames are an exact copy of its first one and supervising the copy
# against a changing truth would only blur it.  qlfn2 does not take it: its
# ten frames are ten different advected fields, so they are supervised
# against the ten different truth frames.

# The flags are split between the two stages, because they mean different
# things there:
#
#   * ``--no-nwp`` and ``--model-kwargs`` apply to both the training and the
#     scoring of an arm, and they have to.  An arm that was trained without
#     the NWP but scored with it would measure a different model than the
#     one that was trained; an arm whose advection was switched off at
#     training time but switched back on at scoring time would be scored
#     with the mechanism it was supposed to have removed.  Both would look
#     like a result.
#   * ``--qpe-single-frame`` is a training-only flag; the scoring script
#     does not have it.
#
# The four arms are therefore:
#
#   arm 0  qlfn2                       train: --                          score: --
#   arm 1  qlfn2 without the NWP       train: --no-nwp                    score: --no-nwp
#   arm 2  qlfn2 without the advection train: --model-kwargs motion=none   score: the same
#   arm 3  physical_qpe (the control)  train: --qpe-single-frame          score: --
#
# The motion mode is a runtime switch rather than a parameter: the model
# built with ``motion_mode="none"`` has exactly the same state dict as the
# one built with ``motion_mode="corr"``.  So a checkpoint trained by arm 2
# loads happily into a default model, and the arm would silently be scored
# with the advection it was supposed to have switched off.

case "${ARM}" in
    0)
        ARM_NAME="qlfn2"
        MODEL="qlfn2"
        TRAIN_EXTRA=()
        EVAL_EXTRA=()
        ;;
    1)
        ARM_NAME="qlfn2_no_nwp"
        MODEL="qlfn2"
        TRAIN_EXTRA=(--no-nwp)
        EVAL_EXTRA=(--no-nwp)
        ;;
    2)
        ARM_NAME="qlfn2_no_motion"
        MODEL="qlfn2"
        TRAIN_EXTRA=(--model-kwargs '{"motion_mode": "none"}')
        EVAL_EXTRA=(--model-kwargs '{"motion_mode": "none"}')
        ;;
    3)
        ARM_NAME="physical_qpe"
        MODEL="physical_qpe"
        TRAIN_EXTRA=(--qpe-single-frame)
        EVAL_EXTRA=()
        ;;
    *)
        echo "unknown arm ${ARM}" >&2
        exit 2
        ;;
esac

OUT_DIR="${RUN_ROOT}/${ARM_NAME}/seed${SEED}"
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " qlfn2 experiment   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}   array: ${SLURM_ARRAY_TASK_ID:-none}"
echo "   arm        : ${ARM} (${ARM_NAME})   model: ${MODEL}"
echo "   train flags: ${TRAIN_EXTRA[*]:-<none>}"
echo "   eval flags : ${EVAL_EXTRA[*]:-<none>}"
echo "   seed       : ${SEED}"
echo "   epochs     : ${EPOCHS}   patch: ${PATCH_SIZE}   stride: ${SLIDE_STRIDE}"
echo "   out        : ${OUT_DIR}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"device memory  : {total:.1f} GiB")
PY
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo
echo "--- train -----------------------------------------------------------"
if [[ "${MODEL}" == "qlfn2" ]]; then
    TRAIN_CMD=(python -u "${REPO_ROOT}/scripts/train_qlfn2.py")
else
    TRAIN_CMD=(python -u "${BENCH_ROOT}/scripts/train.py")
fi

"${TRAIN_CMD[@]}" \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --lr 2e-4 \
    --loss balanced \
    --qpe-weight "${QPE_WEIGHT}" \
    --log-every 20 \
    --seed "${SEED}" \
    --device cuda \
    --out-dir "${OUT_DIR}/train" \
    ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "--- score ----------------------------------------------------------"
# The benchmark writes the checkpoint into a timestamped subdirectory of
# the output directory (``<out-dir>/<model>_<unix-time>/last.pt``), so the
# newest one is found rather than assumed.
CKPT="$(ls -1t "${OUT_DIR}"/train/*/last.pt 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
    echo "FATAL: no checkpoint under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint: ${CKPT}"

if [[ "${MODEL}" == "qlfn2" ]]; then
    EVAL_CMD=(python -u "${REPO_ROOT}/scripts/eval_qlfn2.py")
else
    EVAL_CMD=(python -u "${BENCH_ROOT}/scripts/evaluate.py")
fi

"${EVAL_CMD[@]}" \
    --model "${MODEL}" \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split test \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --thresholds 0.1 0.5 1 2 4 8 16 \
    --fss-radii 1 2 4 \
    --seed "${SEED}" \
    --device cuda \
    --out-json "${OUT_DIR}/eval_test.json" \
    ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}
eval_rc=$?
echo "eval rc=${eval_rc}"

# A report full of NaN means the sliding walk left a gap in the grid and
# nothing was scored.  The job would still complete, so the check has to be
# explicit.  It is the same check the pilot uses, in one place, so that the
# two cannot drift apart.
REPORT="${OUT_DIR}/eval_test.json"
python "${REPO_ROOT}/tools/check_eval_report.py" "${REPORT}"
report_rc=$?
echo "report rc=${report_rc}"

echo
echo "=========================================================="
echo " arm ${ARM_NAME} seed ${SEED} finished $(date -Is)"
echo "   train rc = ${train_rc}   eval rc = ${eval_rc}   report rc = ${report_rc}"
echo "   report   = ${REPORT}"
echo "=========================================================="

exit $(( train_rc | eval_rc | report_rc ))

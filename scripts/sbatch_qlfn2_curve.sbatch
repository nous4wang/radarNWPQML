#!/bin/bash
#SBATCH --job-name=qlfn2-curve
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

# One qlfn2 run under the benchmark's own protocol, read off the epoch curve.
#
# Why this script exists, in the order the two defects were found.
#
# Defect 1 -- the wrong checkpoint.  The sweep scored ``last.pt`` and called
# the result "the model".  The benchmark's own epoch curve shows these
# models peak at epoch 0 and decay: physical_qpe scores 0.1226 at ep00, is
# below the persistence floor at ep01 and at 51 % of the floor by ep02.
# ``scripts/train_keep_epochs.py`` exists upstream precisely so the score can
# be read off a known epoch.  This script keeps every epoch and scores every
# epoch, and reports the peak and the last checkpoint separately.
#
# Defect 2 -- the wrong geometry.  The upstream authority scripts all score
# with ``--patch-size 64 --slide-stride 64`` (25 tiles, 13450 samples).  The
# sweep scored with ``--patch-size 128 --slide-stride 64`` (16 tiles, 8608
# samples), which is a different stitched field and therefore a different
# measurement.  The gap is not negligible: the benchmark's own physical_qpe
# epoch_00 scores 0.1226 under 64/64 and 0.1395 under 128/128 -- the same
# weights, the same protocol, +13.8 % from the geometry alone.  This script
# uses 64/64, the geometry the oracle ceiling (0.1653) and the persistence
# floor (0.0924) were measured with, so the numbers are comparable.
#
# The arms differ only in the flags, so a difference in the score is a
# difference in the mechanism:
#
#   0  qlfn2                       motion=corr, CAPPI on, NWP on
#   1  qlfn2  --no-nwp             the NWP is withheld
#   2  qlfn2  motion=none          the advection is switched off
#   3  physical_qpe                the upstream control, same harness
#
# The train/eval flags are split between the two stages, because they mean
# different things there:
#
#   * ``--no-nwp`` and ``--model-kwargs`` apply to BOTH stages.  An arm
#     trained without the NWP but scored with it measures a different model
#     than the one that was trained; an arm whose advection was switched off
#     at training time but switched back on at scoring time is scored with
#     the mechanism it was supposed to have removed.  Both would look like a
#     result.
#   * ``--qpe-single-frame`` is a training-only flag; the scoring script
#     does not have it.
#
# ``motion_mode`` is a runtime switch rather than a parameter: the model
# built with ``motion_mode="none"`` has exactly the same state dict as the
# one built with ``motion_mode="corr"``.  A checkpoint trained by arm 2
# therefore loads happily into a default model, and the arm would silently
# be scored with the advection it was supposed to have switched off.
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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qlfn2_curve}"

ARM="${ARM:-0}"
SEED="${SEED:-10}"
EPOCHS="${EPOCHS:-10}"
# The upstream authority geometry.  64/64 tiles the 301x301 grid into 25
# symmetric positions (origins 0, 59, 118, 178, 237) and is what produced
# the oracle ceiling and the persistence floor this work is compared with.
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
QPE_WEIGHT="${QPE_WEIGHT:-0.5}"
THRESHOLDS="${THRESHOLDS:-0.1 0.5 1 2 4 8 16}"
FSS_RADII="${FSS_RADII:-1 2 4}"

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
echo " qlfn2 epoch curve   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   arm        : ${ARM} (${ARM_NAME})   model: ${MODEL}"
echo "   train flags: ${TRAIN_EXTRA[*]:-<none>}"
echo "   eval flags : ${EVAL_EXTRA[*]:-<none>}"
echo "   seed       : ${SEED}"
echo "   epochs     : ${EPOCHS}"
echo "   geometry   : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
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
echo "--- train (every epoch kept) -----------------------------------------"
if [[ "${MODEL}" == "qlfn2" ]]; then
    TRAIN_CMD=(python -u "${REPO_ROOT}/scripts/train_qlfn2_keep.py")
else
    TRAIN_CMD=(python -u "${BENCH_ROOT}/scripts/train_keep_epochs.py")
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
    --log-every 100 \
    --seed "${SEED}" \
    --device cuda \
    --out-dir "${OUT_DIR}/train" \
    ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "--- locate the checkpoints -------------------------------------------"
# The benchmark writes the checkpoint into a timestamped subdirectory of the
# output directory (``<out-dir>/<model>_<unix-time>/last.pt``), so the
# newest one is found rather than assumed.
CKPT_DIR="$(ls -1dt "${OUT_DIR}"/train/*/ 2>/dev/null | head -1)"
if [[ -z "${CKPT_DIR}" ]]; then
    echo "FATAL: no checkpoint directory under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint dir: ${CKPT_DIR}"
ls -1 "${CKPT_DIR}" | sed 's/^/   /'

echo
echo "--- score every epoch ------------------------------------------------"
# The two flags that have to travel from training to scoring are passed as
# named switches.  They used to travel as a free-form remainder, which
# silently misordered them: ``argparse.REMAINDER`` collects everything after
# the first flag, so a second flag arrived as a stray token.
#
# The switches are spelled out here rather than derived from ``EVAL_EXTRA``,
# because the JSON value of ``--model-kwargs`` is a separate token and
# splitting it out of a shell array by hand is exactly the kind of
# string-handling that goes wrong silently.  One flag per case, each written
# out in full, cannot be misordered.
case "${ARM}" in
    1) SCORE_EXTRA=(--no-nwp) ;;
    2) SCORE_EXTRA=(--model-kwargs '{"motion_mode": "none"}') ;;
    *) SCORE_EXTRA=() ;;
esac

python -u "${REPO_ROOT}/tools/score_epoch_curve.py" \
    --model "${MODEL}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}/curve" \
    --split test \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --thresholds ${THRESHOLDS} \
    --fss-radii ${FSS_RADII} \
    --seed "${SEED}" \
    --device cuda \
    ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"}
score_rc=$?
echo "score rc=${score_rc}"

echo
echo "--- report check ------------------------------------------------------"
CURVE="${OUT_DIR}/curve/${MODEL}_epoch_curve.json"
python - "$CURVE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(f"FATAL: no curve summary at {path}")
    raise SystemExit(5)
summary = json.loads(path.read_text(encoding="utf-8"))
print(f"geometry       : {summary['geometry']}")
print(f"checkpoints    : {len(summary['rows'])}")
if "peak_csi2" in summary:
    print(f"peak           : {summary['peak_checkpoint']}  "
          f"CSI@2.0 = {summary['peak_csi2']:.4f}")
else:
    print("FATAL: no peak in the summary")
    raise SystemExit(6)
if "last_csi2" in summary:
    print(f"last.pt        : CSI@2.0 = {summary['last_csi2']:.4f}")
PY
report_rc=$?
echo "report rc=${report_rc}"

echo
echo "=========================================================="
echo " arm ${ARM_NAME} seed ${SEED} finished $(date -Is)"
echo "   train rc = ${train_rc}   score rc = ${score_rc}   report rc = ${report_rc}"
echo "   curve    = ${CURVE}"
echo "=========================================================="

exit $(( train_rc | score_rc | report_rc ))

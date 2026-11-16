#!/bin/bash
#SBATCH --job-name=qlfn2-score
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

# Score the epochs of an already-trained run.
#
# Why this script exists separately from sbatch_qlfn2_curve.sbatch: jobs 65
# and 66 trained their ten epochs successfully and then failed in the
# scoring stage, because the flag plumbing was wrong (a leftover ``--extra``
# switch, and a JSON value that lost its quoting).  The checkpoints those
# jobs produced are still on disk, so the fix is to re-run only the scoring
# half rather than pay for the training half again.
#
# The scoring protocol is unchanged: the upstream authority geometry
# (patch 64 / stride 64, 25 tiles, 13450 samples) and the benchmark's own
# evaluator.  The number reported is the peak over the epoch curve, with the
# last checkpoint reported separately.

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
BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
DATA_ROOT="${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qlfn2_curve}"

ARM="${ARM:-0}"
SEED="${SEED:-10}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
THRESHOLDS="${THRESHOLDS:-0.1 0.5 1 2 4 8 16}"
FSS_RADII="${FSS_RADII:-1 2 4}"

case "${ARM}" in
    0)
        ARM_NAME="qlfn2"
        MODEL="qlfn2"
        ;;
    1)
        ARM_NAME="qlfn2_no_nwp"
        MODEL="qlfn2"
        ;;
    2)
        ARM_NAME="qlfn2_no_motion"
        MODEL="qlfn2"
        ;;
    3)
        ARM_NAME="physical_qpe"
        MODEL="physical_qpe"
        ;;
    *)
        echo "unknown arm ${ARM}" >&2
        exit 2
        ;;
esac

# The two flags that have to travel from training to scoring are spelled out
# per arm.  They used to travel as a free-form remainder, which misordered
# them: ``argparse.REMAINDER`` collects everything after the first flag, so
# the second one arrived as a stray token and the scorer rejected it.
case "${ARM}" in
    1) SCORE_EXTRA=(--no-nwp) ;;
    2) SCORE_EXTRA=(--model-kwargs '{"motion_mode": "none"}') ;;
    *) SCORE_EXTRA=() ;;
esac

OUT_DIR="${RUN_ROOT}/${ARM_NAME}/seed${SEED}"
CKPT_DIR="$(ls -1dt "${OUT_DIR}"/train/*/ 2>/dev/null | head -1)"
if [[ -z "${CKPT_DIR}" ]]; then
    echo "FATAL: no checkpoint directory under ${OUT_DIR}/train" >&2
    exit 3
fi

echo "=========================================================="
echo " qlfn2 scoring only   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   arm        : ${ARM} (${ARM_NAME})   model: ${MODEL}"
echo "   seed       : ${SEED}"
echo "   geometry   : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   checkpoint : ${CKPT_DIR}"
echo "   score flags: ${SCORE_EXTRA[*]:-<none>}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

n_kept="$(ls -1 "${CKPT_DIR}"/epoch_*.pt 2>/dev/null | wc -l)"
echo "epoch checkpoints on disk: ${n_kept}"
if [[ "${n_kept}" -eq 0 ]]; then
    echo "FATAL: no epoch checkpoints in ${CKPT_DIR}" >&2
    exit 4
fi

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
echo " arm ${ARM_NAME} seed ${SEED} scored $(date -Is)"
echo "   score rc = ${score_rc}   report rc = ${report_rc}"
echo "   curve    = ${CURVE}"
echo "=========================================================="

exit $(( score_rc | report_rc ))

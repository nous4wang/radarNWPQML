#!/bin/bash
#SBATCH --job-name=qlfn2-ctrl
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

# The control arm, trained with the upstream's own recipe.
#
# Why this script exists (F29): the first curve experiment trained the
# upstream control with a recipe of ours -- ``--qpe-weight 0.5`` on top of
# ``--qpe-single-frame``, ten epochs, seed 10/11 -- and got a jagged curve
# whose peak (0.093 / 0.062) sat well below the upstream's own 0.1226.
# The upstream's run that produced 0.1226 was configured differently:
#
#     upstream   runs/physical_qpe/seed0/physical_qpe_1793928770/config.json
#                qpe_weight 0.0
#                epochs     3
#                seed       0
#                config     configs/models/physical_qpe.yaml
#                loss       balanced
#                alpha_csi  0.5
#                lr         2e-4
#                batch      16
#                patch 64 / stride 64
#
# So "qlfn2 beats the upstream control" was, in that run, measured against a
# control we had quietly weakened.  This script removes that.
#
# What it does: train the control with the upstream's recipe, keep every
# epoch, score every epoch under the upstream authority geometry
# (patch 64 / stride 64, 25 tiles, 13450 samples), and report the curve.
#
# The pass condition is written down before the run, not after it:
#
#   the upstream's own log (runs/physical_qpe/eval/epoch_0*.log) records
#       epoch_00  CSI@2.0 = 0.123
#       epoch_01  CSI@2.0 = 0.084
#       epoch_02  CSI@2.0 = 0.047
#
#   If our harness reproduces that curve, the harness is sound and the
#   qlfn2-vs-control comparison becomes meaningful.  If it does not, the
#   harness has another problem, and no "qlfn2 is better" claim may be
#   made until that is found.
#
# The evaluator, the geometry, the thresholds and the split are the
# upstream's own; nothing about the scoring is changed here.

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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qlfn2_control}"

SEED="${SEED:-0}"
# The upstream's own numbers for this recipe, to be reproduced, not matched
# by construction.
UPSTREAM_EPOCHS=3
UPSTREAM_QPE_WEIGHT=0.0
# ``--config`` takes either a path or a bare name.  A bare name is resolved
# against the shipped configurations (configs/models/<name>.yaml), which is
# where physical_qpe.yaml lives; passing the path instead makes the resolver
# append ".yaml" to it and fail.  The upstream's own run recorded the path
# in its config.json because the path is what the resolver returned.
UPSTREAM_CONFIG="physical_qpe"

PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
THRESHOLDS="${THRESHOLDS:-0.1 0.5 1 2 4 8 16}"
FSS_RADII="${FSS_RADII:-1 2 4}"

OUT_DIR="${RUN_ROOT}/seed${SEED}"
mkdir -p "${OUT_DIR}"

echo "=========================================================="
echo " upstream-recipe control   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   seed       : ${SEED}   (the upstream run used seed 0)"
echo "   epochs     : ${UPSTREAM_EPOCHS}   (the upstream run used 3)"
echo "   qpe-weight : ${UPSTREAM_QPE_WEIGHT}   (the upstream run used 0.0)"
echo "   config     : ${UPSTREAM_CONFIG}"
echo "   geometry   : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   out        : ${OUT_DIR}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

echo
echo "--- train with the upstream's recipe (every epoch kept) ----------------"
python -u "${BENCH_ROOT}/scripts/train_keep_epochs.py" \
    --model physical_qpe \
    --data-root "${DATA_ROOT}" \
    --packed \
    --split train \
    --epochs "${UPSTREAM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${WORKERS}" \
    --patch-size "${PATCH_SIZE}" \
    --slide-stride "${SLIDE_STRIDE}" \
    --lr 2e-4 \
    --loss balanced \
    --alpha-csi 0.5 \
    --qpe-weight "${UPSTREAM_QPE_WEIGHT}" \
    --qpe-single-frame \
    --log-every 100 \
    --seed "${SEED}" \
    --device cuda \
    --config "${UPSTREAM_CONFIG}" \
    --out-dir "${OUT_DIR}/train"
train_rc=$?
echo "train rc=${train_rc}"

echo
echo "--- locate the checkpoints -------------------------------------------"
CKPT_DIR="$(ls -1dt "${OUT_DIR}"/train/*/ 2>/dev/null | head -1)"
if [[ -z "${CKPT_DIR}" ]]; then
    echo "FATAL: no checkpoint directory under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint dir: ${CKPT_DIR}"
ls -1 "${CKPT_DIR}" | sed 's/^/   /'

echo
echo "--- the recipe actually used ------------------------------------------"
cat "${CKPT_DIR}/config.json" | sed 's/^/   /'

echo
echo "--- score every epoch ------------------------------------------------"
python -u "${REPO_ROOT}/tools/score_epoch_curve.py" \
    --model physical_qpe \
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
    --device cuda
score_rc=$?
echo "score rc=${score_rc}"

echo
echo "--- compare against the upstream's own numbers -------------------------"
# The pass condition, checked by the job itself.
python - "${OUT_DIR}/curve/physical_qpe_epoch_curve.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(f"FATAL: no curve summary at {path}")
    raise SystemExit(5)

summary = json.loads(path.read_text(encoding="utf-8"))
print(f"geometry    : {summary['geometry']}")
print(f"checkpoints : {len(summary['rows'])}")

# The upstream's own logs, runs/physical_qpe/eval/epoch_0*.log.
EXPECTED = {0: 0.123, 1: 0.084, 2: 0.047}
TOLERANCE = 0.005

rows = {r.get("epoch"): r for r in summary["rows"] if not r.get("is_last")}
print()
print(f"{'epoch':>6} {'expected':>10} {'measured':>10} {'diff':>9}  verdict")
ok = True
for epoch, expected in sorted(EXPECTED.items()):
    row = rows.get(epoch)
    if row is None:
        print(f"{epoch:>6} {expected:>10.3f} {'-':>10} {'-':>9}  MISSING")
        ok = False
        continue
    measured = row["csi_2.0"]
    diff = measured - expected
    verdict = "match" if abs(diff) <= TOLERANCE else "MISMATCH"
    if abs(diff) > TOLERANCE:
        ok = False
    print(f"{epoch:>6} {expected:>10.3f} {measured:>10.4f} {diff:+9.4f}  {verdict}")

print()
if ok:
    print("VERDICT: the harness reproduces the upstream's control curve.")
    print("         The control arm is trustworthy, and qlfn2 may be compared")
    print("         against it.")
else:
    print("VERDICT: the harness does NOT reproduce the upstream's control curve.")
    print("         The control arm stays untrustworthy until this is explained;")
    print("         no 'qlfn2 beats the control' claim may be made from it.")
raise SystemExit(0 if ok else 7)
PY
verify_rc=$?
echo "verify rc=${verify_rc}"

echo
echo "=========================================================="
echo " control recipe seed ${SEED} finished $(date -Is)"
echo "   train rc = ${train_rc}   score rc = ${score_rc}   verify rc = ${verify_rc}"
echo "   curve    = ${OUT_DIR}/curve/physical_qpe_epoch_curve.json"
echo "=========================================================="

exit $(( train_rc | score_rc ))

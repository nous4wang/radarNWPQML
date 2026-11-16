#!/bin/bash
#SBATCH --job-name=qlfn2-f36
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

# F36 -- mechanism attribution for F35's negative NWP effect.
#
# F35 measured a reproducible *negative* NWP effect (-0.0528 / -0.0578 on
# median CSI@2.0) with arm0 (NWP on) vs arm1 (--no-nwp).  A later code audit
# found the context builder was reading NWP channels 0..4 by position instead
# of the five named variables, and the audit also could not separate the two
# NWP subpaths (halo_proj: a global 20-d context projection; nwp_gate: a
# local multiplicative 1x1 gate).  This experiment re-runs the same protocol
# after fixing the channel semantics, with the two subpaths split orthogonally.
#
# The arms differ ONLY in --model-kwargs / --no-nwp:
#   0  both_fixed   use_halo=true,  use_nwp_gate=true   (fixed-channel full model)
#   1  halo_only    use_halo=true,  use_nwp_gate=false
#   2  gate_only    use_halo=false, use_nwp_gate=true
#   3  no_nwp       --no-nwp (all-zero tensor), the in-batch control
#
# CRITICAL: use_halo / use_nwp_gate are CONSTRUCTOR arguments.  Turning one off
# removes real parameters from state_dict (32 -> 30 / 28 keys).  Unlike
# motion_mode, this is not a runtime switch.  The same JSON MUST be passed to
# training AND to scoring; a checkpoint from one arm cannot be loaded by
# another arm.  Both stages spell the flag out explicitly below.
#
# Geometry is the upstream authority: patch 64 / stride 64, 25 tiles,
# n_samples=538.  Every epoch is kept and scored; the reported number is the
# median (primary) and the peak (auxiliary) of the epoch curve, never last.pt.
#
# The GPU is only visible inside a Slurm job; the login slice's device cgroup
# refuses /dev/nvidiactl with EPERM.  Use cuda:0 and keep Slurm's
# CUDA_VISIBLE_DEVICES.

set -eo pipefail

# conda's activation scripts reference unset variables, so set -u must be off
# during activation (MKL_INTERFACE_LAYER: unbound variable otherwise).
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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/f36_nwp}"

ARM="${ARM:-0}"
SEED="${SEED:-10}"
EPOCHS="${EPOCHS:-10}"
PATCH_SIZE="${PATCH_SIZE:-64}"
SLIDE_STRIDE="${SLIDE_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
QPE_WEIGHT="${QPE_WEIGHT:-0.5}"
THRESHOLDS="${THRESHOLDS:-0.1 0.5 1 2 4 8 16}"
FSS_RADII="${FSS_RADII:-1 2 4}"

case "${ARM}" in
    0)
        ARM_NAME="both_fixed"
        ARM_KWARGS='{"use_halo": true, "use_nwp_gate": true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    1)
        ARM_NAME="halo_only"
        ARM_KWARGS='{"use_halo": true, "use_nwp_gate": false}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    2)
        ARM_NAME="gate_only"
        ARM_KWARGS='{"use_halo": false, "use_nwp_gate": true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    3)
        ARM_NAME="no_nwp"
        ARM_KWARGS='{}'
        TRAIN_EXTRA=(--no-nwp)
        SCORE_EXTRA=(--no-nwp)
        ;;
    *)
        echo "unknown arm ${ARM}" >&2
        exit 2
        ;;
esac
export ARM_NAME ARM_KWARGS

OUT_DIR="${RUN_ROOT}/${ARM_NAME}/seed${SEED}"
mkdir -p "${OUT_DIR}" run_logs/slurm

echo "=========================================================="
echo " F36 mechanism attribution   $(date -Is)"
echo "   job id     : ${SLURM_JOB_ID}"
echo "   arm        : ${ARM} (${ARM_NAME})"
echo "   model-kwargs: ${ARM_KWARGS}"
echo "   train flags: ${TRAIN_EXTRA[*]:-<none>}"
echo "   eval flags : ${SCORE_EXTRA[*]:-<none>}"
echo "   seed       : ${SEED}   epochs: ${EPOCHS}"
echo "   geometry   : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   out        : ${OUT_DIR}"
echo "   python     : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

echo
echo "--- arm identity preflight (before any GPU time) ----------------------"
python - <<'PY'
import json
import os
import sys

from qlfn2.qlfn2_model import (
    HALO_CONTEXT_INDICES, NWP_VARS, QLFN2Adapter, archive_to_net,
)
import torch

name = os.environ["ARM_NAME"]
kwargs = json.loads(os.environ.get("ARM_KWARGS", "{}"))
print("NWP_VARS[22:24]      :", NWP_VARS[22:24])
print("HALO_CONTEXT_INDICES :", HALO_CONTEXT_INDICES)
assert HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23), "channel table drifted"
assert NWP_VARS[23] == "TTdMean74", "TTdMean74 is not at index 23"

torch.manual_seed(103)
model = QLFN2Adapter(width=8, **kwargs)
has_halo = model.core.halo_proj is not None
has_gate = model.core.nwp_gate is not None
want = {
    "both_fixed": (True, True),
    "halo_only": (True, False),
    "gate_only": (False, True),
    "no_nwp": (True, True),
}[name]
print(f"arm {name}: halo_proj={has_halo} nwp_gate={has_gate} "
      f"state_dict_keys={len(model.state_dict())} want={want}")
if (has_halo, has_gate) != want:
    print("FATAL: --model-kwargs did not construct the intended arm")
    sys.exit(4)

raw = torch.zeros(2, 11, 10, 16, 16)
raw[:, :, 6] = 300.0
raw[:, :, 9] = 200.0
comp = archive_to_net(raw)
nwp = torch.rand(2, 2, 27, 16, 16)
with torch.no_grad():
    model(comp, nwp)
print("diagnostics           :", model.last_diagnostics)
PY
preflight_rc=$?
echo "preflight rc=${preflight_rc}"
if [[ "${preflight_rc}" -ne 0 ]]; then
    exit 4
fi

echo
echo "--- self-checks (30/30 required) -------------------------------------"
python -m qlfn2.checks
checks_rc=$?
echo "checks rc=${checks_rc}"
if [[ "${checks_rc}" -ne 0 ]]; then
    exit 5
fi

python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
PY

echo
echo "--- train (every epoch kept) -----------------------------------------"
python -u "${REPO_ROOT}/scripts/train_qlfn2_keep.py" \
    --model qlfn2 \
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
CKPT_DIR="$(ls -1dt "${OUT_DIR}"/train/*/ 2>/dev/null | head -1)"
if [[ -z "${CKPT_DIR}" ]]; then
    echo "FATAL: no checkpoint directory under ${OUT_DIR}/train" >&2
    exit 3
fi
echo "checkpoint dir: ${CKPT_DIR}"
ls -1 "${CKPT_DIR}" | sed 's/^/   /'

echo
echo "--- score every epoch (same flags as training) ------------------------"
python -u "${REPO_ROOT}/tools/score_epoch_curve.py" \
    --model qlfn2 \
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
CURVE="${OUT_DIR}/curve/qlfn2_epoch_curve.json"
python - "$CURVE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(f"FATAL: no curve summary at {path}")
    raise SystemExit(5)
summary = json.loads(path.read_text(encoding="utf-8"))
rows = [r for r in summary["rows"] if r.get("epoch") is not None
        and "csi_2.0" in r]
print(f"geometry       : {summary['geometry']}")
print(f"scored epochs  : {len(rows)}")
print(f"n_samples      : {sorted({r.get('n_samples') for r in rows})}")
if not rows or len(rows) != 10:
    print("FATAL: expected 10 scored epoch checkpoints")
    raise SystemExit(6)
if summary["geometry"] != {"patch_size": 64, "slide_stride": 64}:
    print("FATAL: geometry is not 64/64")
    raise SystemExit(7)
if {r.get("n_samples") for r in rows} != {538}:
    print("FATAL: n_samples is not 538")
    raise SystemExit(8)
print(f"peak           : {summary['peak_checkpoint']}  "
      f"CSI@2.0 = {summary['peak_csi2']:.4f}")
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

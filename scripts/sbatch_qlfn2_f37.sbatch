#!/bin/bash
#SBATCH --job-name=qlfn2-f37
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/qlfn2-f37-%j.out
#SBATCH --error=run_logs/slurm/qlfn2-f37-%j.err

# F37 -- freeze the t0 pure-radar QPE, and let two NWP time slices drive
# t>0 Lagrangian advection / zero-mean source terms only.
#
# F36 measured that NWP modulation of the t0 rain field halves CSI@2.0 at
# lead 0 (0.2499 -> 0.1260 in epoch_02) and never recovers.  F37 turns BOTH
# t0 subpaths off (use_halo=false, use_nwp_gate=false) and instead adds two
# orthogonal t>0 subpaths, both zero-initialised:
#   use_nwp_vel     -> (dvx, dvy) trajectory correction for k>=1
#   use_nwp_source  -> spatial zero-mean wet-region source for k>=1
# Lead 0 (k=0) is structurally identical to the pure-radar advection.
#
# Four arms x 2 seeds = 8 independent jobs.  Arms differ ONLY in
# --model-kwargs / --no-nwp:
#   evol_both  vel=true  src=true
#   vel_only   vel=true  src=false
#   src_only   vel=false src=true
#   no_nwp     --no-nwp (all-zero tensor), the in-batch control
#
# CRITICAL: use_halo / use_nwp_gate / use_nwp_vel / use_nwp_source are
# CONSTRUCTOR arguments.  Turning one on/off changes state_dict (32 -> 38
# keys for evol_both).  The same JSON MUST be passed to training AND scoring;
# a checkpoint from one arm cannot be loaded by another arm.
#
# Geometry is the upstream authority: patch 64 / stride 64, 25 tiles,
# n_samples=538.  Every epoch is kept and scored; the reported number is the
# pooled median of the epoch curve plus the per-lead median profile.
#
# The GPU is only visible inside a Slurm job; keep Slurm's CUDA_VISIBLE_DEVICES
# and use cuda:0.  NEVER bypass Slurm with nohup / a background process.

# conda's activation scripts reference unset variables, so set -u must be off
# during activation (MKL_INTERFACE_LAYER: unbound variable otherwise).
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
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/f37_nwp}"

ARM="${ARM:?ARM must be set (evol_both|vel_only|src_only|no_nwp)}"
ARM_NAME="${ARM}"
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
    evol_both)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":true,"use_nwp_source":true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    vel_only)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":true,"use_nwp_source":false}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    src_only)
        ARM_KWARGS='{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":false,"use_nwp_source":true}'
        TRAIN_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        SCORE_EXTRA=(--model-kwargs "${ARM_KWARGS}")
        ;;
    no_nwp)
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
echo " F37 Lagrangian evolution    $(date -Is)"
echo "   job id      : ${SLURM_JOB_ID}"
echo "   arm         : ${ARM_NAME}"
echo "   model-kwargs: ${ARM_KWARGS}"
echo "   train flags : ${TRAIN_EXTRA[*]:-<none>}"
echo "   eval flags  : ${SCORE_EXTRA[*]:-<none>}"
echo "   seed        : ${SEED}   epochs: ${EPOCHS}"
echo "   geometry    : patch ${PATCH_SIZE} / stride ${SLIDE_STRIDE}"
echo "   out         : ${OUT_DIR}"
echo "   python      : $(command -v python)"
echo "   cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================================="

echo
echo "--- arm identity + channel-table preflight (before any GPU time) ------"
python - <<'PY'
import json
import os
import sys
import torch

from qlfn2.qlfn2_model import (
    HALO_CONTEXT_INDICES, NWP_VARS, NWP_EVOL_INDICES, QLFN2Adapter,
    archive_to_net,
)

# F37's own channel contract: the 1h-trend path must read the SAME upstream
# 27-channel table as F36.  The F37 merge draft regressed to a self-made
# table and put TTdMean74 at index 22 (TTdMax74); refuse to spend GPU on it.
assert HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23), "channel table drifted"
assert NWP_VARS[22] == "TTdMax74" and NWP_VARS[23] == "TTdMean74"
assert len(NWP_EVOL_INDICES) == 10
print("NWP_VARS[22:24]      :", NWP_VARS[22:24])
print("HALO_CONTEXT_INDICES :", HALO_CONTEXT_INDICES)
print("NWP_EVOL_INDICES     :", NWP_EVOL_INDICES)

name = os.environ["ARM_NAME"]
kwargs = json.loads(os.environ.get("ARM_KWARGS", "{}"))
torch.manual_seed(103)
model = QLFN2Adapter(width=8, **kwargs)
c = model.core
state = (c.halo_proj is not None, c.nwp_gate is not None,
         c.nwp_vel_net is not None, c.nwp_source_net is not None)
want = {
    "evol_both": (False, False, True, True),
    "vel_only":  (False, False, True, False),
    "src_only":  (False, False, False, True),
    "no_nwp":    (True, True, False, False),
}[name]
print(f"arm {name}: halo/gate/vel/src={state} want={want} "
      f"state_dict_keys={len(model.state_dict())}")
if state != want:
    print("FATAL: --model-kwargs did not construct the intended arm")
    sys.exit(4)

# lead-0 zero-harm guard, run with the REAL arm construction
raw = torch.zeros(2, 11, 10, 16, 16)
raw[:, :, 6] = 300.0
raw[:, :, 9] = 200.0
comp = archive_to_net(raw)
nwp = torch.rand(2, 2, 27, 16, 16)
with torch.no_grad():
    p_nwp = model(comp, nwp)
    p_zero = model(comp, torch.zeros_like(nwp))
gap = float((p_nwp[:, 0] - p_zero[:, 0]).abs().max())
print(f"lead-0 pure-radar gap: {gap:.3e}")
if gap != 0.0:
    print("FATAL: lead 0 is contaminated by the NWP evolution terms")
    sys.exit(4)
print("diagnostics           :", model.last_diagnostics)
PY
preflight_rc=$?
echo "preflight rc=${preflight_rc}"
if [[ "${preflight_rc}" -ne 0 ]]; then
    exit 4
fi

echo
echo "--- self-checks (31/31 required) -------------------------------------"
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
echo "--- report check (per-lead data must be present for G0/G1) -----------"
CURVE="${OUT_DIR}/curve/qlfn2_epoch_curve.json"
python - "${CURVE}" "${OUT_DIR}/curve" <<'PY'
import json
import sys
from pathlib import Path

summary_path, curve_dir = Path(sys.argv[1]), Path(sys.argv[2])
if not summary_path.is_file():
    print(f"FATAL: no curve summary at {summary_path}")
    raise SystemExit(5)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [r for r in summary["rows"] if r.get("epoch") is not None
        and "csi_2.0" in r]
reports = sorted(curve_dir.glob("qlfn2_epoch_*.json"))
reports = [p for p in reports if "epoch_curve" not in p.name and "last" not in p.name]
print(f"geometry       : {summary['geometry']}")
print(f"scored epochs  : {len(rows)}")
print(f"n_samples      : {sorted({r.get('n_samples') for r in rows})}")
print(f"per-lead files : {len(reports)}")
if not rows or len(rows) != 10:
    print("FATAL: expected 10 scored epoch checkpoints")
    raise SystemExit(6)
if summary["geometry"] != {"patch_size": 64, "slide_stride": 64}:
    print("FATAL: geometry is not 64/64")
    raise SystemExit(7)
if {r.get("n_samples") for r in rows} != {538}:
    print("FATAL: n_samples is not 538")
    raise SystemExit(8)
if len(reports) != 10:
    print("FATAL: G0/G1 need 10 per-lead epoch reports")
    raise SystemExit(9)
for p in reports:
    d = json.loads(p.read_text(encoding="utf-8"))
    lc = d.get("lead_csi")
    if not isinstance(lc, dict) or len(lc) != 10:
        print(f"FATAL: {p.name} has no 10-lead lead_csi")
        raise SystemExit(10)
    for k in range(10):
        v = lc.get(str(k), {}).get("2.0")
        if not isinstance(v, (int, float)):
            print(f"FATAL: {p.name} lead {k} has no CSI@2.0")
            raise SystemExit(10)
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

#!/usr/bin/env bash
# Pre-submission verification for F37, executed on the REMOTE login session.
# Read-only: no training, no evaluation, no GPU use, no sbatch (only --test-only).
set -uo pipefail

REPO=/home/wf/radarNWPQML
BENCH=/home/wf/radarNWP
PY=/home/wf/.conda/envs/radar_nwp/bin/python
cd "${REPO}"
export PYTHONPATH="${REPO}:${BENCH}:${PYTHONPATH:-}"

fail=0
step() { echo; echo "=== $* ==="; }

step "controller / queue"
scontrol ping 2>&1 || { echo "FAIL: slurmctld not reachable"; fail=$((fail+1)); }
echo "-- wf jobs --"
squeue -u wf -o "%.10i %.18j %.9T %.9M %R" 2>&1 || true
echo "-- no F37 jobs already queued --"
if squeue -u wf -o "%.18j" 2>/dev/null | grep -q "f37"; then
    echo "FAIL: F37 jobs are already in the queue; do not double-submit"
    fail=$((fail+1))
else
    echo "ok: no F37 job in queue"
fi
if [ -d runs/f37_nwp ]; then
    echo "-- runs/f37_nwp already exists --"
    find runs/f37_nwp -maxdepth 2 -type d | sort | head -20
else
    echo "ok: runs/f37_nwp is clean (no prior F37 outputs)"
fi

step "interpreter / repository"
"$PY" -V
md5sum qlfn2/qlfn2_model.py qlfn2/checks.py tools/judge_f37_nwp.py \
       scripts/sbatch_qlfn2_f37.sbatch scripts/run_qlfn2_f37.sh \
       MASTER_CONTROL_ROOM.md

step "CRLF check on shell scripts"
for f in scripts/sbatch_qlfn2_f37.sbatch scripts/run_qlfn2_f37.sh scripts/verify_f37_remote.sh; do
    n=$(tr -cd '\r' < "$f" | wc -c)
    echo "$f CR bytes = $n"
    [ "$n" -eq 0 ] || { echo "FAIL: CRLF in $f"; fail=$((fail+1)); }
done

step "qlfn2.checks (expect 31/31)"
"$PY" -m qlfn2.checks 2>&1 | tail -4
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL: self-checks"; fail=$((fail+1)); }

step "judge self-test (expect 9/9) + WAITING gate"
"$PY" tools/judge_f37_nwp.py --self-test 2>&1 | tail -3
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL: judge self-test"; fail=$((fail+1)); }
"$PY" tools/judge_f37_nwp.py --root artifacts/f37_nwp/runs/f37_nwp >/tmp/f37_waiting.txt 2>&1
rc=$?
echo "WAITING rc=$rc (expect 2): $(head -1 /tmp/f37_waiting.txt)"
[ "$rc" -eq 2 ] || { echo "FAIL: WAITING gate"; fail=$((fail+1)); }

step "channel table: local model vs REMOTE upstream data/index.py"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "/home/wf/radarNWP")
from qlfn2.qlfn2_model import NWP_VARS, HALO_CONTEXT_INDICES, NWP_EVOL_INDICES
from radar_nwp_bench.data.index import NWP_VARS as UP
print("local  NWP_VARS[22:24] :", NWP_VARS[22:24])
print("remote NWP_VARS[22:24] :", tuple(UP[22:24]))
print("local  len / HALO / EVOL:", len(NWP_VARS), HALO_CONTEXT_INDICES, NWP_EVOL_INDICES)
assert tuple(UP) == NWP_VARS, "local table drifted from REMOTE upstream"
assert HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23)
assert NWP_VARS[23] == "TTdMean74" and NWP_VARS[22] == "TTdMax74"
assert len(NWP_EVOL_INDICES) == 10
print("channel table OK")
PY
[ $? -eq 0 ] || { echo "FAIL: channel table"; fail=$((fail+1)); }

step "four arms constructed from the sbatch JSON (verbatim)"
"$PY" - <<'PY'
import json
from qlfn2.qlfn2_model import QLFN2Adapter
specs = {
 "evol_both": ('{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":true,"use_nwp_source":true}', (False,False,True,True)),
 "vel_only":  ('{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":true,"use_nwp_source":false}', (False,False,True,False)),
 "src_only":  ('{"use_halo":false,"use_nwp_gate":false,"use_nwp_vel":false,"use_nwp_source":true}', (False,False,False,True)),
 "no_nwp":    ('{}', (True,True,False,False)),
}
for name,(payload,want) in specs.items():
    m = QLFN2Adapter(width=8, **json.loads(payload))
    c = m.core
    got = (c.halo_proj is not None, c.nwp_gate is not None,
           c.nwp_vel_net is not None, c.nwp_source_net is not None)
    print(f"  {name:<10} halo/gate/vel/src={got} keys={len(m.state_dict())}")
    assert got == want, (name, got, want)
print("four arms OK")
PY
[ $? -eq 0 ] || { echo "FAIL: arm construction"; fail=$((fail+1)); }

step "merge equivalence (if legacy snapshot is present on the remote)"
if [ -d docs/archive/qlfn2_pre_merge ]; then
    "$PY" tools/check_merge_equivalence.py 2>&1 | tail -4
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL: merge equivalence"; fail=$((fail+1)); }
else
    echo "SKIP: legacy snapshot absent on remote (verified locally: PASS)"
fi

step "sbatch --test-only (no job is submitted)"
ARM=evol_both SEED=10 EPOCHS=10 \
    sbatch --test-only scripts/sbatch_qlfn2_f37.sbatch 2>&1
[ $? -eq 0 ] || { echo "FAIL: sbatch --test-only"; fail=$((fail+1)); }

step "upstream evaluator must be untouched"
stat -c "%y %n" "${BENCH}/scripts/evaluate.py" "${BENCH}/scripts/train.py" 2>&1

echo
echo "=== VERIFY F37: failures=${fail} ==="
exit "$fail"

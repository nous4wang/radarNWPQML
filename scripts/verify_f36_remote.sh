#!/usr/bin/env bash
# F36 pre-submission verification, run on the REMOTE host in a login session.
# CPU only: no training, no evaluation, no GPU job is submitted here.
#
#   MSYS_NO_PATHCONV=1 python tools/remote.py --sync <files> -- \
#       'bash scripts/verify_f36_remote.sh'
#
# The login slice cannot see the GPU (device cgroup EPERM); that is expected
# and is exactly why the real work goes through sbatch.

set -uo pipefail
cd /home/wf/radarNWPQML
PY=/home/wf/.conda/envs/radar_nwp/bin/python

echo "=== 1. qlfn2 self-checks (30/30 required) ==="
"$PY" -m qlfn2.checks | tail -4
checks_rc=${PIPESTATUS[0]}
echo "checks rc=${checks_rc}"

echo
echo "=== 2. NWP channel table: local copy vs REMOTE upstream ==="
"$PY" - <<'PY'
import sys
sys.path.insert(0, "/home/wf/radarNWP")
from radar_nwp_bench.data.index import NWP_VARS as UP
from qlfn2.qlfn2_model import NWP_VARS as LOCAL, HALO_CONTEXT_INDICES
print("remote upstream len :", len(UP))
print("remote [22:24]      :", UP[22:24])
print("local == upstream   :", tuple(UP) == LOCAL)
print("HALO_CONTEXT_INDICES:", HALO_CONTEXT_INDICES)
assert tuple(UP) == LOCAL, "local NWP_VARS drifted from REMOTE upstream"
assert HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23), "channel indices drifted"
PY
table_rc=$?
echo "table rc=${table_rc}"

echo
echo "=== 3. --model-kwargs JSON constructs the intended arm ==="
"$PY" - <<'PY'
import json
from qlfn2.qlfn2_model import QLFN2Adapter
specs = {
    "both_fixed": ('{"use_halo": true, "use_nwp_gate": true}', True, True),
    "halo_only": ('{"use_halo": true, "use_nwp_gate": false}', True, False),
    "gate_only": ('{"use_halo": false, "use_nwp_gate": true}', False, True),
    "no_nwp": ('{}', True, True),
}
for name, (payload, want_halo, want_gate) in specs.items():
    m = QLFN2Adapter(width=8, **json.loads(payload))
    hh = m.core.halo_proj is not None
    gg = m.core.nwp_gate is not None
    print(f"{name:11s} halo={hh} gate={gg} keys={len(m.state_dict())} "
          f"want=({want_halo},{want_gate})")
    assert (hh, gg) == (want_halo, want_gate), name
print("arm construction OK")
PY
kwargs_rc=$?
echo "kwargs rc=${kwargs_rc}"

echo
echo "=== 4. file integrity / line endings ==="
md5sum qlfn2/qlfn2_model.py qlfn2/checks.py MASTER_CONTROL_ROOM.md \
    scripts/sbatch_qlfn2_f36.sbatch scripts/run_qlfn2_f36.sh \
    scripts/verify_f36_remote.sh tools/judge_f36_nwp.py
echo "CR byte count (must be 0):"
for f in scripts/sbatch_qlfn2_f36.sbatch scripts/run_qlfn2_f36.sh \
         scripts/verify_f36_remote.sh; do
    printf "  %s : " "$f"
    tr -cd '\r' < "$f" | wc -c
done

echo
echo "=== 5. upstream evaluator untouched ==="
stat -c '%y %n' /home/wf/radarNWP/scripts/evaluate.py

echo
echo "=== 6. judge branch self-test ==="
"$PY" tools/judge_f36_nwp.py --self-test | tail -2
judge_rc=${PIPESTATUS[0]}
echo "judge rc=${judge_rc}"

echo
echo "=== 7. sbatch --test-only ==="
mkdir -p run_logs/slurm
sbatch --test-only scripts/sbatch_qlfn2_f36.sbatch
sbatch_rc=$?
echo "sbatch --test-only rc=${sbatch_rc}"

echo
echo "=== 8. queue ==="
squeue -u "$USER" -o "%.10i %.18j %.9T %.9M %R" || true

echo
echo "=== SUMMARY ==="
echo "checks=${checks_rc} table=${table_rc} kwargs=${kwargs_rc} "\
"judge=${judge_rc} sbatch_test=${sbatch_rc}"
exit $(( checks_rc | table_rc | kwargs_rc | judge_rc | sbatch_rc ))

#!/usr/bin/env bash
# Run the qmlbench verification chain directly on the server.
#
#   cd /home/wf/radarNWPQML && bash scripts/run_remote.sh
#
# Why this wrapper exists: qmlbench.sources.BenchSource defaults to the
# Windows path R:/1radarNWPprediction/radar-nwp-bench, which does not exist
# on the server. Every check therefore needs RADAR_NWP_BENCH_ROOT pointing
# at the evidence mirror, and the three runtime checks additionally need
# RADAR_NWP_LIVE_ROOT pointing at the live upstream checkout. The deploy
# driver exports both; this script does the same for a hand-typed SSH session.
set -u

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${QMLBENCH_PYTHON:-/home/wf/.conda/envs/radar_nwp/bin/python}"
export RADAR_NWP_BENCH_ROOT="${RADAR_NWP_BENCH_ROOT:-$WS/bench_ref}"
export RADAR_NWP_LIVE_ROOT="${RADAR_NWP_LIVE_ROOT:-/home/wf/radarNWP}"

cd "$WS" || exit 1
mkdir -p artifacts

echo "workspace : $WS"
echo "python    : $PY"
echo "evidence  : $RADAR_NWP_BENCH_ROOT"
echo "live      : $RADAR_NWP_LIVE_ROOT"
echo

fails=0
run() {
    label="$1"
    shift
    echo "----- $label -----"
    "$@"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] $label (exit=$rc)"
        fails=$((fails + 1))
    else
        echo "[PASS] $label"
    fi
    echo
}

run "interpreter + import" \
    "$PY" -c "import sys, qmlbench; print(sys.version.split()[0], qmlbench.__file__)"
run "regenerate catalogue from evidence mirror" \
    "$PY" scripts/sync_baselines.py --out artifacts/baselines_remote.json
run "verify_integration (14 checks)" \
    "$PY" scripts/verify_integration.py
run "export tables" \
    "$PY" scripts/export_tables.py
run "live runtime registry cross-check" \
    "$PY" scripts/verify_runtime_registry.py
run "GPU forward-contract smoke (25 baselines)" \
    "$PY" scripts/smoke_forward.py --seconds 45 --out artifacts/smoke_forward.json
run "real data pipeline contract" \
    "$PY" scripts/verify_data_contract.py --out artifacts/data_contract.json

# configs/baselines.json stores the originating workstation's absolute
# path and git revision, so this probe is expected to fail off-workstation.
# It is a portability probe, not a correctness check.
echo "----- sync --check (portability probe, expected non-zero) -----"
"$PY" scripts/sync_baselines.py --check
echo "[expected] sync --check exit=$?"
echo

echo "===== summary: $fails unexpected failure(s) ====="
exit "$fails"

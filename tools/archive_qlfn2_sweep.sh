#!/usr/bin/env bash
# Archive the qlfn2 sweep: pull the reports, the checkpoints and the Slurm
# logs off the remote host, then build the summary table locally.
#
# The archive is one tarball rather than a dozen downloads, because the
# remote transport is one connection per file and the sweep produces eight
# reports, eight checkpoints and eight logs.
#
# Usage: bash tools/archive_qlfn2_sweep.sh [tag]
#
# "tag" defaults to today's date and is used for the local directory name
# and for the remote tarball name, so that several archives can live side
# by side.

set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-$(date +%Y%m%d%H%M)}"
PY="/c/Users/nous/anaconda3/python.exe"
REMOTE="tools/remote.py"
REMOTE_ROOT="/home/wf/radarNWPQML"
TARBALL="qlfn2_sweep_${TAG}.tar.gz"
LOCAL_DIR="artifacts/qlfn2_sweep_${TAG}"
LOCAL_TAR="${LOCAL_DIR}/${TARBALL}"

echo "--- 1/4  packing the sweep on the remote host ---"
MSYS_NO_PATHCONV=1 "$PY" "$REMOTE" -- \
    "cd ${REMOTE_ROOT} && tar czf /tmp/${TARBALL} runs/qlfn2_sweep run_logs/slurm && \
     ls -lh /tmp/${TARBALL}"

echo
echo "--- 2/4  downloading the archive ---"
mkdir -p "${LOCAL_DIR}"
MSYS_NO_PATHCONV=1 "$PY" "$REMOTE" --download "/tmp/${TARBALL}" --out "${LOCAL_TAR}"

echo
echo "--- 3/4  unpacking ---"
tar -xzf "${LOCAL_TAR}" -C "${LOCAL_DIR}"
find "${LOCAL_DIR}/runs" -name eval_test.json | sort

echo
echo "--- 4/4  building the summary ---"
"$PY" tools/summarize_qlfn2_sweep.py \
    --root "${LOCAL_DIR}/runs/qlfn2_sweep" \
    --markdown "${LOCAL_DIR}/summary.md" \
    --json "${LOCAL_DIR}/summary.json"

echo
echo "archive: ${LOCAL_DIR}"

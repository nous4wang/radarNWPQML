#!/usr/bin/env bash
# Launch the two baseline arms on a seed set that has never been used.
#
# Why this wrapper exists
# -----------------------
# The first queue reused seeds 0..9 and turned out to be a byte-exact
# deterministic replay of the archived runs (20/20 identical epoch_00.pt
# md5, 20/20 identical scores). A replay is not an independent resample,
# so it cannot tell us whether "qpe_0 beats base_concat" survives a new
# draw of training randomness.
#
# Server-side audit before this launch: every saved "seed" field under
# /home/wf/radarNWP and every archived eval filename uses 0..9 only; no
# --seed 10..19 appears in any script or log. Seeds 10..19 are therefore
# genuinely unused.
#
# Isolation: everything goes to runs/seeds_10_19, a directory that did not
# exist before. Nothing in runs/seeds10 is read or overwritten, and the
# summariser is told the expected seed set explicitly so a complete
# 10..19 batch is not judged against the 0..9 completeness rule.
#
# Usage (from the workstation, detached):
#   RADAR_ROOT=/home/wf/radarNWPQML MSYS_NO_PATHCONV=1 \
#     python tools/remote.py --script scripts/run_newseed_queue.sh \
#       --detached /home/wf/radarNWPQML/runs/seeds_10_19_queue.log

set -euo pipefail

QUEUE="/home/wf/radarNWPQML/scripts/run_seed10_queue.sh"
export SEEDS="${SEEDS:-10 11 12 13 14 15 16 17 18 19}"
export OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/seeds_10_19}"
if [[ -z "${SUMMARISER_ARGS:-}" ]]; then
    SUMMARISER_ARGS="--expected-seeds ${SEEDS}"
fi
export SUMMARISER_ARGS

# The detached launcher redirects its output to runs/*_queue.log before this
# script starts, so the log's parent already exists; these are the queue's
# own outputs and are created here rather than assumed.
mkdir -p "${OUT_DIR}/train" "${OUT_DIR}/evals" "${OUT_DIR}/logs"

echo "[newseed] fresh-seed replication: arms=qpe_0 base_concat seeds=${SEEDS}"
echo "[newseed] out=${OUT_DIR} summariser_args=${SUMMARISER_ARGS}"
exec bash "${QUEUE}"

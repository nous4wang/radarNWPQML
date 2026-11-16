#!/usr/bin/env bash
# Verify the uploaded fresh-seed queue scripts on the server, read-only.
#
# Usage (from the workstation):
#   cd R:/1radarNWPQML && python tools/remote.py \
#       --script scripts/check_newseed_upload.sh
#
# Why a separate script: run over SSH as a quoted command, a for-loop's "$f"
# is expanded by the *workstation* shell. Feeding the checks through bash -s
# keeps the expansion on the server.

set -euo pipefail

S=/home/wf/radarNWPQML/scripts
P=/home/wf/.conda/envs/radar_nwp/bin/python
ARCHIVE=/home/wf/radarNWPQML/runs/seeds10/evals

echo "=== shell syntax ==="
for f in run_seed10_queue.sh seed10_status.sh run_newseed_queue.sh newseed_status.sh; do
    bash -n "${S}/${f}"
    echo "  OK  ${f}"
done

echo "=== python syntax ==="
for f in summarize_seed10.py verify_newseed_independence.py; do
    "${P}" -m py_compile "${S}/${f}"
    echo "  OK  ${f}"
done

echo "=== CRLF (expect no output) ==="
for f in run_seed10_queue.sh seed10_status.sh run_newseed_queue.sh newseed_status.sh; do
    if LC_ALL=C grep -q $'\r' "${S}/${f}"; then
        echo "  !! CRLF in ${f}"
        exit 1
    fi
done
echo "  OK  all shell scripts are LF"

echo "=== regression: archived 0..9 batch with default expectation ==="
"${P}" "${S}/summarize_seed10.py" --evals-dir "${ARCHIVE}"

echo "=== regression: same dir judged against seeds 10..19 (must be incomplete) ==="
"${P}" "${S}/summarize_seed10.py" --evals-dir "${ARCHIVE}" \
    --expected-seeds 10 11 12 13 14 15 16 17 18 19

echo "=== launcher defaults (dry read, no jobs started) ==="
grep -nE '^(export SEEDS|export OUT_DIR|SUMMARISER_ARGS|exec bash)' \
    "${S}/run_newseed_queue.sh"
grep -nE '^(SUMMARISER_ARGS|    \$\{SUMMARISER_ARGS\}|        \$\{SUMMARISER_ARGS\})' \
    "${S}/run_seed10_queue.sh"

echo "=== archived batch untouched ==="
ls -la /home/wf/radarNWPQML/runs/seeds10/summary.json
echo "CHECK_OK"

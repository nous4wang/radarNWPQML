#!/usr/bin/env bash
# Read-only progress report for the fresh-seed (10..19) queue.
#
# Usage (server side):
#   bash /home/wf/radarNWPQML/scripts/newseed_status.sh
# From the workstation:
#   cd R:/1radarNWPQML && python tools/remote.py \
#       "bash /home/wf/radarNWPQML/scripts/newseed_status.sh"

set -uo pipefail

export OUT_DIR="${OUT_DIR:-/home/wf/radarNWPQML/runs/seeds_10_19}"
export QUEUE_LOG="${QUEUE_LOG:-/home/wf/radarNWPQML/runs/seeds_10_19_queue.log}"
export SEEDS="${SEEDS:-10 11 12 13 14 15 16 17 18 19}"
exec bash /home/wf/radarNWPQML/scripts/seed10_status.sh

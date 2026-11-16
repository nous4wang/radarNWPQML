#!/usr/bin/env bash
# Detached entry point for the full 25-model sweep.
#
#   MSYS_NO_PATHCONV=1 python tools/remote.py \
#       --script scripts/launch_full_sweep.sh \
#       --detached /home/wf/radarNWPQML/runs/full_sweep_20261108_queue.log
#
# Every value that affects the measurement is written here AND copied into
# OUT_DIR/sweep_config.txt by the main script, so a reader never has to trust
# the launcher's command line.

set -uo pipefail

export PYTHON=/home/wf/.conda/envs/radar_nwp/bin/python
export REPO_ROOT=/home/wf/radarNWP
export DATA_ROOT=/home/wf/data/radarNWP/BJ_mmap
export OUT_DIR=/home/wf/radarNWPQML/runs/full_sweep_20261108
export SUMMARISER=/home/wf/radarNWPQML/scripts/summarize_full_sweep.py

export EPOCHS=10
export MAX_STEPS=0
export EVAL_MAX_BATCHES=0
export PARALLEL=8
export BATCH_SIZE=16
export WORKERS=6
export PATCH_SIZE=64
export SLIDE_STRIDE=64
export LR=2e-4
export SEED=0

# The queue log's parent and every output directory must exist before the
# launcher redirects into the log; a missing parent makes redirection fail
# while the launcher still prints LAUNCHED.
mkdir -p /home/wf/radarNWPQML/runs \
         "${OUT_DIR}/train" "${OUT_DIR}/evals" "${OUT_DIR}/logs"

exec bash /home/wf/radarNWPQML/scripts/run_full_model_sweep.sh

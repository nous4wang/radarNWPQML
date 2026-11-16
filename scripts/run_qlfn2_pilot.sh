#!/usr/bin/env bash
# Submit the qlfn2 pilot to Slurm.
#
# The platform's rules forbid running heavy compute outside Slurm: the GPU
# is not visible from a login session at all (the device cgroup of the
# login slice refuses /dev/nvidiactl with EPERM, so ``cuInit`` returns 100
# there), so a background process would have no GPU to run on.
#
# This script therefore does what the rules say it should: it submits the
# job with ``sbatch`` and returns.  The job's own log lives under
# ``run_logs/slurm/``.
#
# What the job does
# -----------------
# 1. the qlfn2 self-checks, on the cluster's own interpreter;
# 2. a short training run through the **benchmark's own** ``scripts/train.py``,
#    with qlfn2 registered as a baseline -- so the loss, the split, the
#    interpolation policy and the checkpoint layout are the benchmark's;
# 3. a scoring run through the **benchmark's own** ``scripts/evaluate.py``,
#    with the same sliding-window stitching and the same verifier.
#
# The point is not a score.  The point is that the number the benchmark
# would report for qlfn2 comes out of the benchmark's own harness, with no
# special case for qlfn2 anywhere in it.
#
# Usage:
#     bash scripts/run_qlfn2_pilot.sh
#     squeue -u "$USER"
#     sacct -u "$USER" -S today --format=JobID,JobName%24,State%18,Elapsed,ExitCode

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REPO_ROOT}/run_logs/slurm}"

cd "${REPO_ROOT}"

# Slurm creates the log files itself, so the directory has to exist before
# the submission.
mkdir -p "${SLURM_LOG_DIR}"

sbatch --export=ALL,REPO_ROOT="${REPO_ROOT}" \
       "${REPO_ROOT}/scripts/sbatch_qlfn2_pilot.sbatch"

echo
echo "--- queue ---"
squeue -u "${USER}"

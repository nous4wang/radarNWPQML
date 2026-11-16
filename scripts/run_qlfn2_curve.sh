#!/usr/bin/env bash
# Submit the four-arm epoch-curve experiment, one independent job per arm.
#
# The jobs are submitted one at a time rather than as an array, because the
# platform counts an array's sub-tasks separately against the per-user
# limits ("作业数组按子任务计数，不能借数组绕过").  Four independent jobs
# also mean a failure in one arm cannot take the other three with it.
#
# The order is by arm, and the seeds are interleaved inside the arm: the
# arms are the thing being compared, so they should be on the cluster at the
# same time, and two seeds of the same arm should not be on it at the same
# time (they would compete for the same node's cache).
#
# Every job runs the upstream protocol:
#
#   * patch 64 / stride 64 -- the geometry the oracle ceiling and the
#     persistence floor were measured with;
#   * every epoch kept, every epoch scored, the number reported is the peak.
#
# The job script is scripts/sbatch_qlfn2_curve.sbatch.  ARM and SEED are
# passed through the environment.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
cd "${REPO_ROOT}"

EPOCHS="${EPOCHS:-10}"
SEEDS="${SEEDS:-10 11}"
ARMS="${ARMS:-0 1 2 3}"

mkdir -p run_logs/slurm

echo "=== submitting the epoch-curve experiment ==="
echo "    arms: ${ARMS}"
echo "    seeds: ${SEEDS}"
echo "    epochs: ${EPOCHS}"
echo

for ARM in ${ARMS}; do
    for SEED in ${SEEDS}; do
        echo "--- arm ${ARM} seed ${SEED} ---"
        ARM="${ARM}" SEED="${SEED}" EPOCHS="${EPOCHS}" \
            sbatch --export=ALL,ARM="${ARM}",SEED="${SEED}",EPOCHS="${EPOCHS}" \
            scripts/sbatch_qlfn2_curve.sbatch
        sleep 2
    done
done

echo
echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.18j %.9T %.9M %.6D %R"

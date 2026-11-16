#!/usr/bin/env bash
# Submit the qlfn2 experiment queue to Slurm.
#
# The platform's rules cap the account at four running jobs, six GPUs in
# total and twelve submitted jobs, and they say plainly that an array counts
# as its subtasks and cannot be used to get around the cap.  The queue is
# therefore submitted as individual jobs, eight of them, which leaves four
# slots of the twelve in reserve for a re-run of a failed arm.
#
# The submission order is interleaved by seed: the four arms of seed 10 go
# first and the four arms of seed 11 second.  Slurm starts at most four
# jobs at a time, so the first wave is one seed's four arms and the second
# wave is the other seed's four arms.  That keeps the arms of a seed in the
# same wave, which matters because the arms are compared to each other and
# a wave is a set of jobs that share the machine.
#
# Usage (from the repository root on the remote):
#     bash scripts/run_qlfn2_sweep.sh pilot
#     bash scripts/run_qlfn2_sweep.sh sweep
#
# "pilot" submits the single seam check; "sweep" submits the arms.
# The script prints the job ids so that ``squeue``/``sacct`` can be used to
# follow them.

set -euo pipefail

MODE="${1:-pilot}"
REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REPO_ROOT}/run_logs/slurm}"

cd "${REPO_ROOT}"

# Slurm writes the log files itself, so the directory has to exist before
# the submission -- the rules say so explicitly, and the job would
# otherwise fail to start.
mkdir -p "${SLURM_LOG_DIR}"

case "${MODE}" in
    pilot)
        echo "--- submitting the pilot ---"
        sbatch --export=ALL,REPO_ROOT="${REPO_ROOT}" \
               "${REPO_ROOT}/scripts/sbatch_qlfn2_pilot.sbatch"
        ;;
    sweep)
        # Two seeds per arm.  The seeds start at 10 because 0-9 were already
        # used by the earlier R1 comparison on this archive; reusing them
        # would make the two experiments share the random draws.
        N_SEEDS="${N_SEEDS:-2}"
        SEED_BASE="${SEED_BASE:-10}"
        echo "--- submitting the sweep (4 arms x ${N_SEEDS} seeds from ${SEED_BASE}) ---"
        for ((seed_index = 0; seed_index < N_SEEDS; seed_index++)); do
            seed=$((SEED_BASE + seed_index))
            for arm in 0 1 2 3; do
                sbatch --export=ALL,REPO_ROOT="${REPO_ROOT}",ARM="${arm}",SEED="${seed}" \
                       --job-name="qlfn2-arm${arm}s${seed}" \
                       "${REPO_ROOT}/scripts/sbatch_qlfn2_exp.sbatch"
                sleep 1
            done
        done
        ;;
    *)
        echo "usage: $0 {pilot|sweep}" >&2
        exit 2
        ;;
esac

echo
echo "--- queue ---"
squeue -u "${USER}"

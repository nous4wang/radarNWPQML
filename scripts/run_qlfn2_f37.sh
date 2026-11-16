#!/usr/bin/env bash
# Submit F37: 4 arms x 2 seeds = 8 independent Slurm jobs.
#
# Independent jobs (not a job array) because the platform counts array
# sub-tasks against the per-user limits, and because a failure in one arm
# must not take the other arms down with it.
#
# Usage (on the REMOTE host, from /home/wf/radarNWPQML):
#   bash scripts/run_qlfn2_f37.sh
#   ARMS="evol_both vel_only src_only no_nwp" SEEDS="10 11" bash scripts/run_qlfn2_f37.sh
#
# This is a Slurm submission only.  No training/evaluation runs in the
# login session (the login slice cannot see the GPU at all).
#
# 2026-11-14 lesson (do not repeat the 312 mistake):
#   14:23  first F36 submission accepted only job 312, then slurmctld went DOWN.
#   18:54  admin restored slurmctld (+ 30-accounting-order.conf).
#   18:55  job 312 was CANCELLED during the controller restart; its curve had
#          only epoch_00..08 and NO summary, so it could NOT be judged.
#   23:15  full re-submission of all 8 jobs after archiving the aborted run.
# "A job id exists" != "a usable result exists".
#
# Resume rule (re-check before EVERY re-submission):
#   1. scontrol ping                      # must report "is UP"
#   2. sacct -j <job> -o JobID,JobName,State,ExitCode
#      - COMPLETED with 10 scored epochs + curve summary -> SKIP_LIST="<arm>:<seed>"
#      - CANCELLED / FAILED / incomplete curve           -> RE-RUN that arm+seed
#   3. archive the aborted run directory before re-running that arm+seed
#   4. check "SUBMITTED ok=N failed=0" and squeue
# NEVER bypass Slurm with nohup / a background process.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/wf/radarNWPQML}"
cd "${REPO_ROOT}"

EPOCHS="${EPOCHS:-10}"
SEEDS="${SEEDS:-10 11}"
ARMS="${ARMS:-evol_both vel_only src_only no_nwp}"

# #SBATCH --output does not expand variables; create the directory first.
mkdir -p run_logs/slurm runs/f37_nwp

# SKIP_LIST lets a partial submission be resumed without duplicating jobs.
# Format is space-separated "<arm>:<seed>" pairs,
# e.g. SKIP_LIST="evol_both:10" when that job was already accepted.
SKIP_LIST="${SKIP_LIST:-}"

# Check the controller BEFORE trying anything, and never report "submitted"
# when sbatch actually failed.
PING="$(scontrol ping 2>&1 || true)"
echo "slurmctld  : ${PING}"
if [[ "${PING}" != *"is UP"* ]]; then
    echo "FATAL: slurmctld is not UP. No jobs submitted; retry when the "
    echo "       controller is back. (Never bypass Slurm with a background job.)"
    exit 9
fi

echo "=== F37 submission ==="
echo "    arms   : ${ARMS}"
echo "    seeds  : ${SEEDS}"
echo "    epochs : ${EPOCHS}"
echo "    out    : runs/f37_nwp/"
echo "    skip   : ${SKIP_LIST:-<none>}"
echo

ok=0
fail=0
for ARM in ${ARMS}; do
    for SEED in ${SEEDS}; do
        key="${ARM}:${SEED}"
        if [[ " ${SKIP_LIST} " == *" ${key} "* ]]; then
            echo "--- arm ${ARM} seed ${SEED} SKIPPED (already submitted) ---"
            continue
        fi
        echo "--- arm ${ARM} seed ${SEED} ---"
        out="$(ARM="${ARM}" SEED="${SEED}" EPOCHS="${EPOCHS}" \
            sbatch --export=ALL,ARM="${ARM}",SEED="${SEED}",EPOCHS="${EPOCHS}" \
                scripts/sbatch_qlfn2_f37.sbatch 2>&1)"
        rc=$?
        echo "${out}"
        if [[ ${rc} -eq 0 ]]; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
        sleep 2
    done
done

echo
echo "=== queue ==="
squeue -u "$USER" -o "%.10i %.18j %.9T %.9M %.6D %R" || true
echo
echo "=== logs ==="
echo "    run_logs/slurm/qlfn2-f37-<jobid>.out"
echo
echo "=== SUBMITTED ok=${ok} failed=${fail} ==="
if [[ ${fail} -ne 0 ]]; then
    echo "Some jobs were NOT submitted. Re-run with SKIP_LIST listing the "
    echo "jobs that were accepted, once the controller is UP."
    exit 1
fi

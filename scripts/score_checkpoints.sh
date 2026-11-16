#!/usr/bin/env bash
# Score a set of checkpoints under the benchmark's own protocol.
#
# The benchmark keeps a copy of every epoch so the score can be read off a
# known epoch rather than off a moment.  The peak of these models is not the
# last epoch -- the last checkpoint is the collapsed end -- so a run that
# scores only ``last.pt`` reports the worst checkpoint of the run and calls
# it the model.
#
# Usage: bash scripts/score_checkpoints.sh <model> <split> <ckpt> [<ckpt> ...]
set -eo pipefail

MODEL="$1"
SPLIT="$2"
shift 2

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate radar_nwp
set -u

BENCH_ROOT="${BENCH_ROOT:-/home/wf/radarNWP}"
OUT_DIR="${OUT_DIR:-/tmp/score_ckpts}"
mkdir -p "${OUT_DIR}"

for CKPT in "$@"; do
    NAME="$(basename "${CKPT}" .pt)"
    JSON="${OUT_DIR}/${MODEL}_${NAME}.json"
    echo "=== ${MODEL} / ${NAME} -> ${JSON}"
    python -u "${BENCH_ROOT}/scripts/evaluate.py" \
        --model "${MODEL}" \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT:-/home/wf/data/radarNWP/BJ_mmap}" \
        --packed \
        --split "${SPLIT}" \
        --device cuda \
        --out-json "${JSON}" || echo "   evaluate rc=$?"
    if [[ -f "${JSON}" ]]; then
        python "${REPO_ROOT:-/home/wf/radarNWPQML}/tools/check_eval_report.py" "${JSON}" || true
    else
        echo "   FATAL: no report at ${JSON}" >&2
    fi
done

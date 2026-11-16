#!/usr/bin/env bash
# Post-run analysis for the fresh-seed (10..19) two-arm queue.
#
# Usage (server side):
#   bash /home/wf/radarNWPQML/scripts/report_newseed.sh
# From the workstation:
#   cd R:/1radarNWPQML && python tools/remote.py \
#       "bash /home/wf/radarNWPQML/scripts/report_newseed.sh"
#
# Refuses to report results before the queue's terminal marker appears. The
# only files it writes are derived JSON reports inside the new batch directory.

set -uo pipefail

S=/home/wf/radarNWPQML/scripts
P=/home/wf/.conda/envs/radar_nwp/bin/python
NEW=/home/wf/radarNWPQML/runs/seeds_10_19
OLD=/home/wf/radarNWPQML/runs/seeds10
LOG=/home/wf/radarNWPQML/runs/seeds_10_19_queue.log

bash "${S}/newseed_status.sh"

if ! grep -q QUEUE_DONE "${LOG}" 2>/dev/null; then
    echo "QUEUE NOT DONE - no result analysis reported; the status above is current."
    exit 0
fi

echo "=== independence of the fresh-seed draw ==="
"${P}" "${S}/verify_newseed_independence.py" \
    --new-root "${NEW}" --old-root "${OLD}" \
    --out "${NEW}/independence.json"

echo "=== pass-rate sensitivity on the fresh seed set ==="
"${P}" "${S}/analyze_seed_sensitivity.py" \
    --summary "${NEW}/summary.json" --out "${NEW}/sensitivity.json"

echo "=== fresh batch vs the 0..9 anchor ==="
"${P}" - "${NEW}/summary.json" "${OLD}/summary.json" \
    "${NEW}/sensitivity.json" "${NEW}/independence.json" <<'PY'
import json
import sys

paths = sys.argv[1:5]
new, old, sens, indep = (json.load(open(p, encoding="utf-8")) for p in paths)
for tag, rep in (("seeds 0..9 (archived)", old),
                 ("seeds 10..19 (fresh)", new)):
    print(tag)
    # The archived 0..9 summary predates the --expected-seeds field.
    print("  expected seeds :",
          rep["protocol"].get("expected_seeds", list(range(10))))
    for arm in ("qpe_0", "base_concat"):
        row = rep["arms"][arm]
        print("  %-12s %2d/%-2d  min=%.6f  max=%.6f  peak_end=%s" % (
            arm, row["pass_count"], row["n_seeds"],
            row["min_csi2"], row["max_csi2"], row["peak_end_ratio"]))
    print("  fisher p       :", rep["fisher_one_sided_p"])
    print("  batch_complete :", rep["batch_complete"])
print("independent draw :", indep["is_independent_draw"],
      indep["checks"])
gap = sens["pass_rate_difference"]
if gap is not None:
    print("resampled gap 95%  :", gap["ci95"],
          "P(diff<=0) =", gap["p_diff_zero_or_negative"])
print("supports_improvement_claim :",
      new["verdict"]["supports_improvement_claim"])
PY

echo "REPORT_DONE"

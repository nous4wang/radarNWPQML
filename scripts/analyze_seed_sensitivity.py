#!/usr/bin/env python3
"""Quantify how little a fixed-seed pass rate can discriminate.

The original seeds 0..9 queue turned out to be a byte-exact deterministic
replay of the archived runs (see compare_seed10_weights.py). A fixed seed
therefore maps to one fixed CSI@2.0, and any chosen seed set is a set of
*chosen* deterministic configurations, not draws from a distribution. This
script treats those values as an empirical pool and asks how much the pass
rate moves under leave-one-out and bootstrap resampling.

The replay finding belongs to the 0..9 batch only. This script is also used
for the fresh seeds 10..19 batch, so nothing in its verdict assumes a replay;
independence of the draw is checked separately by
verify_newseed_independence.py.

This is deliberately NOT presented as a confidence interval for a true pass
rate: with deterministic replay there is no sampling distribution to estimate.
It is a sensitivity diagnostic showing that the observed 5/10 vs 4/10 gap is
smaller than the variation induced by merely re-picking the seed set.

No third-party dependency. Bootstrap uses a fixed RNG seed for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

FLOOR_CSI2 = 0.0924
HEADROOM = 0.0729
N_BOOTSTRAP = 10_000
RNG_SEED = 20261108


def pass_rate(values: list[float]) -> float:
    """Fraction of CSI@2.0 values strictly above the persistence floor."""
    return sum(v > FLOOR_CSI2 for v in values) / len(values) if values else 0.0


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def analyse(scores: dict[str, list[float]]) -> dict[str, Any]:
    """Build the sensitivity report for two arms."""
    rng = random.Random(RNG_SEED)
    arms: dict[str, Any] = {}
    for arm, values in scores.items():
        loo = [pass_rate(values[:i] + values[i + 1:]) for i in range(len(values))]
        boot = [pass_rate([values[rng.randrange(len(values))] for _ in values])
                for _ in range(N_BOOTSTRAP)]
        boot_sorted = sorted(boot)
        arms[arm] = {
            "n_seeds": len(values),
            "observed_pass_rate": pass_rate(values),
            "observed_pass_count": sum(v > FLOOR_CSI2 for v in values),
            "min_csi2": min(values),
            "max_csi2": max(values),
            "single_seed_range": round(max(values) - min(values), 6),
            "leave_one_out_pass_rate": {
                "min": round(min(loo), 4),
                "max": round(max(loo), 4),
                "distinct_values": sorted({round(x, 4) for x in loo}),
            },
            "resampled_pass_rate": {
                "mean": round(sum(boot) / len(boot), 4),
                "ci95": [round(quantile(boot_sorted, 0.025), 4),
                         round(quantile(boot_sorted, 0.975), 4)],
                "min": round(boot_sorted[0], 4),
                "max": round(boot_sorted[-1], 4),
            },
        }

    names = list(scores)
    if len(names) == 2:
        a, b = scores[names[0]], scores[names[1]]
        observed_diff = pass_rate(a) - pass_rate(b)
        diffs = [
            pass_rate([a[rng.randrange(len(a))] for _ in a])
            - pass_rate([b[rng.randrange(len(b))] for _ in b])
            for _ in range(N_BOOTSTRAP)
        ]
        diffs_sorted = sorted(diffs)
        difference = {
            "observed": round(observed_diff, 4),
            "resampled_mean": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(quantile(diffs_sorted, 0.025), 4),
                     round(quantile(diffs_sorted, 0.975), 4)],
            "p_diff_positive": round(sum(d > 0 for d in diffs) / len(diffs), 4),
            "p_diff_zero_or_negative": round(
                sum(d <= 0 for d in diffs) / len(diffs), 4
            ),
        }
    else:
        difference = None

    widest = max(
        (row["max_csi2"] - row["min_csi2"]) for row in arms.values()
    )
    n_seeds = len(next(iter(scores.values()))) if scores else 0
    ci = difference["ci95"] if difference is not None else None
    straddles_zero = ci is not None and ci[0] <= 0 <= ci[1]
    return {
        "source": "ep00 CSI@2.0 from a two-arm seed batch (path set by --summary)",
        "floor_csi2": FLOOR_CSI2,
        "oracle_headroom": HEADROOM,
        "n_seeds_per_arm": n_seeds,
        "widest_single_seed_range": round(widest, 6),
        "range_over_headroom": round(widest / HEADROOM, 4),
        "bootstrap_draws": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "arms": arms,
        "pass_rate_difference": difference,
        "caveat": (
            "A fixed seed is byte-exact deterministic in this pipeline, so "
            "these values are chosen deterministic configurations, not random "
            "draws. The resampled intervals describe sensitivity to the seed "
            "set, NOT a confidence interval for an underlying pass rate. "
            "Whether a batch is a replay of an earlier one is a separate "
            "question (see verify_newseed_independence.py)."
        ),
        "verdict": {
            "supports_improvement_claim": False,
            "reason": (
                f"A fixed seed maps to one deterministic score, so a "
                f"{n_seeds}-seed pass rate is a function of the chosen seed "
                "indices rather than a stable estimate of model skill. "
                f"Re-picking seeds spreads the arm difference over {ci}"
                + (" (includes zero)" if straddles_zero else "")
                + ". Two baseline arms alone cannot clear P2/P4, whatever "
                "the sign of that difference."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path,
                        help="seed10_summary.json produced by summarize_seed10.py")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    scores = {
        arm: [float(v) for v in row["csi2_by_seed"].values()]
        for arm, row in payload["arms"].items()
    }
    sizes = {len(v) for v in scores.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        parser.error("every arm needs the same non-zero number of seed scores")
    report = analyse(scores)
    report["source"] = str(args.summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for arm, row in report["arms"].items():
        print(f"{arm:12s} observed {row['observed_pass_count']}/{row['n_seeds']}"
              f"  leave-one-out {row['leave_one_out_pass_rate']['min']}"
              f"..{row['leave_one_out_pass_rate']['max']}"
              f"  resampled 95% {row['resampled_pass_rate']['ci95']}")
    diff = report["pass_rate_difference"]
    if diff is not None:
        print(f"difference   observed {diff['observed']}"
              f"  resampled 95% {diff['ci95']}"
              f"  P(diff<=0) {diff['p_diff_zero_or_negative']}")
    print(f"widest range {report['widest_single_seed_range']}"
          f"  = {report['range_over_headroom']}x headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

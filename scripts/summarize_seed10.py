#!/usr/bin/env python3
"""Summarise a pass-rate batch under MASTER.md 11.9 (P1-P6).

Reads ``*.eval.json`` reports named ``<arm>_s<seed>_ep<epoch>.eval.json``
and reports the ep00 CSI@2.0 pass rate per arm, the one-sided Fisher p, and
the peak-end dispersion. It is a *diagnostic* summariser: two baseline arms
alone cannot support an improvement claim (P2/P4), so the verdict field says
so explicitly.

No third-party dependency. Runs on the workstation as well as the server,
which lets the historical 96-report archive be used as a self-test.

The expected seed set is an argument, not a constant. A batch using fresh
seeds 10..19 must not be judged against the historical 0..9 completeness
check, or ``batch_complete`` would be false for a perfectly complete batch.
The default stays 0..9 so the archived reports keep their old behaviour.

Reference constants (docs/MASTER.md 11.9, qmlbench/results.py):
  FLOOR_CSI2      = 0.0924  persistence floor
  ORACLE_CSI2     = 0.1653  perfect-QPE + no-motion upper bound (E15)
  HEADROOM        = 0.0729  ORACLE - FLOOR
  SEED_DISPERSION = 0.1671  single-seed range in the historical batch
Historical anchors: qpe_0 5/10, base_concat 4/10, Fisher one-sided p=0.5,
peak-end dispersion 9.59x over [0.0194, 0.1865].
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

FLOOR_CSI2 = 0.0924
ORACLE_CSI2 = 0.1653
HEADROOM = round(ORACLE_CSI2 - FLOOR_CSI2, 4)
SEED_DISPERSION = 0.1671

ANCHOR_PASS = {"qpe_0": 5, "base_concat": 4}
ANCHOR_PEAK_END_RATIO = 9.59
ANCHOR_FISHER_ONE_SIDED_P = 0.5
ARMS = ("qpe_0", "base_concat")
EPOCH = "00"
DEFAULT_SEEDS = tuple(range(10))

_PATTERN = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_ep(?P<epoch>\d{2})\.eval\.json$")


def fisher_one_sided(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p for ``arm1 pass rate > arm2 pass rate``.

    Uses the hypergeometric tail over 2x2 tables with the same margins.
    ``math.comb`` keeps this dependency-free; n <= 10 per arm here.
    """
    total = a + b
    population = n1 + n2
    denominator = math.comb(population, total)
    high = min(n1, total)
    numerator = sum(
        math.comb(n1, k) * math.comb(n2, total - k)
        for k in range(a, high + 1)
    )
    return min(1.0, numerator / denominator) if denominator else 1.0


def read_scores(evals_dir: Path) -> dict[str, dict[int, float]]:
    """Return {arm: {seed: csi@2.0}} for ep00 reports only."""
    out: dict[str, dict[int, float]] = {}
    for path in sorted(evals_dir.glob("*.eval.json")):
        match = _PATTERN.match(path.name)
        if not match or match["epoch"] != EPOCH:
            continue
        arm, seed = match["arm"], int(match["seed"])
        if arm not in ARMS:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            value = report["thresholds"]["2.0"]["csi"]
            score = float(value)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not math.isfinite(score):
            continue
        out.setdefault(arm, {})[seed] = score
    return out


def arm_summary(scores: dict[int, float], anchor_pass: int,
                expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    """Summarise one arm; missing seeds are listed, never filled in."""
    seeds = sorted(scores)
    values = [scores[s] for s in seeds]
    passed = [s for s in seeds if scores[s] > FLOOR_CSI2]
    low, high = (min(values), max(values)) if values else (None, None)
    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "missing_seeds": sorted(set(expected_seeds) - set(seeds)),
        "unexpected_seeds": sorted(set(seeds) - set(expected_seeds)),
        "csi2_by_seed": {str(s): scores[s] for s in seeds},
        "epoch": EPOCH,
        "pass_count": len(passed),
        "pass_rate": round(len(passed) / len(seeds), 4) if seeds else None,
        "passing_seeds": passed,
        "min_csi2": low,
        "max_csi2": high,
        "peak_end_ratio": round(high / low, 4) if low and high else None,
        "anchor_pass_count": anchor_pass,
    }


def summarise(evals_dir: Path,
              expected_seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    """Build the whole diagnostic report from a directory of eval reports."""
    n = len(expected_seeds)
    scores = read_scores(evals_dir)
    arms = {
        arm: arm_summary(scores.get(arm, {}), ANCHOR_PASS[arm], expected_seeds)
        for arm in ARMS
    }
    q, b = arms["qpe_0"], arms["base_concat"]
    complete = all(
        row["n_seeds"] == n
        and not row["missing_seeds"]
        and not row["unexpected_seeds"]
        for row in arms.values()
    )
    fisher_p = (
        fisher_one_sided(q["pass_count"], n, b["pass_count"], n)
        if complete else None
    )
    observed = [
        row["peak_end_ratio"] for row in arms.values()
        if row["peak_end_ratio"] is not None
    ]
    return {
        "evals_dir": str(evals_dir),
        "protocol": {
            "epoch": "ep00 (MASTER 11.9 P3)",
            "metric": "CSI@2.0 on the clean test split, whole 301x301 grid",
            "expected_seeds": list(expected_seeds),
            "floor_csi2": FLOOR_CSI2,
            "oracle_csi2": ORACLE_CSI2,
            "headroom": HEADROOM,
            "historical_single_seed_range": SEED_DISPERSION,
            "range_over_headroom": round(SEED_DISPERSION / HEADROOM, 4),
        },
        "arms": arms,
        "fisher_one_sided_p": fisher_p,
        "max_peak_end_ratio": max(observed) if observed else None,
        "historical_anchor": {
            "pass_counts": ANCHOR_PASS,
            "fisher_one_sided_p": ANCHOR_FISHER_ONE_SIDED_P,
            "peak_end_ratio": ANCHOR_PEAK_END_RATIO,
            "score_range": [0.0194, 0.1865],
            "source": "docs/MASTER.md 11.9; tools/_evals/route_seeds/evals (96 reports)",
        },
        "batch_complete": complete,
        "verdict": {
            "supports_improvement_claim": False,
            "reason": (
                "Only the two baseline arms are present. P2 uses them as a "
                "same-batch anchor, P4 additionally requires paired per-seed "
                "differences and a two-level paired bootstrap against a "
                "candidate arm. A baseline-only batch cannot clear the gate."
            ),
            "usable_as_p2_anchor": complete,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int,
                        default=list(DEFAULT_SEEDS),
                        help="seed ids this batch is required to contain "
                             "(default: 0..9, the archived anchor)")
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON output path")
    args = parser.parse_args()
    if not args.evals_dir.is_dir():
        parser.error(f"not a directory: {args.evals_dir}")
    expected = tuple(args.expected_seeds)
    if len(set(expected)) != len(expected):
        parser.error("--expected-seeds contains duplicates")
    report = summarise(args.evals_dir, expected)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for arm, row in report["arms"].items():
        print(
            f"{arm:12s} ep00  n={row['n_seeds']:2d}  "
            f"pass={row['pass_count']:2d}/{row['n_seeds']:2d}  "
            f"range=({row['min_csi2']}, {row['max_csi2']})  "
            f"peak_end_ratio={row['peak_end_ratio']}"
        )
    print(f"expected seeds     = {list(expected)}")
    print(f"fisher one-sided p = {report['fisher_one_sided_p']}")
    print(f"batch_complete     = {report['batch_complete']}")
    print("supports_improvement_claim = False (baseline arms only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

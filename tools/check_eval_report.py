#!/usr/bin/env python3
"""Check that an evaluation report carries a real score.

The benchmark's verifier reports "0 samples" when the sliding walk leaves a
gap in the grid: nothing is scored, every score is NaN, the job still
completes, and the JSON file still looks like a result.  That is a silent
failure, so a report is checked explicitly before it is read as a number.

What is and is not checked
--------------------------
The upstream verifier computes the CSI as ``hits / (hits + misses +
false_alarms)`` and returns NaN when that denominator is zero, that is, when
a threshold has **no events at all**.  The archive's highest thresholds (8 and
16 mm per 6-minute accumulation, i.e. 80 and 160 mm/h) routinely have no
events, so a NaN there is a property of the data and not a failure.

A NaN at a threshold the benchmark's tables report is a different matter: it
means the walk did not tile the grid and nothing was scored.  So the check is

* ``n_samples`` must be positive;
* every threshold at or below ``--core-max`` (2.0 mm, the short-duration
  heavy-rain line the tables are built around) must carry a finite CSI;
* at least one threshold anywhere must carry a finite CSI.

Exit codes
----------
0   the report carries a score
4   nothing was scored, or a reported threshold has a NaN CSI
5   the report does not exist
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

#: The threshold the benchmark's tables are built around: 2.0 mm per
#: 6-minute accumulation is 20 mm/h, the short-duration heavy-rain line.
CORE_MAX = 2.0


def _fmt(value: Any) -> str:
    """Format one number for the table, keeping NaN visible."""
    if value is None:
        return "    nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "     n/a"
    if math.isnan(number):
        return "    nan"
    return f"{number:7.4f}"


def main() -> int:
    """Check the report and print its score table.

    Returns:
        0 when the report carries a score, 4 when it does not, 5 when the
        report is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="the evaluation JSON to check")
    parser.add_argument(
        "--core-max",
        type=float,
        default=CORE_MAX,
        help="the highest threshold that must carry a finite CSI "
             "(default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError:
        print(f"FATAL: no report at {args.report}", file=sys.stderr)
        return 5
    except json.JSONDecodeError as error:
        print(f"FATAL: {args.report} is not valid JSON: {error}", file=sys.stderr)
        return 5

    n_samples = report.get("n_samples")
    thresholds = report.get("thresholds") or report.get("per_threshold") or {}

    print(f"report          : {args.report}")
    print(f"n_samples       : {n_samples}")
    print(f"thresholds      : {len(thresholds)}")
    print()
    print("  thr |     CSI      POD      FAR     BIAS      FSS")
    print("-------+-----------------------------------------------")

    finite_any = False
    bad_core: list[str] = []
    for name, entry in sorted(thresholds.items(), key=lambda kv: float(kv[0])):
        if not isinstance(entry, dict):
            continue
        try:
            threshold = float(name)
        except (TypeError, ValueError):
            continue
        csi = entry.get("csi")
        if isinstance(csi, (int, float)) and not math.isnan(float(csi)):
            finite_any = True
        elif threshold <= args.core_max:
            bad_core.append(name)
        print(
            f"{threshold:5.1f} | {_fmt(csi)} {_fmt(entry.get('pod'))} "
            f"{_fmt(entry.get('far'))} {_fmt(entry.get('bias'))} "
            f"{_fmt(entry.get('fss'))}"
        )

    print()

    if not isinstance(n_samples, int) or n_samples <= 0:
        print(
            "FATAL: the report scored no samples -- the sliding walk did not "
            "tile the grid",
            file=sys.stderr,
        )
        return 4

    if bad_core:
        print(
            f"FATAL: the CSI is NaN at thresholds {bad_core} -- nothing was "
            "scored at a threshold the tables report",
            file=sys.stderr,
        )
        return 4

    if not finite_any:
        print(
            "FATAL: no threshold carries a finite CSI -- nothing was scored",
            file=sys.stderr,
        )
        return 4

    print(f"report carries a finite CSI at {len(thresholds)} thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

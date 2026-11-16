"""Put the upstream oracle arms and the sweep's own arms side by side.

The benchmark's oracle probe measured a hard ceiling: taking the observed
ground rainfall at the issue time and copying it forward without motion
reaches CSI@2.0 = 0.1653, while the zero-motion radar copy (the persistence
floor) reaches 0.0924.  Both numbers come from the same verifier over the
same 538 stitched windows, so a model that scores above 0.1653 is either
doing something the oracle does not model or reading the answer.

This script prints the per-lead profile of every arm in one table, because
the total alone cannot tell those two apart: a model that reads the truth
would score high at every lead, while a model that forecasts a moving storm
decays with the lead the way the persistence floor does.

Usage:
    python tools/lead_table.py <oracle_ceiling.json> <report.json> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LEADS = [str(i) for i in range(10)]


def _dig(value: Any) -> Any:
    """Unwrap a value that may be a bare number or a metric dictionary."""
    if isinstance(value, dict):
        for key in ("csi", "CSI"):
            if key in value:
                return value[key]
    return value


def _number(value: Any) -> float | None:
    value = _dig(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.4f}"


def _oracle(path: Path) -> tuple[dict[str, list[float | None]], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles: dict[str, list[float | None]] = {}
    totals: dict[str, float] = {}
    for name, arm in payload.get("arms", {}).items():
        table = arm.get("lead_csi_2.0") or {}
        profiles[name] = [_number(table.get(lead)) for lead in LEADS]
        overall = _number((arm.get("thresholds") or {}).get("2.0", {}).get("csi"))
        if overall is not None:
            totals[name] = overall
    return profiles, totals


def _report(path: Path) -> tuple[list[float | None], float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("lead_csi") or {}
    profile = [_number((table.get(lead) or {}).get("2.0")) for lead in LEADS]
    overall = _number((payload.get("thresholds") or {}).get("2.0", {}).get("csi"))
    return profile, overall


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    oracle_path = Path(sys.argv[1])
    report_paths = [Path(p) for p in sys.argv[2:]]

    oracle_profiles, oracle_totals = _oracle(oracle_path)

    columns: list[tuple[str, list[float | None], float | None]] = []
    for name, profile in oracle_profiles.items():
        columns.append((f"oracle:{name}", profile, oracle_totals.get(name)))
    for path in report_paths:
        profile, overall = _report(path)
        label = "/".join(path.parts[-3:])
        columns.append((label, profile, overall))

    name_width = max(len(name) for name, _, _ in columns) + 2
    header = "arm".ljust(name_width) + "overall".rjust(10)
    for lead in LEADS:
        header += f"L{lead}".rjust(9)
    print(header)
    print("-" * len(header))
    for name, profile, overall in columns:
        row = name.ljust(name_width)
        row += ("n/a" if overall is None else f"{overall:.4f}").rjust(10)
        for value in profile:
            row += _fmt(value).rjust(9)
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

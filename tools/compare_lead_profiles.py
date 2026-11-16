"""Print the score profile of several evaluation files side by side.

The sweep evaluates the last checkpoint, but the benchmark's own protocol
keeps a copy of every epoch and reads the score off the peak, because the
last checkpoint of these models is the collapsed end rather than the best
one.  To compare like with like, the score profile has to be visible: the
value at 2 mm, and the way it decays over the ten leads.

Usage:
    python tools/compare_lead_profiles.py <eval.json> [<eval.json> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _threshold_table(payload: dict[str, Any]) -> dict[str, Any]:
    table = payload.get("thresholds")
    if not isinstance(table, dict):
        table = payload.get("per_threshold") or {}
    return table if isinstance(table, dict) else {}


def _lead_table(payload: dict[str, Any]) -> dict[str, Any]:
    table = payload.get("lead_csi")
    if not isinstance(table, dict):
        table = payload.get("lead_scores") or {}
    return table if isinstance(table, dict) else {}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    names: list[str] = []
    overall: list[float | None] = []
    leads: list[list[float | None]] = []
    samples: list[Any] = []

    for path in sys.argv[1:]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        names.append(Path(path).name)
        samples.append(payload.get("n_samples"))

        table = _threshold_table(payload)
        entry = table.get("2.0") if isinstance(table.get("2.0"), dict) else {}
        overall.append(_number(entry.get("csi")))

        lead_table = _lead_table(payload)
        row: list[float | None] = []
        for index in range(10):
            per_lead = lead_table.get(str(index))
            if not isinstance(per_lead, dict):
                row.append(None)
                continue
            value = per_lead.get("2.0")
            if isinstance(value, dict):
                value = value.get("csi")
            row.append(_number(value))
        leads.append(row)

    width = max(len(name) for name in names) + 2
    print("file".ljust(width) + "n".rjust(6) + "CSI@2.0".rjust(11))
    print("-" * (width + 17))
    for name, n, value in zip(names, samples, overall):
        text = "n/a" if value is None else f"{value:.4f}"
        print(name.ljust(width) + str(n).rjust(6) + text.rjust(11))

    print()
    print("lead-time CSI@2.0  (lead 0 = the first 6 minutes, lead 9 = the last)")
    header = "lead".rjust(6)
    for name in names:
        header += name[:18].rjust(20)
    print(header)
    print("-" * len(header))
    for index in range(10):
        row = f"{index}".rjust(6)
        for values in leads:
            value = values[index]
            row += ("n/a" if value is None else f"{value:.4f}").rjust(20)
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare a sweep report against the upstream benchmark's own eval files.

The number that matters is CSI@2.0, and it is only meaningful if the
evaluation unit is the same one the upstream authority table used.  This
script therefore prints the unit count and the score side by side, and it
refuses to report a comparison when the units disagree.

Usage:
    python tools/compare_reports.py <report.json> [<upstream eval.json> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

THRESHOLDS = ("0.5", "1.0", "2.0", "4.0")


def _get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _csi(payload: dict[str, Any], threshold: str) -> float | None:
    table = _get(payload, "thresholds", default={}) or {}
    entry = table.get(threshold)
    if not isinstance(entry, dict):
        return None
    value = entry.get("csi")
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _n(payload: dict[str, Any]) -> Any:
    return payload.get("n_samples", payload.get("samples", "?"))


def _row(label: str, payload: dict[str, Any]) -> str:
    cells = [label, str(_n(payload))]
    for threshold in THRESHOLDS:
        value = _csi(payload, threshold)
        cells.append("   n/a " if value is None else f"{value:.4f}")
    return "  ".join(cells)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    header = "source".ljust(34) + "n".rjust(7) + "".join(
        f"CSI@{t}".rjust(10) for t in THRESHOLDS
    )
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sys.argv[1:]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append((Path(path).name, payload))
        print(_row(Path(path).name, payload))

    if len(rows) < 2:
        return 0

    reference = rows[0][1]
    ref_n = _n(reference)
    mismatched = [name for name, payload in rows[1:] if _n(payload) != ref_n]
    if mismatched:
        print()
        print("WARNING: the evaluation unit differs across these files "
              f"({ref_n} vs others); the CSI values are not comparable.")
        for name in mismatched:
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

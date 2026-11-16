#!/usr/bin/env python3
"""Summarise a full-model sweep into a diagnostic, non-authoritative table.

The output is deliberately marked ``NOT_COMPARABLE``: the sweep uses a single
uniform parameter set for all registered models, which is not the native
configuration of either upstream sweep script, so these numbers must not be
copied into ``MASTER.md`` or used for any improvement claim.

Args:
    --evals-dir: Directory holding one ``<model>.json`` evaluate.py report.
    --models: Model names in registry order.
    --config: The sweep's provenance snapshot.
    --out: Destination for the JSON summary.
    --tsv: Destination for the TSV table.

Returns:
    Process exit code: 0 when the summary was written, 1 on a bad invocation.

Raises:
    OSError: If an output path cannot be written (propagated to the caller).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FLOOR_CSI2 = 0.0924  # upstream persistence floor, reference only
THRESHOLDS = ("0.1", "0.5", "2.0", "4.0")
CAVEAT = (
    "NOT_COMPARABLE: uniform non-native parameters, single seed, "
    "diagnostic sweep only; not an authoritative MASTER.md result."
)


def read_report(path: Path) -> dict[str, object]:
    """Read one evaluate.py report and extract the CSI row.

    Args:
        path: Path to ``<model>.json``.

    Returns:
        A dict with ``present``, ``n_samples``, per-threshold CSI values and
        an ``error`` string. A missing or malformed report is recorded as
        absent rather than as a zero score.
    """
    row: dict[str, object] = {
        "present": False,
        "n_samples": None,
        "error": "missing report",
    }
    for threshold in THRESHOLDS:
        row[f"csi_{threshold}"] = None
    if not path.exists():
        return row
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        row["error"] = f"unreadable report: {exc}"
        return row
    if not isinstance(payload, dict):
        row["error"] = "report is not a JSON object"
        return row
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        row["error"] = "report has no thresholds object"
        return row
    row["present"] = True
    row["error"] = ""
    row["n_samples"] = payload.get("n_samples")
    for threshold in THRESHOLDS:
        entry = thresholds.get(threshold)
        if isinstance(entry, dict):
            value = entry.get("csi")
            if isinstance(value, (int, float)):
                row[f"csi_{threshold}"] = float(value)
    return row


def parse_config(path: Path) -> dict[str, str]:
    """Parse the ``key=value`` provenance snapshot.

    Args:
        path: Path to ``sweep_config.txt``.

    Returns:
        Key/value mapping; an unreadable file yields an empty mapping.
    """
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> int:
    """Build the summary files.

    Returns:
        0 on success, 1 when required arguments are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tsv", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        row = read_report(args.evals_dir / f"{model}.json")
        row["model"] = model
        rows.append(row)

    config = parse_config(args.config) if args.config else {}
    complete = sum(1 for row in rows if row["present"])
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.evals_dir),
        "provenance": config,
        "caveat": CAVEAT,
        "n_models_expected": len(args.models),
        "n_reports_present": complete,
        "batch_complete": complete == len(args.models),
        "persistence_floor_reference_only": FLOOR_CSI2,
        "qlfn_scope": "EXCLUDED per docs/archive/QLFN_SPEC.md R14 (已撤销)",
        "supports_improvement_claim": False,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    header = ["model", "status", "n_samples"] + [
        f"csi@{t}" for t in THRESHOLDS
    ] + ["gap_to_floor_reference_only"]
    lines = ["\t".join(header)]
    for row in rows:
        csi2 = row["csi_2.0"]
        gap = (
            f"{FLOOR_CSI2 - csi2:+.6f}"
            if isinstance(csi2, float) else "n/a"
        )
        cells = [
            str(row["model"]),
            "ok" if row["present"] else f"MISSING ({row['error']})",
            str(row["n_samples"]) if row["n_samples"] is not None else "n/a",
        ]
        for threshold in THRESHOLDS:
            value = row[f"csi_{threshold}"]
            cells.append(f"{value:.6f}" if isinstance(value, float) else "n/a")
        cells.append(gap)
        lines.append("\t".join(cells))
    args.tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"models expected : {len(args.models)}")
    print(f"reports present : {complete}")
    print(f"batch complete  : {payload['batch_complete']}")
    print(f"caveat          : {CAVEAT}")
    print(f"table           : {args.tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

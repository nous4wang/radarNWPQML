#!/usr/bin/env python3
"""Print the structure of an evaluation report's lead-time tables.

The lead table is what says whether a mechanism helps *more* the longer the
lead is.  A forecast that carries skill only at lead one and nothing beyond
it is not doing nowcasting; a forecast whose skill grows with the lead is
doing something that a static field cannot.  So the structure has to be
known before the summary can use it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Print the report's keys and the shape of its lead tables."""
    path = Path(sys.argv[1])
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    print(f"report : {path}")
    print(f"keys   : {list(report.keys())}")
    print(f"n_samples: {report.get('n_samples')}")

    lead_scores = report.get("lead_scores") or {}
    print(f"lead_scores leads: {list(lead_scores.keys())}")
    for lead in list(lead_scores)[:2]:
        entry = lead_scores[lead]
        print(f"  lead {lead}: thresholds {list(entry.keys())}")
        for threshold in list(entry)[:2]:
            print(f"    {threshold}: {entry[threshold]}")

    lead_csi = report.get("lead_csi") or {}
    print(f"lead_csi leads: {list(lead_csi.keys())}")
    for lead in list(lead_csi)[:3]:
        print(f"  lead {lead}: {lead_csi[lead]}")

    ablation = report.get("ablation")
    if ablation is not None:
        print(f"ablation: {ablation}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

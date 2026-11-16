#!/usr/bin/env python3
"""Read the upstream's own oracle-ceiling report and print it flat.

The upstream report is nested and this project's shell quoting keeps losing
parentheses, so the reader is a file rather than an inline command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/wf/radarNWP/runs/four_tiers/oracle_ceiling.json")

report = json.loads(path.read_text(encoding="utf-8"))
print(f"file      : {path}")
print(f"top keys  : {list(report.keys())}")
for key in ("n_samples", "grid", "zr_over_ground"):
    if key in report:
        print(f"{key:10}: {report[key]}")

arms = report.get("arms", report)
print("arms:")
for name, value in arms.items():
    if not isinstance(value, dict):
        print(f"  {name:16}: {value}")
        continue
    if "csi" in value:
        csi = value["csi"]
        print(f"  {name:16}: CSI@2.0 = {csi}"
              f"  (hits={value.get('hits')} "
              f"misses={value.get('misses')} "
              f"false_alarms={value.get('false_alarms')})")
    else:
        print(f"  {name:16}: {json.dumps(value, ensure_ascii=False)[:200]}")

leads = report.get("lead_profile") or report.get("per_lead")
if leads:
    print("per lead:")
    print(json.dumps(leads, ensure_ascii=False, indent=2)[:2000])

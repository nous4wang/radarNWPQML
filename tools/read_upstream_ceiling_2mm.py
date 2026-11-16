#!/usr/bin/env python3
"""Pull the 2.0 mm column out of the upstream oracle-ceiling report.

The oracle ceiling (0.1653) and the persistence floor (0.0924) are the two
numbers this project is measured against, and they were produced by
``tools/_oracle_advection_ceiling.py`` with the upstream authority geometry
(patch 64 / stride 64).  This reads the report and prints the same column
the project reports, so the comparison is made on identical quantities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/wf/radarNWP/runs/four_tiers/oracle_ceiling.json")

report = json.loads(path.read_text(encoding="utf-8"))
arms = report.get("arms", {})

print(f"{'arm':16} {'CSI@2.0':>9} {'hits':>13} {'misses':>13} "
      f"{'false_alarms':>13} {'POD':>7} {'FAR':>7} {'BIAS':>7}")
for name, value in arms.items():
    if not isinstance(value, dict):
        continue
    thresholds = value.get("thresholds", {})
    entry = thresholds.get("2.0")
    if entry is None:
        continue
    # The counts come back as floats in this report, so they are printed as
    # integers with a cast rather than with an integer format code.
    print(f"{name:16} {entry.get('csi'):9.4f} {int(entry.get('hits')):13d} "
          f"{int(entry.get('misses')):13d} "
          f"{int(entry.get('false_alarms')):13d} "
          f"{entry.get('pod'):7.4f} {entry.get('far'):7.4f} "
          f"{entry.get('bias'):7.4f}")

# The persistence floor is the same number as the analytical Z-R arm, so
# print it next to the ceiling for reference.
floor = arms.get("zr", {}).get("thresholds", {}).get("2.0")
if floor:
    print()
    print(f"persistence floor (zr arm) : CSI@2.0 = {floor['csi']:.4f}")
    print(f"oracle ceiling (ground_zero): CSI@2.0 = "
          f"{arms['ground_zero']['thresholds']['2.0']['csi']:.4f}")

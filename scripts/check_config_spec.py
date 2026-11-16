#!/usr/bin/env python3
"""Check that the upstream's configuration resolver accepts the spec used
by the control-arm script.

``--config`` accepts a bare name (``physical_qpe``) or a path.  The upstream's
own run recorded the *resolved path* in its config.json, which is what made
the path form look correct; but the resolver appends ``.yaml`` to whatever it
is given unless the candidate is already a file, so the path form
``configs/models/physical_qpe.yaml`` fails.  This check is run on the login
node before the job is submitted, because a wrong spec costs a job.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path("/home/wf/radarNWP")
sys.path.insert(0, str(BENCH_ROOT))

from radar_nwp_bench.config import load_config, resolve_config_path  # noqa: E402

failures = 0

for spec in ("physical_qpe", "configs/models/physical_qpe.yaml"):
    try:
        path = resolve_config_path(spec)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {spec!r} -> {type(exc).__name__}: {exc}")
        failures += 1
        continue
    config = load_config(spec)
    model = config.get("model", {})
    print(f"ok    {spec!r} -> {path}")
    print(f"      model section: {model}")
    if not model:
        print("FAIL  the resolved config has no model section")
        failures += 1

print()
if failures:
    print(f"{failures} check(s) failed")
    raise SystemExit(1)
print("all checks passed")

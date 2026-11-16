"""Optional cross-check: the static catalogue against the *live* upstream registry.

    # workstation (no torch): reports SKIP, exits 0
    python scripts/verify_runtime_registry.py

    # benchmark server: set the live checkout and cross-check 25 registrations
    RADAR_NWP_LIVE_ROOT=/home/wf/radarNWP \
        /home/wf/.conda/envs/radar_nwp/bin/python \
        scripts/verify_runtime_registry.py

``qmlbench`` scans the upstream repository as *data* because the workstation
has neither torch nor numpy. That is a metadata question and needs no import.
This script asks a different one: with the real package importable, does every
name we wired actually resolve to a model class? It is deliberately optional --
a missing package reports SKIP rather than failing the dependency-free chain.

Only ``RADAR_NWP_LIVE_ROOT`` points at the live checkout. The evidence mirror
used by the rest of the chain is a different thing and must not be confused
with it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_PATH = PROJECT_ROOT / "configs" / "baselines.json"


def main() -> int:
    live_root = os.environ.get("RADAR_NWP_LIVE_ROOT")
    if not live_root:
        print("SKIP: RADAR_NWP_LIVE_ROOT is not set (no live upstream package).")
        print("      The dependency-free verification chain is unaffected.")
        return 0

    sys.path.insert(0, str(Path(live_root).resolve()))
    try:
        from radar_nwp_bench.models.registry import (  # type: ignore[import-not-found]
            list_models, model_class, registry_snapshot,
        )
    except Exception as exc:  # noqa: BLE001 - a missing torch is a SKIP, not a defect
        print(f"SKIP: cannot import the upstream registry: {type(exc).__name__}: {exc}")
        return 0

    catalogue = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalogued = sorted(
        row["name"] for row in catalogue["baselines"] if row["adapter"]
    )
    runtime = list_models()

    # model_class() triggers discovery; a broken adapter is a real defect but
    # must not abort the whole comparison.
    unresolved: list[str] = []
    for name in runtime:
        try:
            if model_class(name) is None:
                unresolved.append(name)
        except Exception as exc:  # noqa: BLE001 - reported as an unresolved model
            unresolved.append(f"{name} ({type(exc).__name__}: {exc})")

    try:
        snapshot = registry_snapshot()
        snapshot_size: int | None = len(snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"snapshot unavailable: {type(exc).__name__}: {exc}")
        snapshot_size = None

    missing = sorted(set(catalogued) - set(runtime))
    extra = sorted(set(runtime) - set(catalogued))
    print(f"live checkout        : {live_root}")
    print(f"catalogue registered: {len(catalogued)}")
    print(f"runtime registered  : {len(runtime)}")
    print(f"missing at runtime  : {missing}")
    print(f"unseen in catalogue : {extra}")
    print(f"snapshot classes   : {snapshot_size}")
    print(f"unresolved classes : {unresolved}")

    ok = (
        not missing
        and not extra
        and not unresolved
        and snapshot_size == len(runtime)
    )
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

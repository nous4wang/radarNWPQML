"""Sync the full upstream baseline pool into ``configs/baselines.json``.

Run this whenever the upstream ``radar-nwp-bench`` checkout changes. It is
idempotent and dependency-free: source files are read as data, never imported.

    python scripts/sync_baselines.py
    python scripts/sync_baselines.py --check       # verify, do not write
    python scripts/sync_baselines.py --bench-root <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qmlbench.catalog import BASELINE_METADATA, BLUEPRINT_POOL, CATALOG_PATH, GROUPS
from qmlbench.sources import BenchSource

CONTRACT: dict[str, Any] = {
    "seam": "forward(x_radar, x_nwp) -> y_rain",
    "x_radar": "(B, 11, 10, H, W) float32; 11 frames x 6 min, 10 radar variables",
    "x_nwp": "(B, 2, 27, H, W) float32; valid-time aligned, resampled onto radar grid",
    "y_rain": "(B, 10, 1, H, W) float32; mm per 6-minute accumulation (NOT mm/h)",
    "thresholds_mm_per_6min": [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "primary_metric": "CSI@2.0 mm/6min on the full stitched 301x301 grid",
    "split": "case-level; clean split n=538 windows, robustness split n=962 (disjoint cases)",
    "ablation": "x_nwp=None resolved to zeros in BasePrecipitationModel.__call__",
    "authority": "MASTER.md §13 conflict order: MASTER §10.2 > RESULTS_FINAL §10 > ... > README",
}


def build_catalog(source: BenchSource) -> dict[str, Any]:
    adapters = source.scan_adapters()
    configs = source.scan_configs()
    evals = source.scan_eval_reports(".eval.json")
    robustness = source.scan_eval_reports(".robustness.json")
    score_aware = source.score_aware_table()

    names = sorted(
        set(adapters) | set(configs) | set(evals) | set(BASELINE_METADATA)
    )
    unknown = [n for n in names if n not in BASELINE_METADATA]
    if unknown:
        raise RuntimeError(
            "unclassified upstream baselines (add them to BASELINE_METADATA): "
            + ", ".join(unknown)
        )

    rows: list[dict[str, Any]] = []
    for name in names:
        group, paper, venue, license_name, alias = BASELINE_METADATA[name]
        adapter = adapters.get(name, {})
        config = configs.get(name, {})
        is_learnable = bool(adapter.get("is_learnable", True)) and group != "physical"
        rows.append({
            "name": name,
            "group": group,
            "paper": paper,
            "venue": venue,
            "license": config.get("license") or license_name,
            "blueprint_alias": alias,
            "adapter": adapter.get("adapter"),
            "adapter_class": adapter.get("class"),
            "config": config.get("config"),
            "is_learnable": is_learnable,
            "has_eval": name in evals,
            "has_robustness": name in robustness,
            "has_score_aware": name in score_aware,
            "upstream_notes": config.get("notes"),
            "tags": [],
        })

    return {
        "schema_version": 1,
        "source_repo": source.root.as_posix(),
        "source_revision": source.git_revision(),
        "master_version": source.master_version(),
        "contract": CONTRACT,
        "groups": GROUPS,
        "blueprint_pool": list(BLUEPRINT_POOL),
        "counts": {
            "total": len(rows),
            "registered_adapters": len(adapters),
            "configs": len(configs),
            "eval_reports": len(evals),
            "robustness_reports": len(robustness),
            "score_aware_rows": len(score_aware),
        },
        "baselines": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--out", default=str(CATALOG_PATH))
    ap.add_argument("--check", action="store_true", help="compare only; do not write")
    args = ap.parse_args()

    source = BenchSource.locate(args.bench_root)
    payload = build_catalog(source)
    out = Path(args.out)
    new_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not out.is_file():
            print(f"FAIL: {out} does not exist")
            return 1
        old = json.loads(out.read_text(encoding="utf-8"))
        if old == payload:
            print(f"OK: catalogue is in sync with {source.root} ({source.git_revision()})")
            return 0
        print("FAIL: catalogue is stale. Differences:")
        for key in ("source_revision", "master_version", "counts"):
            if old.get(key) != payload.get(key):
                print(f"  {key}: {old.get(key)} -> {payload.get(key)}")
        old_names = {r["name"] for r in old.get("baselines", [])}
        new_names = {r["name"] for r in payload["baselines"]}
        print(f"  added: {sorted(new_names - old_names)}")
        print(f"  removed: {sorted(old_names - new_names)}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new_text, encoding="utf-8")
    c = payload["counts"]
    print(f"Wrote {out}")
    print(f"  upstream   : {source.root} @ {source.git_revision()} ({payload['master_version']})")
    print(f"  baselines  : {c['total']} ({c['registered_adapters']} registered adapters)")
    print(f"  evidence   : {c['eval_reports']} eval, {c['robustness_reports']} robustness, "
          f"{c['score_aware_rows']} score-aware")
    print(f"  blueprint  : {len(BLUEPRINT_POOL)} / {len(BLUEPRINT_POOL)} wired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

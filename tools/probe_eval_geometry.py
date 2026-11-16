#!/usr/bin/env python3
"""Lock down the evaluation geometry the upstream numbers were produced with.

Three different geometries are in play and they must not be mixed:

  * patch 64  / stride 64  -- what every upstream authority script uses
                              (``epoch_curve_eval.sh``, ``run_four_tiers_stage2.sh``)
  * patch 128 / stride 64  -- what the qlfn2 sweep used
  * patch 128 / stride 128 -- what ``evaluate.py`` falls back to when
                              ``--slide-stride`` is not given (it slides by the
                              patch size itself)

The number of stitched tiles follows from ``symmetric_patch_origins`` and
therefore the number of scored samples follows from it, so the geometry can
be read off a stored report: ``n_samples`` = windows x tiles.

This probe reads the stored reports and prints the geometry each one was
produced with, next to the number it reports.  It is read-only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BENCH = Path("/home/wf/radarNWP")
GRID = 301
WINDOWS = 538


def origins(grid_dim: int, patch_size: int, nominal_stride: int) -> tuple[int, ...]:
    """Re-implementation of ``symmetric_patch_origins`` from the benchmark."""
    if patch_size >= grid_dim:
        return (0,)
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    return tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))


def geometry(patch: int, stride: int) -> dict:
    ys = origins(GRID, patch, stride)
    tiles = len(ys) * len(ys)
    return {
        "patch": patch,
        "stride": stride,
        "origins": list(ys),
        "tiles_per_window": tiles,
        "n_samples": WINDOWS * tiles,
    }


def describe(path: Path) -> None:
    print("=" * 78)
    print(path)
    print("=" * 78)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  unreadable: {exc}")
        return
    try:
        report = json.loads(raw)
    except ValueError as exc:
        print(f"  not json: {exc}")
        print(f"  head: {raw[:200]!r}")
        return

    if "arms" in report:                      # the oracle ceiling file
        print(f"  n_samples = {report.get('n_samples')}")
        print(f"  grid      = {report.get('grid')}")
        arms = report["arms"]
        for name, body in arms.items():
            if isinstance(body, dict):
                csi = body.get("csi")
                hits = body.get("hits")
                thr = body.get("threshold")
                print(f"    {name:14s} thr={thr} csi={csi} hits={hits}")
        return

    thr = report.get("thresholds", {})
    n = report.get("n_samples")
    print(f"  n_samples = {n}")
    for key in ("2.0", "0.1", "0.5", "4.0"):
        if key in thr:
            body = thr[key]
            print(f"    @{key:>5s}  csi={body.get('csi')}  hits={body.get('hits')}"
                  f"  pod={body.get('pod')}  far={body.get('far')}"
                  f"  bias={body.get('bias')}")
    extra = {k: v for k, v in report.items()
             if k not in ("thresholds", "n_samples") and not isinstance(v, (dict, list))}
    if extra:
        print(f"  other scalar fields: {extra}")


def main() -> int:
    print("### geometry table (grid 301, 538 test windows) ###")
    for patch, stride in ((64, 64), (128, 64), (128, 128), (64, 128), (301, 301)):
        g = geometry(patch, stride)
        print(f"  patch={g['patch']:>3d} stride={g['stride']:>3d}  "
              f"origins={g['origins']}  tiles={g['tiles_per_window']:>3d}  "
              f"n_samples={g['n_samples']}")
    print()

    targets: list[Path] = []
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        targets = [
            BENCH / "runs" / "four_tiers" / "oracle_ceiling.json",
            BENCH / "runs" / "four_tiers" / "persistence.eval.json",
            BENCH / "runs" / "four_tiers" / "pysteps_adv.eval.json",
        ]
    for target in targets:
        if target.is_file():
            describe(target)
        else:
            print(f"(missing) {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

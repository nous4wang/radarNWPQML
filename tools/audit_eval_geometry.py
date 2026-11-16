#!/usr/bin/env python3
"""Cross-check the evaluation geometry against the upstream authority files.

The qlfn2 sweep scored qlfn2 at CSI@2.0 = 0.1743, above the upstream oracle
upper bound of 0.1653.  A number cannot beat an oracle under a different
measurement, so the geometries have to be compared before the numbers are.

Three geometries are in play:

  64 / 64   25 tiles   every upstream authority script
  128 / 64  16 tiles   the qlfn2 sweep
  128 / 128  9 tiles   evaluate.py's fallback when --slide-stride is omitted

This probe reads the stored upstream reports and reports the geometry each
one carries, so the comparison can be made on the same footing.  It is
read-only and prints a verdict.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BENCH = Path("/home/wf/radarNWP")
GRID = 301
WINDOWS = 538

GEOMETRIES = {
    (64, 64): "upstream authority",
    (128, 64): "qlfn2 sweep",
    (128, 128): "evaluate.py fallback (--slide-stride omitted)",
}


def origins(grid_dim: int, patch_size: int, nominal_stride: int) -> tuple[int, ...]:
    """Re-implementation of ``symmetric_patch_origins`` from the benchmark."""
    if patch_size >= grid_dim:
        return (0,)
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    return tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))


def geometry_of(patch: int, stride: int) -> dict:
    ys = origins(GRID, patch, stride)
    tiles = len(ys) * len(ys)
    return {"patch": patch, "stride": stride, "tiles": tiles,
            "n_samples": WINDOWS * tiles, "origins": list(ys)}


def geometry_from_samples(n_samples: int) -> str:
    """Which geometry produced this many scored samples?"""
    if not n_samples:
        return "unknown"
    for (patch, stride), label in GEOMETRIES.items():
        if WINDOWS * (len(origins(GRID, patch, stride)) ** 2) == n_samples:
            return f"patch={patch} stride={stride} ({label})"
    return f"no geometry matches n_samples={n_samples}"


def read_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def arm_csi_at_2(body: dict) -> float | None:
    """Pull CSI@2.0 out of either report shape."""
    if "thresholds" in body:
        thr = body["thresholds"]
        if "2.0" in thr:
            return thr["2.0"].get("csi")
    return None


def describe(path: Path) -> dict:
    print("=" * 78)
    print(path)
    print("=" * 78)
    report = read_report(path)
    if report is None:
        print("  (unreadable)")
        return {}

    info: dict = {}

    if "arms" in report:
        arms = report["arms"]
        print(f"  n_samples field : {report.get('n_samples')}")
        for name, body in arms.items():
            csi = None
            if isinstance(body, dict):
                csi = arm_csi_at_2(body)
                if csi is None:
                    csi = body.get("csi")
            print(f"    {name:14s} CSI@2.0 = {csi}")
            if name == "ground_zero":
                info["oracle_csi"] = csi
            if name == "zr":
                info["floor_csi"] = csi
        # The oracle file stores its own sample count under a different key.
        n = report.get("n_samples")
        if n:
            info["geometry"] = geometry_from_samples(int(n))
            print(f"  geometry        : {info['geometry']}")
        return info

    n = report.get("n_samples")
    print(f"  n_samples       : {n}")
    if n:
        info["geometry"] = geometry_from_samples(int(n))
        print(f"  geometry        : {info['geometry']}")
    thr = report.get("thresholds", {})
    for key in ("0.1", "0.5", "2.0", "4.0"):
        if key in thr:
            body = thr[key]
            print(f"    @{key:>5s} csi={body.get('csi')} hits={body.get('hits')} "
                  f"pod={body.get('pod')} far={body.get('far')} bias={body.get('bias')}")
            if key == "2.0":
                info["csi2"] = body.get("csi")
                info["hits2"] = body.get("hits")
    return info


def main() -> int:
    print("### geometry table (grid 301, 538 test windows) ###")
    for (patch, stride), label in GEOMETRIES.items():
        g = geometry_of(patch, stride)
        print(f"  patch={patch:>3d} stride={stride:>3d}  origins={g['origins']}  "
              f"tiles={g['tiles']:>3d}  n_samples={g['n_samples']:>6d}   {label}")
    print()

    oracle_path = BENCH / "runs" / "four_tiers" / "oracle_ceiling.json"
    oracle = describe(oracle_path) if oracle_path.is_file() else {}

    print()
    print("### other stored upstream reports ###")
    candidates = []
    for pattern in ("runs/**/persistence*.eval.json",
                    "runs/**/pysteps*.eval.json",
                    "runs/four_tiers/*.eval.json"):
        candidates.extend(sorted(BENCH.glob(pattern))[:6])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        describe(candidate)
        print()

    print("=" * 78)
    print("verdict")
    print("=" * 78)
    oracle_csi = oracle.get("oracle_csi")
    floor_csi = oracle.get("floor_csi")
    print(f"  upstream oracle upper bound CSI@2.0 = {oracle_csi}")
    print(f"  upstream persistence floor   CSI@2.0 = {floor_csi}")
    print(f"  qlfn2 sweep (arm 0, seed 10, last.pt) = 0.1743")
    if oracle_csi is not None and 0.1743 > oracle_csi:
        print()
        print("  The sweep number is ABOVE the oracle bound.  That is only")
        print("  meaningful if both were measured with the same geometry, so")
        print("  the comparison must be redone under one geometry before the")
        print("  number can be read as a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

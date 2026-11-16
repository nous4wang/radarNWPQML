#!/usr/bin/env python3
"""Summarise the geometry comparison into the table the control room needs.

One checkpoint, three evaluation geometries, one number each.  The point is
to make the measurement instrument explicit: the same weights score
differently depending on how the tiles are stitched, so a score without its
geometry is a score without a unit.

The geometry is not a detail of the evaluation.  It enters through
``symmetric_patch_origins``:

    total_span = grid_dim - patch_size
    n_steps    = ceil(total_span / stride)

so both numbers decide where the tiles land, and the tiles are merged back
into the full field with a Hann window, which weights the overlap.  Change
either number and the merged field changes, and with it the contingency
counts the CSI is computed from.

Upstream's authority scripts all use 64/64, and that is the geometry the
oracle ceiling (0.1653) and the persistence floor (0.0924) were measured
with, so 64/64 is the column the comparison has to be made in.

Usage:
    python tools/summarize_geometry.py --root /tmp/geometry
    python tools/summarize_geometry.py --root artifacts/qlfn2_geometry \\
        --markdown artifacts/qlfn2_geometry/geometry.md \\
        --json artifacts/qlfn2_geometry/geometry.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GEOMETRIES: tuple[tuple[int, int, str], ...] = (
    (64, 64, "upstream authority"),
    (128, 64, "the qlfn2 sweep"),
    (128, 128, "evaluate.py fallback"),
)

ARMS: tuple[tuple[str, str], ...] = (
    ("qlfn2", "qlfn2 (motion=corr, CAPPI on, NWP on)"),
    ("qlfn2_no_nwp", "qlfn2 without the NWP"),
    ("qlfn2_no_motion", "qlfn2 without the advection"),
    ("physical_qpe", "physical_qpe (the upstream control)"),
)

AUTHORITY = {
    "physical_qpe ep00 under 64/64": 0.1226,
    "persistence floor": 0.0924,
    "oracle ceiling": 0.1653,
}


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def csi_at(report: dict, threshold: str) -> float | None:
    body = report.get("thresholds", {}).get(threshold)
    if not isinstance(body, dict):
        return None
    value = body.get("csi")
    return float(value) if isinstance(value, (int, float)) else None


def fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/tmp/geometry")
    ap.add_argument("--threshold", default="2.0")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"no geometry reports under {root}")
        return 1

    rows: dict[str, dict[tuple[int, int], float]] = {}
    for arm_dir, _ in ARMS:
        found: dict[tuple[int, int], float] = {}
        for patch, stride, _ in GEOMETRIES:
            report = load(root / f"{arm_dir}_p{patch}_s{stride}.json")
            if report is None:
                continue
            value = csi_at(report, args.threshold)
            if value is not None:
                found[(patch, stride)] = value
        if found:
            rows[arm_dir] = found

    if not rows:
        print(f"no usable reports under {root}")
        return 1

    lines: list[str] = []
    md: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    def emit_md(text: str = "") -> None:
        md.append(text)

    emit("=" * 78)
    emit(f" geometry comparison   CSI@{args.threshold}   root: {root}")
    emit("=" * 78)
    emit_md(f"# Geometry comparison (CSI@{args.threshold})")
    emit_md()
    emit_md("One checkpoint per arm, scored under every geometry. The geometry")
    emit_md("is part of the measurement, not a detail of it.")
    emit_md()

    header = f"{'arm':>22s}"
    for patch, stride, label in GEOMETRIES:
        header += f"{f'{patch}/{stride}':>12s}"
    header += f"{'64/64 -> 128/64':>16s}"
    emit()
    emit(header)
    emit("-" * len(header))
    emit_md("| arm | " + " | ".join(f"{p}/{s}" for p, s, _ in GEOMETRIES)
            + " | 64/64 to 128/64 |")
    emit_md("|---" * (len(GEOMETRIES) + 2) + "|")

    payload: dict = {"root": str(root), "threshold": args.threshold,
                     "authority": AUTHORITY, "arms": {}}

    for arm_dir, label in ARMS:
        found = rows.get(arm_dir)
        if not found:
            continue
        line = f"{label[:22]:>22s}"
        for patch, stride, _ in GEOMETRIES:
            line += f"{fmt(found.get((patch, stride))):>12s}"
        base = found.get((64, 64))
        alt = found.get((128, 64))
        if base and alt:
            ratio = alt / base
            line += f"{f'{ratio:.3f}x':>16s}"
            payload["arms"].setdefault(arm_dir, {})["ratio_64_64_to_128_64"] = ratio
        else:
            line += f"{'-':>16s}"
        emit(line)
        emit_md(f"| {label} | "
                + " | ".join(fmt(found.get((p, s))) for p, s, _ in GEOMETRIES)
                + f" | {(f'{alt / base:.3f}x' if base and alt else '-')} |")
        payload["arms"].setdefault(arm_dir, {})["scores"] = {
            f"{p}/{s}": found.get((p, s)) for p, s, _ in GEOMETRIES
        }

    emit()
    emit("upstream authority (measured under 64/64):")
    for name, value in AUTHORITY.items():
        emit(f"  {name:34s} {value}")
    emit_md()
    emit_md("Upstream authority (64/64):")
    for name, value in AUTHORITY.items():
        emit_md(f"- {name}: {value}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"json: {out}")
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"markdown: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

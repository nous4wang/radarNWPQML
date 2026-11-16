#!/usr/bin/env python3
"""Summarise the epoch-curve runs into the table the control room needs.

The protocol this summarises is the upstream one: every epoch is kept, every
epoch is scored, and the number that is reported is the **peak**, not the
last checkpoint.  The benchmark's own epoch curve records why (E08/E19):

    physical_qpe   ep00 0.1226   ep01 0.0839   ep02 0.0467

so the last checkpoint is the collapsed end of the run.

The geometry is part of the result and is printed with it, because the same
weights score 13.8 % apart under 64/64 and 128/128 (F24).  A table without
the geometry is a table without a unit.

The decision that matters is not any single arm's score but the **paired**
difference between two arms trained on the same seed with the same budget
that differ only in the mechanism:

    arm 2 - arm 0   the advection
    arm 0 - arm 3   qlfn2 against the upstream control
    arm 1 - arm 0   the NWP

The upstream's warning applies to every one of them: the CSI at a fixed
threshold is extremely sensitive to the training draw, and with two seeds a
contrast can show a direction but not an interval.  The summary prints every
seed's difference and says so, rather than reporting a mean as if it were
a measurement.

Usage:
    python tools/summarize_qlfn2_curve.py --root runs/qlfn2_curve
    python tools/summarize_qlfn2_curve.py --root runs/qlfn2_curve \\
        --markdown artifacts/qlfn2_curve.md --json artifacts/qlfn2_curve.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ARMS: tuple[tuple[int, str, str], ...] = (
    (0, "qlfn2", "qlfn2 (motion=corr, CAPPI on, NWP on)"),
    (1, "qlfn2_no_nwp", "qlfn2 without the NWP"),
    (2, "qlfn2_no_motion", "qlfn2 without the advection"),
    (3, "physical_qpe", "physical_qpe (the upstream control)"),
)

CONTRASTS: tuple[tuple[str, int, int, str], ...] = (
    ("advection", 2, 0, "arm 2 - arm 0: does the advection carry anything"),
    ("qlfn2 vs upstream", 0, 3, "arm 0 - arm 3: qlfn2 against the control"),
    ("NWP", 1, 0, "arm 1 - arm 0: what the NWP contributes"),
)

#: The upstream authority numbers, all measured under 64/64.
FLOOR_CSI2 = 0.0924          # persistence floor, Z-R k=1
ORACLE_CSI2 = 0.1653         # oracle ceiling, perfect QPE + no motion
UPSTREAM_QPE_PEAK = 0.1226   # physical_qpe ep00, 64/64


def load_curve(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_curves(root: Path) -> dict[tuple[int, int], dict]:
    """Map (arm, seed) to the curve summary found on disk."""
    found: dict[tuple[int, int], dict] = {}
    for arm_id, arm_dir, _ in ARMS:
        arm_root = root / arm_dir
        if not arm_root.is_dir():
            continue
        for seed_dir in sorted(arm_root.glob("seed*")):
            seed = int(seed_dir.name.replace("seed", "") or -1)
            for candidate in sorted(seed_dir.glob("curve/*_epoch_curve.json")):
                curve = load_curve(candidate)
                if curve is None:
                    continue
                curve["_path"] = str(candidate)
                curve["_arm_id"] = arm_id
                curve["_seed"] = seed
                found[(arm_id, seed)] = curve
    return found


def peak_of(curve: dict) -> float | None:
    value = curve.get("peak_csi2")
    return float(value) if isinstance(value, (int, float)) else None


def last_of(curve: dict) -> float | None:
    value = curve.get("last_csi2")
    return float(value) if isinstance(value, (int, float)) else None


def geometry_of(curve: dict) -> str:
    geo = curve.get("geometry") or {}
    return f"{geo.get('patch_size', '?')}/{geo.get('slide_stride', '?')}"


def fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/qlfn2_curve")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    curves = find_curves(root)
    if not curves:
        print(f"no curve summaries under {root}")
        return 1

    seeds = sorted({seed for _, seed in curves})
    lines: list[str] = []
    md: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    def emit_md(text: str = "") -> None:
        md.append(text)

    emit("=" * 78)
    emit(" qlfn2 epoch curve -- peak vs last, upstream protocol")
    emit(f" root: {root}")
    emit("=" * 78)
    emit_md("# qlfn2 epoch curve (peak vs last)")
    emit_md()

    # -- geometry table -------------------------------------------------
    geoms = sorted({geometry_of(c) for c in curves.values()})
    emit()
    emit(f"geometry in use: {', '.join(geoms)}")
    emit("(64/64 is the upstream authority geometry; the oracle 0.1653 and the")
    emit(" floor 0.0924 were measured under it)")
    emit_md(f"- geometry: {', '.join(geoms)} (64/64 = upstream authority)")
    emit_md()

    # -- per arm, per seed ----------------------------------------------
    emit()
    header = (f"{'arm':>22s}{'seed':>6s}{'peak':>9s}{'@ep':>5s}"
              f"{'last':>9s}{'peak/last':>11s}")
    emit(header)
    emit("-" * len(header))
    emit_md("| arm | seed | peak CSI@2.0 | peak epoch | last.pt CSI@2.0 | peak/last |")
    emit_md("|---|---|---|---|---|---|")

    peaks: dict[tuple[int, int], float] = {}
    lasts: dict[tuple[int, int], float] = {}
    for arm_id, arm_dir, label in ARMS:
        for seed in seeds:
            curve = curves.get((arm_id, seed))
            if curve is None:
                continue
            peak = peak_of(curve)
            last = last_of(curve)
            peaks[(arm_id, seed)] = peak
            lasts[(arm_id, seed)] = last
            epoch = curve.get("peak_epoch")
            ratio = (peak / last) if (peak and last) else None
            emit(f"{label[:22]:>22s}{seed:>6d}{fmt(peak):>9s}"
                 f"{(epoch if epoch is not None else -1):>5d}"
                 f"{fmt(last):>9s}{fmt(ratio, 2) + 'x' if ratio else '-':>11s}")
            emit_md(f"| {label} | {seed} | {fmt(peak)} | "
                     f"{epoch if epoch is not None else '-'} | {fmt(last)} | "
                     f"{fmt(ratio, 2) + 'x' if ratio else '-'} |")

    # -- authority lines ------------------------------------------------
    emit()
    emit("upstream authority (all under 64/64):")
    emit(f"  persistence floor   CSI@2.0 = {FLOOR_CSI2}")
    emit(f"  oracle ceiling      CSI@2.0 = {ORACLE_CSI2}")
    emit(f"  physical_qpe ep00   CSI@2.0 = {UPSTREAM_QPE_PEAK}")
    emit_md()
    emit_md("Upstream authority (64/64):")
    emit_md(f"- persistence floor CSI@2.0 = {FLOOR_CSI2}")
    emit_md(f"- oracle ceiling CSI@2.0 = {ORACLE_CSI2}")
    emit_md(f"- physical_qpe ep00 CSI@2.0 = {UPSTREAM_QPE_PEAK}")
    emit_md()

    # -- paired contrasts ------------------------------------------------
    emit()
    emit("paired contrasts (same seed, same budget, only the mechanism differs)")
    for name, arm_a, arm_b, note in CONTRASTS:
        emit()
        emit(f"  {name}  [{note}]")
        emit(f"  {'seed':>6s}{'a':>10s}{'b':>10s}{'a-b':>11s}")
        emit(f"  {'-' * 37}")
        emit_md()
        emit_md(f"### {name}")
        emit_md(f"{note}")
        emit_md()
        emit_md("| seed | a | b | a - b |")
        emit_md("|---|---|---|---|")
        deltas: list[float] = []
        for seed in seeds:
            value_a = peaks.get((arm_a, seed))
            value_b = peaks.get((arm_b, seed))
            if value_a is None or value_b is None:
                continue
            delta = value_a - value_b
            deltas.append(delta)
            emit(f"  {seed:>6d}{fmt(value_a):>10s}{fmt(value_b):>10s}"
                 f"{fmt(delta):>11s}")
            emit_md(f"| {seed} | {fmt(value_a)} | {fmt(value_b)} | {fmt(delta)} |")
        if deltas:
            mean = sum(deltas) / len(deltas)
            emit(f"  mean {fmt(mean)}   n={len(deltas)}")
            emit("  (two seeds give a direction, not an interval -- the upstream")
            emit("   records a 9.59x spread on the peak across ten seeds)")
            emit_md()
            emit_md(f"mean {fmt(mean)}, n={len(deltas)}. Two seeds give a "
                    f"direction, not an interval.")
        else:
            emit("  no paired seeds available")

    # -- machine-readable -------------------------------------------------
    payload: dict[str, Any] = {
        "root": str(root),
        "geometries": geoms,
        "authority": {
            "persistence_floor_csi2": FLOOR_CSI2,
            "oracle_ceiling_csi2": ORACLE_CSI2,
            "physical_qpe_ep00_csi2": UPSTREAM_QPE_PEAK,
        },
        "runs": {},
        "contrasts": {},
    }
    for (arm_id, seed), curve in curves.items():
        payload["runs"][f"arm{arm_id}_seed{seed}"] = {
            "arm": arm_dir, "seed": seed,
            "peak_csi2": peak_of(curve),
            "peak_epoch": curve.get("peak_epoch"),
            "last_csi2": last_of(curve),
            "geometry": curve.get("geometry"),
            "path": curve.get("_path"),
        }
    for name, arm_a, arm_b, note in CONTRASTS:
        deltas = []
        for seed in seeds:
            value_a = peaks.get((arm_a, seed))
            value_b = peaks.get((arm_b, seed))
            if value_a is not None and value_b is not None:
                deltas.append({"seed": seed, "a": value_a, "b": value_b,
                               "delta": value_a - value_b})
        payload["contrasts"][name] = {"note": note, "pairs": deltas}

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
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

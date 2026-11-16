"""Read-only probe: what do the six CAP* radar variables actually mean?

Answers three questions with measurements, not guesses:

1. Are ``CAP20..CAP70`` constant-altitude levels (2.0 .. 7.0 km) or
   reflectivity thresholds (>= 20 dBZ composite)?  The directory tag
   (``CAP020`` .. ``CAP070``) is consistent with both.
2. If they are altitude levels, the vertical profile identifies the
   melting layer (bright band), which is what a phase-aware Z-R operator
   needs.
3. Is the NWP halo margin really present on disk, i.e. can a second crop
   path widen the NWP window without extrapolating anything?

The probe reads one case from the mmap archive and writes a JSON report.
It never writes to the archive and never trains anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_ROOT = "/home/wf/radarNWP"
DATA_ROOT = "/home/wf/data/radarNWP/BJ_mmap"

RADAR_VARS = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)


def _percentile_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    nz = a[a > 0]
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "frac_positive": float((a > 0).mean()),
        "p50_positive": float(np.percentile(nz, 50)) if nz.size else 0.0,
        "p99_positive": float(np.percentile(nz, 99)) if nz.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--bench-root", default=BENCH_ROOT)
    ap.add_argument("--case", default=None)
    args = ap.parse_args()

    sys.path.insert(0, args.bench_root)
    from radar_nwp_bench.data.pack import load_packed_mmap  # noqa: E402

    root = Path(args.data_root)
    cases = sorted(p for p in root.glob("CP_*") if p.is_dir())
    if not cases:
        raise SystemExit(f"no cases under {root}")
    case_path = Path(args.case) if args.case else cases[0]

    case = load_packed_mmap(case_path)
    radar = np.asarray(case["radar"])   # (T, 10, 301, 301)
    nwp = np.asarray(case["nwp"])       # (T, 27, 167, 167)
    radar_vars = [str(v) for v in case["radar_vars"]]
    nwp_vars = [str(v) for v in case["nwp_vars"]]
    if radar.shape[0] == 0 or radar.size == 0:
        raise SystemExit(f"empty radar array for {case_path}")

    report: dict = {
        "case": case_path.name,
        "n_radar_frames": int(radar.shape[0]),
        "radar_shape": list(radar.shape),
        "nwp_shape": list(nwp.shape),
        "nwp_vars": nwp_vars,
    }

    # -- 1. per-variable statistics ----------------------------------------
    stats: dict[str, dict] = {}
    for i, name in enumerate(radar_vars):
        stats[name] = _percentile_stats(radar[:, i])
    report["radar_var_stats"] = stats

    # -- 2. are the CAP layers altitude levels? ----------------------------
    # Test A: a strong constraint for altitude levels is that the echo top
    # (ET, km) is above the level whenever that level has signal.  For a
    # dBZ-threshold product this test is meaningless, so it is reported
    # separately for each hypothesis.
    cap_idx = [radar_vars.index(f"CAP{d}") for d in (20, 30, 40, 50, 60, 70)]
    et_idx = radar_vars.index("ET")
    vil_idx = radar_vars.index("VIL")
    cr_idx = radar_vars.index("CR")

    # Test B: pointwise monotone containment.  If CAPxx are levels, then
    # CAP70 > CAP60 + 5 dBZ must be rare (a higher level is normally
    # weaker), while CAP20 > CAP70 + 5 dBZ is common.
    t = radar.shape[0]
    cap_stack = radar[:, cap_idx]                       # (T, 6, H, W)
    pairs = {}
    for lo in range(5):
        a = cap_stack[:, lo]
        b = cap_stack[:, lo + 1]
        diff = a - b
        pairs[f"CAP{20*(lo+1)}_minus_CAP{20*(lo+2)}"] = {
            "median": float(np.median(diff)),
            "frac_below_minus_5dbz": float((diff < -5).mean()),
            "frac_above_plus_5dbz": float((diff > 5).mean()),
        }
    report["cap_level_monotonicity"] = pairs

    # Test C: echo-top consistency.  ET is in units of 0.1 km.
    et_km = radar[:, et_idx] / 10.0
    cap_km = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    et_consistency = {}
    for k, (level_km, ci) in enumerate(zip(cap_km, cap_idx)):
        has_signal = radar[:, ci] > 0
        above = et_km > level_km
        both = has_signal & above
        only_above = above & ~has_signal
        et_consistency[f"level_{level_km:.1f}km"] = {
            "frac_level_has_signal_while_echo_top_above": float(both[above].mean())
            if above.any() else None,
            "frac_echo_top_above_but_level_empty": float(only_above.mean())
            if above.any() else None,
            "n_pixels_echo_top_above": int(above.sum()),
        }
    report["echo_top_consistency"] = et_consistency

    # Test D: how far is the vertical profile "layered"?  If CAPxx were
    # dBZ-threshold composites, CAP20 would essentially equal CR.
    cr = radar[:, cr_idx]
    cr_corr = {}
    for name, ci in zip(["CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"], cap_idx):
        x = radar[:, ci].ravel()
        y = cr.ravel()
        ok = (x != 0) & (y != 0)
        cr_corr[name] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 100 else None
        cr_corr[name + "_mean_over_cr"] = float(
            (radar[:, ci] / np.maximum(cr, 1e-6))[radar[:, ci] > 0].mean()
        ) if (radar[:, ci] > 0).any() else None
    report["cap_vs_cr"] = cr_corr

    # Test E: VIL vs the CAP profile.  VIL is vertically integrated liquid;
    # if the CAP levels are altitudes, the profile of CAPPI vs height must
    # be consistent with the VIL magnitude.
    vil = radar[:, vil_idx]
    report["vil_stats"] = _percentile_stats(vil)

    # -- 3. NWP halo margin on disk ---------------------------------------
    # The upstream registration is 167 @0.03 deg -> bilinear x3 -> 501
    # @0.01 deg -> centre crop to 301.  The crop drops 100 px per side,
    # i.e. 33 native cells.  Confirm the dropped ring carries signal.
    native = 167
    scale = 3
    crop = (native * scale - 301) // 2
    report["halo_geometry"] = {
        "native_shape": native,
        "upsampled_shape": native * scale,
        "radar_shape": 301,
        "crop_px_per_side": crop,
        "native_cells_per_side": crop / scale,
        "halo_degrees_per_side": (crop / scale) * 0.03,
    }
    dvg = nwp_vars.index("DVG850")
    field = nwp[:, dvg]
    upsampled = np.repeat(np.repeat(field, scale, axis=-2), scale, axis=-1)
    centre = upsampled[:, crop:crop + 301, crop:crop + 301]
    halo_mask = np.ones(upsampled.shape[-2:], dtype=bool)
    halo_mask[crop:crop + 301, crop:crop + 301] = False
    halo_vals = upsampled[:, halo_mask]
    centre_vals = centre.ravel()
    report["halo_content"] = {
        "n_native_cells_kept": int(301 / scale),
        "halo_std": float(halo_vals.std()),
        "halo_abs_mean": float(np.abs(halo_vals).mean()),
        "centre_std": float(centre_vals.std()),
        "centre_abs_mean": float(np.abs(centre_vals).mean()),
        "halo_frac_cells": float(halo_mask.mean()),
        "halo_is_not_trivial": bool(
            np.abs(halo_vals).mean() > 0.05 * np.abs(centre_vals).mean()
        ),
    }

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {args.out}")
    print(json.dumps(report["halo_content"], ensure_ascii=False))
    print(json.dumps(report["cap_level_monotonicity"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

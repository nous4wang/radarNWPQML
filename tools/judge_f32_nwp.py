#!/usr/bin/env python3
"""F32 verdict reader: does consuming the NWP actually move CSI@2.0?

The criterion is transcribed verbatim from ``MASTER_CONTROL_ROOM.md`` F31 §7
(written BEFORE any GPU was spent) and is deliberately NOT parameterised:

    main     the median-gap ``arm(use_nwp=True) - arm(use_nwp=False)`` has the
             SAME SIGN in seed 10 and seed 11 and its magnitude is > 0.002
             (the nondeterminism floor measured in F30)
    auxiliary the peak-gap must have the same sign in both seeds
    geometry  patch 64 / stride 64; every scored window count = 538
    failure   any seed with a different sign, or |gap| <= 0.002
              => "NWP 无可分辨增量"; the criterion may not be relaxed after
                 the fact, and the peak alone is never sufficient.

There is intentionally no ``--floor`` flag.  A threshold that can be changed
from the command line after the numbers arrive is not a pre-registered
criterion.

Median is defined exactly as in ``summarize_curve_statistics.stats_of``:
``statistics.median`` over the epoch rows' ``csi_2.0``, excluding ``last.pt``.
That module is imported rather than reimplemented, so the two cannot drift.

Usage (after pulling the four curve JSONs back from the remote)::

    python tools/judge_f32_nwp.py --root artifacts/f32_nwp/runs/f32_nwp
    python tools/judge_f32_nwp.py --self-test

Exit codes: 0 = verdict produced (PASS or FAIL are both valid outcomes),
2 = data not ready / incomplete, 3 = a precondition was violated.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from summarize_curve_statistics import load as load_runs  # noqa: E402

# --- Pre-registered constants; do not edit after seeing the results. -------
FLOOR = 0.002
SEEDS = (10, 11)
ARM_ON = "qlfn2"
ARM_OFF = "qlfn2_no_nwp"
EXPECTED_GEOMETRY = {"patch_size": 64, "slide_stride": 64}
EXPECTED_N_SAMPLES = 538
RUN_TOKEN = "f32_nwp"          # guards against judging F30's old checkpoints
DEFAULT_ROOT = ROOT / "artifacts" / "f32_nwp" / "runs" / "f32_nwp"


def median_of(summary: dict) -> float:
    """Same expression as summarize_curve_statistics.stats_of."""
    values = [r["csi_2.0"] for r in summary["rows"] if not r.get("is_last")]
    if not values:
        raise ValueError("no scored epoch rows")
    return float(statistics.median(values))


def check_preconditions(summary: dict, label: str) -> list[str]:
    """Geometry, sample count and version provenance.  All are hard gates."""
    problems: list[str] = []
    geom = summary.get("geometry")
    if geom != EXPECTED_GEOMETRY:
        problems.append(f"{label}: geometry {geom} != {EXPECTED_GEOMETRY}")
    ckpt_dir = str(summary.get("checkpoint_dir", ""))
    if RUN_TOKEN not in ckpt_dir:
        problems.append(
            f"{label}: checkpoint_dir {ckpt_dir!r} is not the F32 run "
            f"(F31 §9.4 forbids cross-version comparisons)")
    rows = summary.get("rows", [])
    if not rows:
        problems.append(f"{label}: no scored rows")
    counts = {r.get("n_samples") for r in rows}
    if counts != {EXPECTED_N_SAMPLES}:
        problems.append(
            f"{label}: n_samples {sorted(map(str, counts))} != "
            f"[{EXPECTED_N_SAMPLES}] -- a different stitching geometry")
    for r in rows:
        if "csi_2.0" not in r:
            problems.append(f"{label}: {r.get('checkpoint')} has no CSI@2.0")
            break
    return problems


def judge(per_seed: dict[int, dict]) -> dict:
    """Apply the pre-registered criterion.  ``per_seed`` maps seed -> gaps."""
    med = [per_seed[s]["median_gap"] for s in SEEDS]
    peak = [per_seed[s]["peak_gap"] for s in SEEDS]
    same_sign_med = all(v > 0 for v in med) or all(v < 0 for v in med)
    same_sign_peak = all(v > 0 for v in peak) or all(v < 0 for v in peak)
    above_floor = all(abs(v) > FLOOR for v in med)
    passed = same_sign_med and same_sign_peak and above_floor
    return {
        "median_gaps": med,
        "peak_gaps": peak,
        "same_sign_median": same_sign_med,
        "same_sign_peak": same_sign_peak,
        "above_floor": above_floor,
        "floor": FLOOR,
        "passed": passed,
        "verdict": ("PASS: NWP 有可分辨增量" if passed
                    else "FAIL: NWP 无可分辨增量（不得事后改判据、不得只报峰值）"),
    }


def _self_test() -> int:
    """The judge itself is covered by branch, not just by function."""
    def sd(m: float, p: float) -> dict:
        return {"median_gap": m, "peak_gap": p}

    cases = [
        ("PASS  two seeds +, above floor, peak agrees",
         {10: sd(0.020, 0.030), 11: sd(0.035, 0.041)}, True),
        ("FAIL  one seed below the floor",
         {10: sd(0.001, 0.030), 11: sd(0.035, 0.041)}, False),
        ("FAIL  opposite signs",
         {10: sd(0.020, 0.030), 11: sd(-0.025, 0.041)}, False),
        ("FAIL  peak sign disagrees",
         {10: sd(0.020, 0.030), 11: sd(0.035, -0.041)}, False),
    ]
    bad = 0
    for name, data, expected in cases:
        got = judge(data)["passed"]
        ok = got == expected
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: passed={got} (want {expected})")
    print(f"\n{len(cases) - bad}/{len(cases)} judge branches passed")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="directory containing <arm>/seed<NN>/curve/*.json")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional path for the machine-readable verdict")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    root: Path = args.root
    if not root.is_dir():
        print(f"WAITING: no pulled results at {root}")
        print("  pull the four curve JSONs from the remote first "
              "(--download ... --out <local path>)")
        return 2

    runs = load_runs(root)
    missing = [f"{arm}/seed{seed}" for seed in SEEDS
               for arm in (ARM_ON, ARM_OFF)
               if seed not in runs.get(arm, {})]
    if missing:
        print("WAITING: missing curve summaries: " + ", ".join(missing))
        return 2

    problems: list[str] = []
    per_seed: dict[int, dict] = {}
    print(f"root     : {root}")
    print(f"criterion: median gap same sign in seeds {SEEDS}, |gap| > {FLOOR}; "
          f"peak gap same sign")
    print()
    for seed in SEEDS:
        on, off = runs[ARM_ON][seed], runs[ARM_OFF][seed]
        problems += check_preconditions(on, f"{ARM_ON}/seed{seed}")
        problems += check_preconditions(off, f"{ARM_OFF}/seed{seed}")
        try:
            m_on, m_off = median_of(on), median_of(off)
            p_on = float(on["peak_csi2"])
            p_off = float(off["peak_csi2"])
        except (KeyError, ValueError, TypeError) as exc:
            print(f"WAITING: incomplete summary for seed {seed}: {exc}")
            return 2
        per_seed[seed] = {
            "median_on": m_on, "median_off": m_off,
            "peak_on": p_on, "peak_off": p_off,
            "median_gap": m_on - m_off,
            "peak_gap": p_on - p_off,
            "n_epochs": len([r for r in on["rows"] if not r.get("is_last")]),
        }
        d = per_seed[seed]
        print(f"seed {seed}: median {m_on:.4f} - {m_off:.4f} = "
              f"{d['median_gap']:+.4f}   |   peak {p_on:.4f} - {p_off:.4f} = "
              f"{d['peak_gap']:+.4f}   |   {d['n_epochs']} epochs")

    print()
    if problems:
        for p in problems:
            print(f"PRECONDITION VIOLATED  {p}")
        print("\nVERDICT: INVALID -- fix the harness, do not interpret the numbers")
        return 3

    result = judge(per_seed)
    result.update({"root": str(root), "seeds": list(SEEDS),
                   "per_seed": per_seed, "geometry": EXPECTED_GEOMETRY})
    print(f"same sign (median) : {result['same_sign_median']}")
    print(f"above floor        : {result['above_floor']}  (floor {FLOOR})")
    print(f"same sign (peak)   : {result['same_sign_peak']}")
    print(f"\nVERDICT: {result['verdict']}")
    if not result["passed"]:
        print("  Write this outcome into MASTER_CONTROL_ROOM.md as F32; "
              "the criterion is fixed and may not be relaxed afterwards.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

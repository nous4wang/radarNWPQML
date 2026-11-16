#!/usr/bin/env python3
"""F36 verdict reader: where does F35's negative NWP effect come from?

The criterion is pre-registered in ``MASTER_CONTROL_ROOM.md`` F36 (1)-(2),
BEFORE any F36 GPU time was spent.  It is deliberately NOT parameterised:
there is no ``--floor`` / ``--gap`` flag, because a threshold that can be
changed from the command line after the numbers arrive is not a
pre-registered criterion.

Question A -- after fixing the NWP channel semantics (D1), does the effect
survive?  ``delta = median(arm) - median(no_nwp)`` for seed 10 and 11.
    positive   both deltas >  FLOOR  -> resolvable positive effect
    negative   both deltas < -FLOOR  -> resolvable negative effect
    flat       both |deltas| <= FLOOR -> effect vanished with the channel bug
    otherwise  INCONCLUSIVE (no direction may be claimed)

Question B -- can the effect be attributed to one subpath?
    ``sep = delta(halo_only) - delta(gate_only)`` per seed
    attributable iff both seps have the same sign AND both |sep| > ATTRIB_GAP.
    sep < 0 means halo_only is worse (halo_proj carries the effect);
    sep > 0 means gate_only is worse (nwp_gate carries it).
    Otherwise UNRESOLVED -- no attribution may be written.

FLOOR = 0.002 is the nondeterminism floor measured in F30.
ATTRIB_GAP = 0.02 is ~38% of F35's effect size (0.053); it is large enough
that only a genuinely separated subpath can clear it.

Hard gates (any violation => rc=3, INVALID, do not interpret the numbers):
    geometry 64/64, n_samples=538 on every scored row, 10 epoch rows,
    checkpoint_dir contains 'f36_nwp' (F31 §9.4 forbids cross-version
    comparisons; F35's four curves are a DIFFERENT channel semantics).

Median is defined exactly as in ``summarize_curve_statistics.stats_of`` and
``judge_f32_nwp.median_of`` (statistics.median over epoch rows' csi_2.0,
excluding last.pt).  ``load`` is imported rather than reimplemented.

Usage (after pulling the eight curve JSONs back from the remote)::

    python tools/judge_f36_nwp.py --root artifacts/f36_nwp/runs/f36_nwp \
        --json artifacts/f36_nwp/verdict.json
    python tools/judge_f36_nwp.py --self-test

Exit codes: 0 = verdict produced (any classification is a valid outcome),
2 = data not ready / incomplete, 3 = a precondition was violated.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
# Insert the directory that actually contains summarize_curve_statistics.
# (The earlier pattern checked ROOT but inserted ROOT/tools, so the import
# silently depended on the caller's cwd / sys.path[0].)
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from summarize_curve_statistics import load as load_runs  # noqa: E402

# --- Pre-registered constants; do not edit after seeing the results. -------
FLOOR = 0.002
ATTRIB_GAP = 0.02
SEEDS = (10, 11)
ARMS = ("both_fixed", "halo_only", "gate_only")
CONTROL = "no_nwp"
ALL_ARMS = ARMS + (CONTROL,)
EXPECTED_GEOMETRY = {"patch_size": 64, "slide_stride": 64}
EXPECTED_N_SAMPLES = 538
EXPECTED_EPOCHS = 10
RUN_TOKEN = "f36_nwp"
DEFAULT_ROOT = ROOT / "artifacts" / "f36_nwp" / "runs" / "f36_nwp"


def median_of(summary: dict) -> float:
    """Same expression as summarize_curve_statistics.stats_of."""
    values = [r["csi_2.0"] for r in summary["rows"] if not r.get("is_last")]
    if not values:
        raise ValueError("no scored epoch rows")
    return float(statistics.median(values))


def same_sign(values) -> bool:
    v = list(values)
    return all(x > 0 for x in v) or all(x < 0 for x in v)


def check_preconditions(summary: dict, label: str) -> list[str]:
    """Geometry, sample count, epoch count and version provenance."""
    problems: list[str] = []
    if summary.get("geometry") != EXPECTED_GEOMETRY:
        problems.append(
            f"{label}: geometry {summary.get('geometry')} != {EXPECTED_GEOMETRY}")
    ckpt_dir = str(summary.get("checkpoint_dir", ""))
    if RUN_TOKEN not in ckpt_dir:
        problems.append(
            f"{label}: checkpoint_dir {ckpt_dir!r} is not the F36 run "
            f"(cross-version comparisons are forbidden)")
    rows = [r for r in summary.get("rows", []) if not r.get("is_last")]
    if len(rows) != EXPECTED_EPOCHS:
        problems.append(
            f"{label}: {len(rows)} scored epoch rows != {EXPECTED_EPOCHS}")
    counts = {r.get("n_samples") for r in rows}
    if counts != {EXPECTED_N_SAMPLES}:
        problems.append(
            f"{label}: n_samples {sorted(map(str, counts))} != "
            f"[{EXPECTED_N_SAMPLES}] -- a different stitching geometry")
    if any("csi_2.0" not in r for r in rows):
        problems.append(f"{label}: a scored row has no CSI@2.0")
    return problems


def classify(gaps: dict) -> dict:
    """Apply the pre-registered criterion.

    ``gaps`` maps arm -> seed -> {"median_gap", "peak_gap"} where each gap is
    ``arm - control`` on the same seed.
    """
    d_both = [gaps["both_fixed"][s]["median_gap"] for s in SEEDS]
    d_halo = [gaps["halo_only"][s]["median_gap"] for s in SEEDS]
    d_gate = [gaps["gate_only"][s]["median_gap"] for s in SEEDS]
    d_peak = [gaps["both_fixed"][s]["peak_gap"] for s in SEEDS]
    sep = [h - g for h, g in zip(d_halo, d_gate)]

    peak_same_sign = same_sign(d_peak)
    if all(x > FLOOR for x in d_both):
        direction = "POSITIVE"
        direction_text = "修复通道语义后 NWP 为可分辨正效应（仍不得写成'NWP 有效'）"
    elif all(x < -FLOOR for x in d_both):
        direction = "NEGATIVE"
        direction_text = "修复通道语义后 NWP 仍为可分辨负效应；F35 负效应不能归因于通道错位"
    elif all(abs(x) <= FLOOR for x in d_both):
        direction = "FLAT"
        direction_text = "修复通道语义后效应落入地板；F35 负效应的主要来源被定位为通道错位"
    else:
        direction = "INCONCLUSIVE"
        direction_text = "方向不可判定（两 seed 不同号或跨地板）；不得写任何方向结论"

    attributable = same_sign(sep) and all(abs(x) > ATTRIB_GAP for x in sep)
    if attributable:
        dominant = "halo_proj" if all(x < 0 for x in sep) else "nwp_gate"
        attribution = dominant
        attribution_text = (
            f"负效应主要由 {dominant} 承担（sep = {sep[0]:+.4f} / {sep[1]:+.4f}）")
    else:
        attribution = "UNRESOLVED"
        attribution_text = (
            f"两条子通路不可分辨（sep = {sep[0]:+.4f} / {sep[1]:+.4f}）；"
            f"不得写归因结论")

    return {
        "direction": direction,
        "direction_text": direction_text,
        "attribution": attribution,
        "attribution_text": attribution_text,
        "peak_gaps_same_sign": peak_same_sign,
        "delta_both": d_both,
        "delta_halo": d_halo,
        "delta_gate": d_gate,
        "sep_halo_minus_gate": sep,
        "gate_marginal": [b - h for b, h in zip(d_both, d_halo)],
        "halo_marginal": [b - g for b, g in zip(d_both, d_gate)],
        "floor": FLOOR,
        "attrib_gap": ATTRIB_GAP,
    }


def _self_test() -> int:
    """Cover the judge's branches, not just run it once."""
    def block(b, h, g, p=(0.0, 0.0)):
        return {
            "both_fixed": {10: {"median_gap": b[0], "peak_gap": p[0]},
                           11: {"median_gap": b[1], "peak_gap": p[1]}},
            "halo_only": {10: {"median_gap": h[0], "peak_gap": 0.0},
                          11: {"median_gap": h[1], "peak_gap": 0.0}},
            "gate_only": {10: {"median_gap": g[0], "peak_gap": 0.0},
                          11: {"median_gap": g[1], "peak_gap": 0.0}},
        }

    cases = [
        ("NEGATIVE + halo_proj dominant",
         block((-0.05, -0.06), (-0.05, -0.06), (-0.001, -0.002)),
         "NEGATIVE", "halo_proj"),
        ("NEGATIVE + nwp_gate dominant",
         block((-0.05, -0.06), (-0.001, -0.002), (-0.05, -0.06)),
         "NEGATIVE", "nwp_gate"),
        ("FLAT + unresolved",
         block((0.001, 0.001), (0.001, 0.001), (0.000, 0.000)),
         "FLAT", "UNRESOLVED"),
        ("POSITIVE + unresolved",
         block((0.03, 0.04), (0.03, 0.04), (0.03, 0.04)),
         "POSITIVE", "UNRESOLVED"),
        ("INCONCLUSIVE (opposite signs)",
         block((0.03, -0.04), (0.03, -0.04), (0.03, -0.04)),
         "INCONCLUSIVE", "UNRESOLVED"),
        ("sep below the attribution gap",
         block((-0.05, -0.06), (-0.05, -0.06), (-0.045, -0.055)),
         "NEGATIVE", "UNRESOLVED"),
    ]
    bad = 0
    for name, data, want_dir, want_attr in cases:
        got = classify(data)
        ok = got["direction"] == want_dir and got["attribution"] == want_attr
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: "
              f"{got['direction']} / {got['attribution']} "
              f"(want {want_dir} / {want_attr})")
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
        print("  pull the eight curve JSONs from the remote first "
              "(--download ... --out <local path>)")
        return 2

    runs = load_runs(root)
    missing = [f"{arm}/seed{seed}" for seed in SEEDS for arm in ALL_ARMS
               if seed not in runs.get(arm, {})]
    if missing:
        print("WAITING: missing curve summaries: " + ", ".join(missing))
        return 2

    problems: list[str] = []
    medians: dict[str, dict[int, float]] = {}
    peaks: dict[str, dict[int, float]] = {}
    print(f"root     : {root}")
    print(f"criterion: floor={FLOOR}, attrib_gap={ATTRIB_GAP}, seeds={SEEDS}")
    print(f"arms     : {', '.join(ARMS)} vs control '{CONTROL}'")
    print()
    for seed in SEEDS:
        for arm in ALL_ARMS:
            summary = runs[arm][seed]
            problems += check_preconditions(summary, f"{arm}/seed{seed}")
            try:
                medians.setdefault(arm, {})[seed] = median_of(summary)
                peaks.setdefault(arm, {})[seed] = float(summary["peak_csi2"])
            except (KeyError, ValueError, TypeError) as exc:
                print(f"WAITING: incomplete summary for {arm}/seed{seed}: {exc}")
                return 2
        print(f"seed {seed}: " + "  ".join(
            f"{a}={medians[a][seed]:.4f}" for a in ALL_ARMS))

    print()
    if problems:
        for p in problems:
            print(f"PRECONDITION VIOLATED  {p}")
        print("\nVERDICT: INVALID -- fix the harness, do not interpret the numbers")
        return 3

    gaps = {
        arm: {
            seed: {
                "median_gap": medians[arm][seed] - medians[CONTROL][seed],
                "peak_gap": peaks[arm][seed] - peaks[CONTROL][seed],
            }
            for seed in SEEDS
        }
        for arm in ARMS
    }
    result = classify(gaps)
    result.update({
        "root": str(root),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "control": CONTROL,
        "geometry": EXPECTED_GEOMETRY,
        "medians": medians,
        "peaks": peaks,
        "gaps": gaps,
    })
    print(f"delta both_fixed : {result['delta_both'][0]:+.4f} / "
          f"{result['delta_both'][1]:+.4f}")
    print(f"delta halo_only  : {result['delta_halo'][0]:+.4f} / "
          f"{result['delta_halo'][1]:+.4f}")
    print(f"delta gate_only  : {result['delta_gate'][0]:+.4f} / "
          f"{result['delta_gate'][1]:+.4f}")
    print(f"peak same sign   : {result['peak_gaps_same_sign']}")
    print(f"\nVERDICT A [{result['direction']}]: {result['direction_text']}")
    print(f"VERDICT B [{result['attribution']}]: {result['attribution_text']}")
    print("\nReminder: 2 seeds give a direction, not an interval.  These four "
          "arms use the corrected NWP channel table and must NOT be placed "
          "side by side with F35's four curves.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

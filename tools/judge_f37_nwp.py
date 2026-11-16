#!/usr/bin/env python3
"""F37 verdict reader: pre-registered per-lead criteria, no CLI relaxation.

The criteria are frozen in ``MASTER_CONTROL_ROOM.md`` F37 (2) BEFORE any GPU
time is spent.  Thresholds are module constants; there is deliberately NO
``--floor`` / ``--gap`` flag, because a threshold that can be changed after
the numbers arrive is not a pre-registered criterion.

    FLOOR            = 0.002   (F30 nondeterminism floor)
    ATTRIB_GAP_LATE  = 0.01    (late-lead subpath attribution gap)
    G0  zero-harm, lead 0..2 (+6..+18 min):
        mean of per-lead 10-epoch median gaps >= -FLOOR on BOTH seeds
    G1  positive late, lead 5..9 (+36..+60 min):
        mean of per-lead 10-epoch median gaps > +FLOOR on BOTH seeds
    G2  pooled CSI@2.0 over all 10 leads:
        10-epoch median gap > +FLOOR on BOTH seeds
    B   subpath attribution on the LATE leads:
        sep_late = gap_vel - gap_src, same sign on both seeds AND
        |sep_late| >= ATTRIB_GAP_LATE, else UNRESOLVED.

Terminal verdict:
    POSITIVE           at least one evolution arm passes G0 + G1 + G2
    PARTIAL_LATE_ONLY  at least one arm passes G0 + G1 but not G2
    NEGATIVE_FALSIFIED no arm passes G0 + G1  -> close the NWP-assistance
                       direction FOR THIS DATASET / THIS RECIPE / THESE ARMS
                       (0-60 min, 0.03 deg hourly NWP vs 0.01 deg/6min radar).
                       This is NOT a claim that "NWP is useless for nowcasting".

Data contract (verified against the F36 pulls, 2026-11-15):
    summary  ``qlfn2_epoch_curve.json`` -> ``rows`` (11 incl. is_last),
             ``geometry``, ``checkpoint_dir``, ``peak_csi2``; each row has
             ``csi_2.0`` and ``n_samples``.
    per-lead ``qlfn2_epoch_XX.json`` -> ``lead_csi`` = {"0".."9":
             {"0.1".."16.0": float}}.  There is NO ``epoch_*.json`` file.
    If any per-lead data is missing -> INVALID (rc=3).  It must NEVER be
    silently replaced by the pooled delta: doing so would quietly discard
    the pre-registered G0/G1 criteria.

Exit codes: 0 = verdict produced, 2 = data not ready, 3 = INVALID gate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Reuse the loader that already reads the REAL summary format correctly.
from summarize_curve_statistics import load as load_runs  # noqa: E402

# --- Pre-registered constants; do not edit after seeing the results. -------
FLOOR: float = 0.002
ATTRIB_GAP_LATE: float = 0.01
SEEDS: tuple[int, ...] = (10, 11)
ARMS: tuple[str, ...] = ("evol_both", "vel_only", "src_only")
CONTROL: str = "no_nwp"
ALL_ARMS: tuple[str, ...] = (*ARMS, CONTROL)
N_EPOCHS: int = 10
N_LEADS: int = 10
EXPECTED_GEOMETRY: dict[str, int] = {"patch_size": 64, "slide_stride": 64}
EXPECTED_N_SAMPLES: int = 538
RUN_TOKEN: str = "f37_nwp"
DEFAULT_ROOT: Path = ROOT / "artifacts" / "f37_nwp" / "runs" / "f37_nwp"


def read_lead_csi2(report: dict[str, Any]) -> list[float] | None:
    """Read the 10-lead CSI@2.0 vector from the REAL evaluator report.

    ``lead_csi`` is keyed by lead index ("0".."9"), each value a dict keyed
    by threshold string.  Any other layout is a contract violation, not a
    format to guess at.
    """
    lc = report.get("lead_csi")
    if not isinstance(lc, dict):
        return None
    out: list[float] = []
    for k in range(N_LEADS):
        row = lc.get(str(k))
        if not isinstance(row, dict):
            return None
        value = row.get("2.0")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        out.append(float(value))
    return out


def _preconditions(summary: dict, label: str) -> list[str]:
    problems: list[str] = []
    if summary.get("geometry") != EXPECTED_GEOMETRY:
        problems.append(
            f"{label}: geometry {summary.get('geometry')} != {EXPECTED_GEOMETRY}")
    ckpt = str(summary.get("checkpoint_dir", ""))
    if RUN_TOKEN not in ckpt:
        problems.append(
            f"{label}: checkpoint_dir {ckpt!r} is not the F37 run "
            f"(cross-version comparisons are forbidden)")
    rows = [r for r in summary.get("rows", []) if not r.get("is_last")]
    if len(rows) != N_EPOCHS:
        problems.append(f"{label}: {len(rows)} epoch rows != {N_EPOCHS}")
    counts = {r.get("n_samples") for r in rows}
    if counts != {EXPECTED_N_SAMPLES}:
        problems.append(
            f"{label}: n_samples {sorted(map(str, counts))} != "
            f"[{EXPECTED_N_SAMPLES}] -- a different stitching geometry")
    if any(not isinstance(r.get("csi_2.0"), (int, float))
           or not math.isfinite(float(r["csi_2.0"])) for r in rows):
        problems.append(f"{label}: a scored row has no finite CSI@2.0")
    return problems


def load_run_data(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load 8 summaries + 80 per-lead reports, with hard INVALID gates."""
    if not root.is_dir():
        print(f"WAITING: no pulled results at {root}")
        print("  pull the eight curve directories from the remote first "
              "(--download ... --out <local path>)")
        raise SystemExit(2)

    runs = load_runs(root)
    missing = [f"{arm}/seed{seed}" for seed in SEEDS for arm in ALL_ARMS
               if seed not in runs.get(arm, {})]
    if missing:
        print("WAITING: missing curve summaries: " + ", ".join(missing))
        raise SystemExit(2)

    problems: list[str] = []
    seen_ckpt: set[str] = set()
    data: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ALL_ARMS}

    for arm in ALL_ARMS:
        for seed in SEEDS:
            label = f"{arm}/seed{seed}"
            summary = runs[arm][seed]
            problems += _preconditions(summary, label)
            ckpt = str(summary.get("checkpoint_dir", ""))
            if ckpt in seen_ckpt:
                problems.append(f"{label}: duplicate checkpoint_dir {ckpt!r}")
            seen_ckpt.add(ckpt)

            rows = [r for r in summary["rows"] if not r.get("is_last")]
            pooled = [float(r["csi_2.0"]) for r in rows
                      if isinstance(r.get("csi_2.0"), (int, float))]

            cdir = root / arm / f"seed{seed}" / "curve"
            epoch_files = sorted(
                p for p in cdir.glob("qlfn2_epoch_*.json")
                if "epoch_curve" not in p.name and "last" not in p.name
            )
            if len(epoch_files) != N_EPOCHS:
                problems.append(
                    f"{label}: {len(epoch_files)} per-lead reports != {N_EPOCHS}; "
                    f"G0/G1 cannot be evaluated without them")
            lead_matrix: list[list[float]] = []
            for p in epoch_files:
                try:
                    rep = json.loads(p.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    problems.append(f"{label}: cannot read {p.name}: {exc}")
                    continue
                vec = read_lead_csi2(rep)
                if vec is None:
                    problems.append(
                        f"{label}: {p.name} has no complete 10-lead "
                        f"lead_csi[k]['2.0'] vector")
                else:
                    lead_matrix.append(vec)

            lead_medians = None
            if len(lead_matrix) == N_EPOCHS:
                lead_medians = [
                    statistics.median(lead_matrix[e][k] for e in range(N_EPOCHS))
                    for k in range(N_LEADS)
                ]

            data[arm][seed] = {
                "median": statistics.median(pooled) if len(pooled) == N_EPOCHS else None,
                "peak": float(summary.get("peak_csi2", float("nan"))),
                "lead_medians": lead_medians,
            }

    if problems:
        for p in problems:
            print(f"PRECONDITION VIOLATED  {p}")
        print("\nVERDICT: INVALID -- fix the harness; do NOT interpret the "
              "numbers. G0/G1 are per-lead criteria and must not be "
              "replaced by the pooled delta.")
        raise SystemExit(3)
    return data


def evaluate_criteria(data: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Apply the frozen G0/G1/G2/B criteria.  No silent fallback."""
    arm_evals: dict[str, Any] = {}
    any_full = False
    any_g0_g1 = False

    for arm in ARMS:
        d_all: list[float] = []
        d_peak: list[float] = []
        d_early: list[float] = []
        d_late: list[float] = []
        for seed in SEEDS:
            base, cur = data[CONTROL][seed], data[arm][seed]
            d_all.append(cur["median"] - base["median"])
            d_peak.append(cur["peak"] - base["peak"])
            # load_run_data guarantees both vectors exist (else rc=3).
            diffs = [cur["lead_medians"][k] - base["lead_medians"][k]
                     for k in range(N_LEADS)]
            d_early.append(statistics.mean(diffs[:3]))   # Lead 0..2
            d_late.append(statistics.mean(diffs[5:]))    # Lead 5..9

        g0 = all(x >= -FLOOR for x in d_early)
        g1 = all(x > FLOOR for x in d_late)
        g2 = all(x > FLOOR for x in d_all)
        if g0 and g1 and g2:
            any_full = True
        if g0 and g1:
            any_g0_g1 = True
        arm_evals[arm] = {
            "delta_all": d_all,
            "delta_peak": d_peak,
            "delta_early_L0_2": d_early,
            "delta_late_L5_9": d_late,
            "G0_zero_harm_early": g0,
            "G1_positive_late": g1,
            "G2_positive_all": g2,
            "pass_full": g0 and g1 and g2,
            "HARMFUL_EARLY": not g0,
        }

    sep_late = [
        arm_evals["vel_only"]["delta_late_L5_9"][i]
        - arm_evals["src_only"]["delta_late_L5_9"][i]
        for i in range(len(SEEDS))
    ]
    if (sep_late[0] * sep_late[1] > 0
            and all(abs(x) >= ATTRIB_GAP_LATE for x in sep_late)):
        attribution = "VEL_DOMINANT" if sep_late[0] > 0 else "SRC_DOMINANT"
        attrib_text = (f"远时效子通路可分辨 ({attribution}, "
                       f"sep_late={sep_late[0]:+.4f}/{sep_late[1]:+.4f})")
    else:
        attribution = "UNRESOLVED"
        attrib_text = (f"两条子通路在远时效不可分辨 "
                       f"(sep_late={sep_late[0]:+.4f}/{sep_late[1]:+.4f})")

    if any_full:
        direction = "POSITIVE"
        dir_text = ("PASS: 至少一个 t>0 演进臂同时通过 G0(短时效零伤害) + "
                    "G1(远时效正增量) + G2(全域正增量)")
    elif any_g0_g1:
        direction = "PARTIAL_LATE_ONLY"
        dir_text = ("PARTIAL: 守住 G0 并在远时效 (Lead 5..9) 取得可分辨正增量，"
                    "但全域中位数未过 +0.002")
    else:
        direction = "NEGATIVE_FALSIFIED"
        dir_text = ("FALSIFIED: 在本数据集 0-60 min 尺度、本配方与这 3 条演进臂上，"
                    "NWP 作用于 t0 幅度 (F35/F36) 或 t>0 平流/生消 (F37) 均无"
                    "有效正增量 ⇒ 该方向在此范围内封口；"
                    "不得推广为「NWP 对降水临近预报无用」")

    return {
        "direction": direction,
        "direction_text": dir_text,
        "attribution": attribution,
        "attribution_text": attrib_text,
        "sep_late": sep_late,
        "floor": FLOOR,
        "attrib_gap_late": ATTRIB_GAP_LATE,
        "seeds": list(SEEDS),
        "arms": arm_evals,
    }


def _run_self_test() -> int:
    def _mk(m_no, m_both, m_vel, m_src, early=0.0, late=0.0):
        out: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ALL_ARMS}
        for i, s in enumerate(SEEDS):
            base_leads = [0.25 - 0.02 * k for k in range(N_LEADS)]
            out["no_nwp"][s] = {"median": m_no[i], "peak": m_no[i] + 0.02,
                                "lead_medians": base_leads}
            for name, arr in (("evol_both", m_both), ("vel_only", m_vel),
                              ("src_only", m_src)):
                leads = [base_leads[k] + (early if k < 3 else (late if k >= 5 else 0.0))
                         for k in range(N_LEADS)]
                out[name][s] = {"median": arr[i], "peak": arr[i] + 0.02,
                                "lead_medians": leads}
        return out

    cases = [
        ("POSITIVE", _mk([.13, .14], [.136, .146], [.135, .145], [.131, .141], .0, +.008), "POSITIVE"),
        ("PARTIAL_LATE_ONLY", _mk([.13, .14], [.131, .141], [.131, .141], [.130, .140], .0, +.005), "PARTIAL_LATE_ONLY"),
        ("G0 violation", _mk([.13, .14], [.135, .145], [.135, .145], [.135, .145], -.010, +.015), "NEGATIVE_FALSIFIED"),
        ("all negative", _mk([.13, .14], [.125, .135], [.128, .138], [.127, .137], -.001, -.005), "NEGATIVE_FALSIFIED"),
    ]
    bad = 0
    for name, data, want in cases:
        got = evaluate_criteria(data)["direction"]
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: {got} (want {want})")

    d5 = _mk([.13, .14], [.136, .146], [.136, .146], [.130, .140], .0, .0)
    for s in SEEDS:
        d5["vel_only"][s]["lead_medians"] = [x + .015 for x in d5["no_nwp"][s]["lead_medians"]]
    r5 = evaluate_criteria(d5)["attribution"]
    ok5 = r5 == "VEL_DOMINANT"
    bad += not ok5
    print(f"{'ok  ' if ok5 else 'FAIL'}  attribution VEL_DOMINANT: {r5}")

    d6 = _mk([.13, .14], [.131, .141], [.131, .141], [.131, .141], .0, .001)
    r6 = evaluate_criteria(d6)["attribution"]
    ok6 = r6 == "UNRESOLVED"
    bad += not ok6
    print(f"{'ok  ' if ok6 else 'FAIL'}  attribution UNRESOLVED: {r6}")

    # Parsing contract: the REAL report layout, and the hard failure when
    # per-lead data is missing (this is the bug that silently voided G0/G1).
    import tempfile
    good = {"lead_csi": {str(k): {"2.0": 0.25 - 0.02 * k} for k in range(10)}}
    ok7 = read_lead_csi2(good) is not None and read_lead_csi2({"lead_csi": {"2.0": [0.1] * 10}}) is None
    bad += not ok7
    print(f"{'ok  ' if ok7 else 'FAIL'}  lead_csi parser accepts real layout / rejects guessed layout")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ALL_ARMS:
            for seed in SEEDS:
                cdir = root / arm / f"seed{seed}" / "curve"
                cdir.mkdir(parents=True)
                rows = [{"epoch": e, "is_last": False, "n_samples": 538,
                         "csi_2.0": 0.13 + 0.001 * e} for e in range(10)]
                rows.append({"epoch": None, "is_last": True, "n_samples": 538,
                             "csi_2.0": 0.12})
                (cdir / "qlfn2_epoch_curve.json").write_text(json.dumps({
                    "geometry": {"patch_size": 64, "slide_stride": 64},
                    "checkpoint_dir": f"/home/wf/radarNWPQML/runs/f37_nwp/{arm}/seed{seed}/train/x",
                    "rows": rows, "peak_csi2": 0.139,
                }), encoding="utf-8")
                for e in range(10):
                    (cdir / f"qlfn2_epoch_{e:02d}.json").write_text(
                        json.dumps(good), encoding="utf-8")
        try:
            loaded = load_run_data(root)
            ok8 = all(loaded[a][s]["lead_medians"] is not None
                      for a in ALL_ARMS for s in SEEDS)
        except SystemExit as exc:
            ok8 = False
            print(f"    unexpected exit {exc.code}")
        bad += not ok8
        print(f"{'ok  ' if ok8 else 'FAIL'}  load_run_data reads all 8 runs with 10 leads")

        # remove one per-lead report -> must be INVALID, never a silent fallback
        (root / ARMS[0] / "seed10" / "curve" / "qlfn2_epoch_09.json").unlink()
        try:
            load_run_data(root)
            ok9 = False
        except SystemExit as exc:
            ok9 = exc.code == 3
        bad += not ok9
        print(f"{'ok  ' if ok9 else 'FAIL'}  missing per-lead report -> INVALID (rc=3), no fallback")

    total = 9
    print(f"self-test: {total - bad}/{total} branches passed")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="F37 预注册判据裁决器")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_run_self_test())

    data = load_run_data(args.root)
    verdict = evaluate_criteria(data)
    verdict["root"] = str(args.root)

    print(f"root     : {args.root}")
    print(f"criterion: floor={FLOOR}, attrib_gap_late={ATTRIB_GAP_LATE}, seeds={SEEDS}")
    for arm in ARMS:
        ev = verdict["arms"][arm]
        print(
            f"{arm:<10} | Δ_all={ev['delta_all'][0]:+.4f}/{ev['delta_all'][1]:+.4f} "
            f"| Δ_early(L0..2)={ev['delta_early_L0_2'][0]:+.4f}/{ev['delta_early_L0_2'][1]:+.4f} (G0={ev['G0_zero_harm_early']}) "
            f"| Δ_late(L5..9)={ev['delta_late_L5_9'][0]:+.4f}/{ev['delta_late_L5_9'][1]:+.4f} (G1={ev['G1_positive_late']})"
        )
    print(f"VERDICT A [{verdict['direction']}]: {verdict['direction_text']}")
    print(f"VERDICT B [{verdict['attribution']}]: {verdict['attribution_text']}")
    print("Reminder: 2 seeds give a direction, not an interval.  F37's 8 curves "
          "must NOT be placed side by side with F35/F36 curves.")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

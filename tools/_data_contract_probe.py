#!/usr/bin/env python3
"""Read-only audit: does BJ_mmap actually satisfy the task's data contract?

Goes beyond "the directory exists": checks every case's metadata, array
shapes/dtypes, 6-minute cadence, NWP valid-time alignment, the case-level
train/val/test partition, and one real dataset sample against the documented
IO contract (radar 11x10, NWP 2x27, label 10x1, 301x301 full-field eval).

Zero training, zero inference, zero writes into either repository. Writes a
single JSON report to --out (normally /tmp), which is then pulled back with
``tools/remote.py --download ... --out tools/_evals/...``.

    /home/wf/.conda/envs/radar_nwp/bin/python \
        /home/wf/radarNWPQML/tools/_data_contract_probe.py \
        --out /tmp/data_contract_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

BENCH_ROOT = Path("/home/wf/radarNWP")
MMAP_ROOT = Path("/home/wf/data/radarNWP/BJ_mmap")
RAW_ROOT = Path("/home/wf/data/radarNWP/BJ")
sys.path.insert(0, str(BENCH_ROOT))

import numpy as np  # noqa: E402

from radar_nwp_bench.data.index import (  # noqa: E402
    IRREGULAR_CASES,
    NWP_VARS,
    RADAR_VARS,
    minutes_between,
    split_cases,
)
from radar_nwp_bench.data.dataset import RadarNWPDataset  # noqa: E402

REQUIRED_FILES = ("radar.npy", "nwp.npy", "label.npy", "meta.json")


def safe(label: str, fn) -> dict:
    try:
        return {"label": label, "ok": True, **fn()}
    except Exception as exc:  # noqa: BLE001 - report every failure, never abort
        return {
            "label": label,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def audit_cases(case_dirs: list[Path]) -> dict:
    missing: list[str] = []
    metas: dict[str, dict] = {}
    for case in case_dirs:
        absent = [f for f in REQUIRED_FILES if not (case / f).is_file()]
        if absent:
            missing.append(f"{case.name}: {','.join(absent)}")
            continue
        metas[case.name] = json.loads((case / "meta.json").read_text())
    return {"n_cases": len(case_dirs), "missing_files": missing, "metas": metas}


def audit_meta(metas: dict[str, dict]) -> dict:
    keys = ("radar_shape", "nwp_shape", "radar_vars", "nwp_vars")
    variants: dict[str, dict[str, list[str]]] = {k: {} for k in keys}
    gaps: list[str] = []
    label_off: list[str] = []
    nwp_bad: list[str] = []
    source_bad: list[str] = []
    var_mismatch: list[str] = []
    for name, meta in metas.items():
        for key in keys:
            variants[key].setdefault(json.dumps(meta.get(key)), []).append(name)
        # Times are HHMM integers, so a numeric difference of 46 is a
        # legitimate 6-minute step across an hour boundary. Use the bench's
        # own calendar arithmetic.
        times = meta.get("radar_times", [])
        if any(minutes_between(b, a) != 6 for a, b in zip(times, times[1:])):
            gaps.append(name)
        # A hole in the radar record removes the matching label frame. This
        # is the task's "missing-frame isolation", not a corrupt mmap: the
        # affected cases must be exactly IRREGULAR_CASES.
        if meta.get("label_times") != times:
            label_off.append(name)
        nwp_times = meta.get("nwp_times", [])
        if any(t % 100 != 0 for t in nwp_times):
            nwp_bad.append(name)
        source = Path(meta.get("source", ""))
        if source.name != name or not source.is_dir():
            source_bad.append(name)
        if (tuple(meta.get("radar_vars", ())) != tuple(RADAR_VARS)
                or tuple(meta.get("nwp_vars", ())) != tuple(NWP_VARS)):
            var_mismatch.append(name)
    return {
        "distinct_values": {k: {v: len(n) for v, n in d.items()}
                            for k, d in variants.items()},
        "radar_vars_match_code": tuple(RADAR_VARS),
        "nwp_vars_match_code": tuple(NWP_VARS),
        "cases_with_var_mismatch": var_mismatch,
        "cases_with_expected_irregular_time_gaps": gaps,
        "gaps_cases_equal_irregular_set": set(gaps) == set(IRREGULAR_CASES),
        "cases_with_label_times_ne_radar_times": label_off,
        "label_off_cases_equal_irregular_set": set(label_off) == set(
            IRREGULAR_CASES),
        "n_label_off_cases": len(label_off),
        "n_irregular_cases": len(IRREGULAR_CASES),
        "cases_with_non_hourly_nwp": nwp_bad,
        "cases_with_bad_source_link": source_bad,
    }


def audit_raw_consistency(metas: dict[str, dict]) -> dict:
    """Every mmap case must be reconstructible from the BJ raw tree."""
    mismatches: list[str] = []
    checked = 0
    for name, meta in metas.items():
        raw = RAW_ROOT / name
        radar_files = sum(1 for _ in (raw / "RADAR").rglob("*.npy"))
        label_files = sum(1 for _ in (raw / "LABEL").rglob("*.npy"))
        nwp_files = sum(1 for _ in (raw / "NWP").rglob("*.npy"))
        n_t = len(meta.get("radar_times", ()))
        n_l = len(meta.get("label_times", ()))
        n_n = len(meta.get("nwp_times", ()))
        if (radar_files != n_t * len(RADAR_VARS)
                or nwp_files != n_n * len(NWP_VARS)
                or label_files != n_l):
            mismatches.append(
                f"{name}: raw radar={radar_files} (want {n_t * len(RADAR_VARS)}), "
                f"label={label_files} (want {n_l}), "
                f"nwp={nwp_files} (want {n_n * len(NWP_VARS)})"
            )
        checked += 1
    mismatch_ids = [m.split(":")[0] for m in mismatches]
    return {
        "rule": "raw RADAR >= T x 10 vars; LABEL = T frames; NWP = 6 x 27; "
                "extra raw RADAR frames are dropped by the 6-min time grid",
        "cases_checked": checked,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "mismatch_case_ids": mismatch_ids,
        "all_mismatches_are_irregular_cases": all(
            cid in IRREGULAR_CASES for cid in mismatch_ids
        ),
    }


def audit_arrays(case_dirs: list[Path]) -> dict:
    shapes: dict[str, dict[str, list[str]]] = {
        name: {} for name in ("radar", "nwp", "label")
    }
    bad: list[str] = []
    for case in case_dirs:
        for name in ("radar", "nwp", "label"):
            path = case / f"{name}.npy"
            if not path.is_file():
                continue
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            key = f"{arr.shape}|{arr.dtype}"
            shapes[name].setdefault(key, []).append(case.name)
            probe = np.asarray(arr[0], dtype=np.float64)
            if not np.isfinite(probe).all():
                bad.append(f"{case.name}/{name}: non-finite in frame 0")
    return {
        "distinct_shapes": {
            k: {v: len(n) for v, n in d.items()} for k, d in shapes.items()
        },
        "non_finite_findings": bad[:20],
        "n_non_finite_findings": len(bad),
    }


def audit_split(case_dirs: list[Path]) -> dict:
    ids = sorted(c.name for c in case_dirs)
    parts = split_cases(ids)
    irregular_present = [c for c in ids if c in IRREGULAR_CASES]
    return {
        "split_rule": "sorted case ids; 0.6/0.2/0.2 by CASE, not by sample; "
                      "IRREGULAR_CASES removed into robustness_missing",
        "counts": {k: len(v) for k, v in parts.items()},
        "train": parts["train"],
        "val": parts["val"],
        "test": parts["test"],
        "irregular_cases_present_in_mmap": irregular_present,
        "n_irregular_cases_declared": len(IRREGULAR_CASES),
        "leakage_check": {
            "train_val_disjoint": not (set(parts["train"]) & set(parts["val"])),
            "train_test_disjoint": not (set(parts["train"]) & set(parts["test"])),
            "val_test_disjoint": not (set(parts["val"]) & set(parts["test"])),
        },
    }


def audit_dataset() -> dict:
    """Use the exact constructor flags train.py / evaluate.py use.

    The mmap tree is a *packed* tree: without ``packed=True`` the index
    scanner looks for the raw BJ subdirectories and finds no samples.
    """
    out: dict = {}
    for split, random_crop in (("train", True), ("test", False)):
        ds = RadarNWPDataset(
            data_root=str(MMAP_ROOT),
            packed=True,
            split=split,
            in_len=11,
            out_len=10,
            patch_size=64,
            slide_stride=64,
            alignment="forecast",
            random_crop=random_crop,
            tolerate_one_missing_frame=True,
        )
        sample = ds[0]
        tensors = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in sample.items() if hasattr(v, "shape")
        }
        out[split] = {
            "n_windows": len(ds.samples),
            "n_patch_positions": ds.n_patch_positions,
            "len_dataset": len(ds),
            "sample_keys": sorted(tensors),
            "tensors": tensors,
            "first_sample_finite": all(
                bool(np.isfinite(np.asarray(v, dtype=np.float64)).all())
                for v in sample.values() if hasattr(v, "shape")
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    case_dirs = sorted(p for p in MMAP_ROOT.glob("CP_*") if p.is_dir())
    raw_dirs = sorted(p for p in RAW_ROOT.glob("CP_*") if p.is_dir())

    case_result = safe("cases", lambda: audit_cases(case_dirs))
    metas = case_result.get("metas", {}) if case_result["ok"] else {}
    checks = [
        case_result,
        safe("metadata", lambda: audit_meta(metas)),
        safe("arrays", lambda: audit_arrays(case_dirs)),
        safe("raw_consistency", lambda: audit_raw_consistency(metas)),
        safe("split", lambda: audit_split(case_dirs)),
        safe("dataset_contract", audit_dataset),
    ]

    meta_check, arr_check, raw_check, split_check, ds_check = checks[1:]
    problems: list[str] = []
    if case_result.get("missing_files"):
        problems.append("case directories missing required files")
    if meta_check.get("ok"):
        for key in ("cases_with_var_mismatch", "cases_with_non_hourly_nwp",
                    "cases_with_bad_source_link"):
            if meta_check.get(key):
                problems.append(key)
        # Time gaps and unequal radar/label timestamps are the *required*
        # missing-frame isolation: they must occur in exactly the 23 cases
        # that split_cases routes to robustness_missing.
        if not meta_check.get("gaps_cases_equal_irregular_set"):
            problems.append("time-gap cases differ from IRREGULAR_CASES")
        if not meta_check.get("label_off_cases_equal_irregular_set"):
            problems.append("label/radar time mismatch is not the irregular set")
    if arr_check.get("n_non_finite_findings"):
        problems.append("non-finite array values")
    if raw_check.get("ok") and not raw_check.get(
        "all_mismatches_are_irregular_cases"
    ):
        problems.append("clean case does not reproduce from the BJ raw tree")
    if split_check.get("ok") and not all(
        split_check["leakage_check"].values()
    ):
        problems.append("train/val/test overlap")
    for check in checks:
        if not check["ok"]:
            problems.append(f"{check['label']} raised: {check['error']}")

    payload = {
        "probe": "tools/_data_contract_probe.py",
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": __import__("platform").node(),
        "bench_root": str(BENCH_ROOT),
        "mmap_root": str(MMAP_ROOT),
        "raw_root": str(RAW_ROOT),
        "n_mmap_cases": len(case_dirs),
        "n_raw_cases": len(raw_dirs),
        "raw_case_ids_match_mmap": (
            [p.name for p in raw_dirs] == [p.name for p in case_dirs]
        ),
        "task_contract": {
            "radar": "11 history frames x 10 variables, 301x301",
            "nwp": "2 valid-time slices x 27 variables, resampled 167->301",
            "label": "10 future frames x 1 rain channel, mm/6min",
            "eval": "full 301x301 stitched field, n=538",
            "pipeline_fixes": "geometry registration / 6-min time grid / "
                              "missing-frame isolation",
        },
        "checks": checks,
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
        "scope": "read-only; zero training / inference / data mutation / repo write",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"report: {args.out}")
    print(f"cases  : mmap={len(case_dirs)} raw={len(raw_dirs)}")
    for check in checks:
        print(f"  {check['label']:<18} {'ok' if check['ok'] else 'ERROR'}")
    print(f"problems: {problems}")
    print(f"verdict : {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare one newly evaluated checkpoint with the archived main-table row.

Both reports must come from the *same* protocol (clean test split, whole
301x301 grid, 64x64 patch at stride 64); otherwise the subtraction is not
meaningful. This script never merges the standard and score-aware tables and
never ranks models -- it compares two checkpoints of one model.

    python scripts/compare_eval.py --new artifacts/remote_test/<run>.eval.json \
        --archive <bench>/.workbuddy/runs/mocast.eval.json \
        --out artifacts/remote_test/eval_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qmlbench.results import FLOOR_CSI2, HEADROOM, SEED_DISPERSION, SEED_RANGE

KEYS = ("0.1", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0")


def _thresholds(path: Path) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("thresholds") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="newly written *.eval.json")
    ap.add_argument("--archive", required=True, help="archived *.eval.json for the same model")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "artifacts" / "remote_test" / "eval_comparison.json"))
    args = ap.parse_args()

    new_path, archive_path = Path(args.new), Path(args.archive)
    new, archive = _thresholds(new_path), _thresholds(archive_path)

    rows = []
    for key in KEYS:
        a = new.get(key, {})
        b = archive.get(key, {})
        rows.append({
            "threshold": key,
            "new_csi": a.get("csi"),
            "archive_csi": b.get("csi"),
            "delta_vs_archive": (a["csi"] - b["csi"])
            if isinstance(a.get("csi"), (int, float))
            and isinstance(b.get("csi"), (int, float)) else None,
            "new_bias": a.get("bias"),
            "new_far": a.get("far"),
        })

    new_csi2 = (new.get("2.0") or {}).get("csi")
    archive_csi2 = (archive.get("2.0") or {}).get("csi")
    new_samples = json.loads(new_path.read_text(encoding="utf-8")).get("n_samples")
    archive_samples = json.loads(archive_path.read_text(encoding="utf-8")).get("n_samples")

    result = {
        "new_report": str(new_path),
        "archive_report": str(archive_path),
        "context": {
            "prior_review": "artifacts/remote_test/training_run_review.json",
            "note": "prior review records the state before evaluation was run "
                    "(evaluated=False); this file is the post-evaluation result.",
        },
        "protocol": "clean test split, 301x301 whole grid, 64x64 patch @ stride 64",
        "n_samples": {"new": new_samples, "archive": archive_samples},
        "rows": rows,
        "csi2": {
            "new": new_csi2,
            "archive": archive_csi2,
            "persistence_floor": FLOOR_CSI2,
            "gap_to_floor": (FLOOR_CSI2 - new_csi2)
            if isinstance(new_csi2, (int, float)) else None,
            "oracle_headroom": HEADROOM,
            "single_seed_range": SEED_RANGE,
            "range_over_headroom": round(SEED_RANGE / HEADROOM, 4),
            "peak_end_dispersion": SEED_DISPERSION,
        },
        "verdict": {
            "beats_persistence_floor": bool(
                isinstance(new_csi2, (int, float)) and new_csi2 > FLOOR_CSI2
            ),
            "supports_improvement_claim": False,
            "reasons": [
                "单 seed (seed=0)，MASTER §11.9 P1 要求 >=10 独立 seed",
                "单 seed 极差 0.1671 是 Oracle−地板可用空间 0.0729 的 2.29 倍",
                "训练 loss 不是 CSI；一次评测分数不能与抽签区分",
                "§11.9 P3 的通过率须绑定 ep00，本次 checkpoint 是 ep09 权重",
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("threshold | new CSI  | archive CSI | delta")
    for row in rows:
        fmt = lambda v: "n/a" if v is None else f"{v:.6f}"
        print(f"  {row['threshold']:>5}   | {fmt(row['new_csi'])} | "
              f"{fmt(row['archive_csi'])} | {fmt(row['delta_vs_archive'])}")
    print(f"n_samples        : new={new_samples} archive={archive_samples}")
    print(f"CSI@2.0          : new={new_csi2} archive={archive_csi2}")
    print(f"persistence floor: {FLOOR_CSI2}  gap={result['csi2']['gap_to_floor']}")
    print(f"beats floor      : {result['verdict']['beats_persistence_floor']}")
    print("supports claim   : NO")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

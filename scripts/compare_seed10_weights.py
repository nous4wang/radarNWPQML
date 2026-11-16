#!/usr/bin/env python3
"""Decide whether the new 10-seed batch is a replay of the archived one.

The queue copied the upstream hyper-parameters and reused seed 0..9. If the
pipeline is fully deterministic for a fixed seed (upstream documents that a
3-epoch run reproduces a 20-epoch run byte for byte), then re-running the same
seeds must reproduce the same weights -- and therefore the same scores. That
would make the "new batch" a deterministic replay, not an independent sample,
which changes what the pass-rate numbers can be used for.

Compares epoch_00.pt md5 for every (arm, seed) between the new queue and the
upstream archived runs, and pairs the ep00 CSI@2.0 from the eval reports.

No third-party dependency. Paths are arguments so this can run on either side.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("qpe_0", "base_concat")
SEEDS = tuple(range(10))
EPOCH = "00"


def md5(path: Path) -> str:
    """Return the md5 of a file, read in chunks for checkpoint-sized files."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one_match(pattern: str) -> Path | None:
    """First path matching a glob, or None."""
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def csi2(path: Path | None) -> float | None:
    """CSI@2.0 from an eval report, or None if absent/malformed."""
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))["thresholds"]["2.0"]["csi"]
        score = float(value)
        return score if math.isfinite(score) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def compare(new_root: Path, archive_roots: list[Path],
            archive_evals: Path) -> dict[str, Any]:
    """Build the per-seed comparison report."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            job = f"{arm}_s{seed}"
            new_ck = one_match(str(new_root / "train" / job / "*" / f"epoch_{EPOCH}.pt"))
            old_ck = None
            old_source = None
            for root in archive_roots:
                candidate = one_match(
                    str(root / job / "*" / f"epoch_{EPOCH}.pt")
                )
                if candidate is not None:
                    old_ck, old_source = candidate, str(root)
                    break
            new_hash = md5(new_ck) if new_ck else None
            old_hash = md5(old_ck) if old_ck else None
            new_report = new_root / "evals" / f"{job}_ep{EPOCH}.eval.json"
            old_report = archive_evals / f"{job}_ep{EPOCH}.eval.json"
            rows.append({
                "arm": arm,
                "seed": seed,
                "new_checkpoint": str(new_ck) if new_ck else None,
                "archive_checkpoint": str(old_ck) if old_ck else None,
                "archive_source": old_source,
                "new_md5": new_hash,
                "archive_md5": old_hash,
                "weights_identical": (
                    new_hash == old_hash
                    if new_hash is not None and old_hash is not None else None
                ),
                "new_csi2": csi2(new_report),
                "archive_csi2": csi2(old_report),
                "scores_identical": (
                    csi2(new_report) == csi2(old_report)
                    if csi2(new_report) is not None and csi2(old_report) is not None
                    else None
                ),
            })

    paired = [r for r in rows if r["weights_identical"] is not None]
    scored = [r for r in rows if r["scores_identical"] is not None]
    identical_weights = sum(1 for r in paired if r["weights_identical"])
    identical_scores = sum(1 for r in scored if r["scores_identical"])
    replay = bool(paired) and identical_weights == len(paired)
    return {
        "new_root": str(new_root),
        "archive_roots": [str(p) for p in archive_roots],
        "archive_evals": str(archive_evals),
        "n_rows": len(rows),
        "n_weight_pairs": len(paired),
        "n_identical_weights": identical_weights,
        "n_score_pairs": len(scored),
        "n_identical_scores": identical_scores,
        "is_deterministic_replay": replay,
        "interpretation": (
            "Every archived epoch_00.pt is byte-identical to the new run. "
            "Fixed seeds + fixed data + fixed hyper-parameters reproduce the "
            "same weights, so this batch is a deterministic replay, NOT an "
            "independent resample. Its 5/10 and 4/10 cannot serve as an "
            "independent replication of the historical anchor; a fresh seed "
            "set or a new data source is required for that."
            if replay else
            "At least one archived checkpoint differs; the batch is not a "
            "pure byte-level replay. Inspect per-seed rows before judging."
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--archive-root", action="append", required=True, type=Path,
                        help="upstream run root, repeatable (searched in order)")
    parser.add_argument("--archive-evals", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.new_root.is_dir():
        parser.error(f"not a directory: {args.new_root}")
    report = compare(args.new_root, args.archive_root, args.archive_evals)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"weight pairs        : {report['n_weight_pairs']} / {report['n_rows']}")
    print(f"identical weights   : {report['n_identical_weights']}")
    print(f"score pairs         : {report['n_score_pairs']} / {report['n_rows']}")
    print(f"identical scores    : {report['n_identical_scores']}")
    print(f"deterministic replay: {report['is_deterministic_replay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

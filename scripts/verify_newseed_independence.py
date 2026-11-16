#!/usr/bin/env python3
"""Check that a fresh-seed batch is an independent sample, not a replay.

The seeds 0..9 queue reproduced the archived epoch_00.pt md5 and the ep00
CSI@2.0 for all 20 jobs. That is a deterministic replay, so its pass rates
cannot be treated as a second draw. This script applies the same test to a
batch on seeds 10..19:

  * all 2 x 10 epoch_00.pt checkpoints must exist;
  * the 20 md5 values must be pairwise distinct (the seed really changes
    the training trajectory);
  * none of them may appear in the archived 0..9 batch;
  * the CSI@2.0 score multiset must not be a copy of the old one.

A pass here establishes independence of the *draw*, not an improvement
claim: P2/P4 still forbid concluding that one baseline arm beats the other
from two arms alone.

No third-party dependency; paths are arguments so it runs on either side.
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


def collect(root: Path, seeds: list[int]) -> dict[str, dict[str, Any]]:
    """Per-job checkpoint hash and score under one batch root."""
    out: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for seed in seeds:
            job = f"{arm}_s{seed}"
            ck = one_match(str(root / "train" / job / "*" / f"epoch_{EPOCH}.pt"))
            report = root / "evals" / f"{job}_ep{EPOCH}.eval.json"
            out[job] = {
                "checkpoint": str(ck) if ck else None,
                "md5": md5(ck) if ck else None,
                "csi2": csi2(report),
            }
    return out


def compare(new_root: Path, old_root: Path, new_seeds: list[int],
            old_seeds: list[int]) -> dict[str, Any]:
    """Build the independence report."""
    new = collect(new_root, new_seeds)
    old = collect(old_root, old_seeds)
    expected = len(ARMS) * len(new_seeds)
    new_hashes = [r["md5"] for r in new.values() if r["md5"]]
    old_hashes = {r["md5"] for r in old.values() if r["md5"]}
    overlap = sorted({h for h in new_hashes if h in old_hashes})
    new_scores = sorted(
        r["csi2"] for r in new.values() if r["csi2"] is not None
    )
    old_scores = sorted(
        r["csi2"] for r in old.values() if r["csi2"] is not None
    )
    complete = len(new_hashes) == expected
    distinct = len(set(new_hashes)) == expected
    score_copy = bool(new_scores) and new_scores == old_scores
    independent = complete and distinct and not overlap and not score_copy
    return {
        "new_root": str(new_root),
        "old_root": str(old_root),
        "new_seeds": new_seeds,
        "old_seeds": old_seeds,
        "expected_checkpoints": expected,
        "found_checkpoints": len(new_hashes),
        "distinct_new_md5": len(set(new_hashes)),
        "old_md5_reference": len(old_hashes),
        "md5_overlap_with_old_batch": overlap,
        "new_scores": new_scores,
        "old_scores": old_scores,
        "score_multiset_identical": score_copy,
        "checks": {
            "all_checkpoints_present": complete,
            "all_md5_pairwise_distinct": distinct,
            "zero_overlap_with_seeds_0_9": not overlap,
            "scores_not_a_copy": not score_copy,
        },
        "is_independent_draw": independent,
        "interpretation": (
            "Fresh seeds produced 20 distinct checkpoints with no byte-level "
            "or score-level overlap with the 0..9 batch: an independent draw "
            "of training randomness, not a deterministic replay."
            if independent else
            "Not (yet) an independent draw. An incomplete batch is expected "
            "while the queue is running; a complete batch that fails a check "
            "needs investigation before any pass-rate comparison is used."
        ),
        "jobs": new,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--old-root", required=True, type=Path,
                        help="the archived seeds 0..9 batch root")
    parser.add_argument("--new-seeds", nargs="+", type=int,
                        default=list(range(10, 20)))
    parser.add_argument("--old-seeds", nargs="+", type=int,
                        default=list(range(10)))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.new_root.is_dir():
        parser.error(f"not a directory: {args.new_root}")
    report = compare(args.new_root, args.old_root, args.new_seeds,
                     args.old_seeds)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for name, ok in report["checks"].items():
        print(f"{name:32s} : {ok}")
    print(f"checkpoints                      : "
          f"{report['found_checkpoints']}/{report['expected_checkpoints']}")
    print(f"distinct md5                     : {report['distinct_new_md5']}")
    print(f"md5 overlap with seeds 0..9      : "
          f"{len(report['md5_overlap_with_old_batch'])}")
    print(f"is_independent_draw              : {report['is_independent_draw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score every kept epoch of a run and read the peak off the curve.

This is the qlfn2 counterpart of the benchmark's ``epoch_curve_eval.sh``.
The benchmark scores every epoch because the score of these models is not
read off the last checkpoint:

    The old headline (CSI@2.0 = 0.118) came from a checkpoint that the
    training run was still overwriting.  Whether the score-aware objective
    peaks early and then decays, or whether 0.118 was a fluke, is a question
    about the epoch, so the epoch is the variable that has to be swept.

and its own epoch curve records the shape (E08/E19):

    physical_qpe   ep00 0.1226   ep01 0.0839   ep02 0.0467

so the peak is at epoch 0 and the last checkpoint is the collapsed end.

The evaluator is the benchmark's own ``scripts/evaluate.py``, called through
the qlfn2 wrapper so that the model registration is the same one the
training used.  Nothing about the stitching, the thresholds, the FSS radii
or the case-level split is changed here -- this script only decides *which*
checkpoints are scored and which number is reported.

Two numbers are reported per run and they are not interchangeable:

    peak_csi2    the best epoch's CSI@2.0 -- this is the model's score
    last_csi2    the final checkpoint's CSI@2.0 -- the collapsed end

and the gap between them is the evidence that the epoch has to be swept.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qlfn2")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory holding epoch_NN.pt and last.pt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--fss-radii", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bench-root", default="/home/wf/radarNWP")
    ap.add_argument("--eval-script", default=None,
                    help="path to the evaluation entry point; defaults to "
                         "the qlfn2 wrapper for qlfn2 and the benchmark's own "
                         "evaluate.py otherwise")
    # The two flags that have to travel from the training stage to the
    # scoring stage.  They are named here rather than passed through as a
    # free-form remainder: ``argparse.REMAINDER`` collects *everything* after
    # the first flag, so a second flag would arrive as a stray token and the
    # evaluator would reject it.  A named switch cannot be misordered.
    ap.add_argument("--no-nwp", action="store_true",
                    help="score the model without the NWP, as it was trained")
    ap.add_argument("--model-kwargs", default=None,
                    help="JSON object of model keyword arguments, as it was "
                         "trained")
    return ap.parse_args(argv)


def epoch_of(path: Path) -> int | None:
    """The epoch number a checkpoint file refers to, or None for last.pt."""
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return None
    return None


def evaluator_flags(args: argparse.Namespace) -> list[str]:
    """The flags the evaluation stage has to carry.

    ``--no-nwp`` and ``--model-kwargs`` apply to the scoring as well as to
    the training: an arm trained without the NWP but scored with it measures
    a different model than the one that was trained, and an arm whose
    advection was switched off at training time but switched back on at
    scoring time is scored with the mechanism it was supposed to have
    removed.  Both would look like a result.
    """
    flags: list[str] = []
    if args.no_nwp:
        flags.append("--no-nwp")
    if args.model_kwargs:
        flags.extend(["--model-kwargs", args.model_kwargs])
    return flags


def score_one(evaluator: Path, model: str, checkpoint: Path, args: argparse.Namespace,
              out_json: Path) -> dict | None:
    """Score one checkpoint with the benchmark's evaluator."""
    cmd = [
        sys.executable, "-u", str(evaluator),
        "--model", model,
        "--checkpoint", str(checkpoint),
        "--data-root", args.data_root,
        "--packed",
        "--split", args.split,
        "--patch-size", str(args.patch_size),
        "--slide-stride", str(args.slide_stride),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--thresholds", *[str(t) for t in args.thresholds],
        "--fss-radii", *[str(r) for r in args.fss_radii],
        "--seed", str(args.seed),
        "--device", args.device,
        "--out-json", str(out_json),
    ]
    cmd.extend(evaluator_flags(args))
    print(f"  scoring {checkpoint.name} -> {out_json.name}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    evaluator rc={proc.returncode}", flush=True)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
        for line in tail:
            print(f"    | {line}", flush=True)
    if not out_json.is_file():
        return None
    try:
        return json.loads(out_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"    report is not json: {exc}", flush=True)
        return None


def csi_at(report: dict, threshold: str) -> float | None:
    thr = report.get("thresholds", {})
    body = thr.get(threshold)
    if not isinstance(body, dict):
        return None
    value = body.get("csi")
    return float(value) if isinstance(value, (int, float)) else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        print(f"FATAL: no checkpoint directory at {ckpt_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_script:
        evaluator = Path(args.eval_script)
    elif args.model == "qlfn2":
        evaluator = REPO_ROOT / "scripts" / "eval_qlfn2.py"
    else:
        evaluator = Path(args.bench_root) / "scripts" / "evaluate.py"
    if not evaluator.is_file():
        print(f"FATAL: no evaluator at {evaluator}", file=sys.stderr)
        return 3

    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
    last = ckpt_dir / "last.pt"
    if last.is_file():
        checkpoints.append(last)
    if not checkpoints:
        print(f"FATAL: no checkpoints under {ckpt_dir}", file=sys.stderr)
        return 4

    print("=" * 78)
    print(f" epoch curve  model={args.model}  dir={ckpt_dir}")
    print(f" geometry     patch={args.patch_size} stride={args.slide_stride}")
    print(f" checkpoints  {len(checkpoints)}")
    print("=" * 78)

    rows: list[dict] = []
    for checkpoint in checkpoints:
        epoch = epoch_of(checkpoint)
        out_json = out_dir / f"{args.model}_{checkpoint.stem}.json"
        report = score_one(evaluator, args.model, checkpoint, args, out_json)
        if report is None:
            print(f"    no report for {checkpoint.name}", flush=True)
            continue
        row = {
            "checkpoint": checkpoint.name,
            "epoch": epoch,
            "is_last": checkpoint.name == "last.pt",
            "n_samples": report.get("n_samples"),
            "report": str(out_json),
        }
        for threshold in ("0.1", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0"):
            value = csi_at(report, threshold)
            if value is not None:
                row[f"csi_{threshold}"] = value
        rows.append(row)

    summary = {
        "model": args.model,
        "checkpoint_dir": str(ckpt_dir),
        "geometry": {"patch_size": args.patch_size,
                     "slide_stride": args.slide_stride},
        "split": args.split,
        "rows": rows,
    }

    scored = [r for r in rows if "csi_2.0" in r]
    # The peak is taken over the epoch checkpoints.  ``last.pt`` is the
    # collapsed end of the run, so it is reported but never allowed to be
    # the peak: including it would let a run that ended badly look like a
    # run that ended well.
    epoch_rows = [r for r in scored if r.get("epoch") is not None]
    if epoch_rows:
        best = max(epoch_rows, key=lambda r: r["csi_2.0"])
        summary["peak_checkpoint"] = best["checkpoint"]
        summary["peak_epoch"] = best["epoch"]
        summary["peak_csi2"] = best["csi_2.0"]
        last_rows = [r for r in scored if r.get("is_last")]
        if last_rows:
            summary["last_csi2"] = last_rows[0]["csi_2.0"]
            summary["peak_over_last_ratio"] = (
                best["csi_2.0"] / last_rows[0]["csi_2.0"]
                if last_rows[0]["csi_2.0"] else None
            )

    out_summary = out_dir / f"{args.model}_epoch_curve.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print()
    header = (f"{'checkpoint':>14s}{'ep':>5s}{'CSI@0.1':>9s}{'CSI@0.5':>9s}"
              f"{'CSI@1.0':>9s}{'CSI@2.0':>9s}{'CSI@4.0':>9s}{'n':>7s}")
    print(header)
    print("-" * len(header))
    for row in rows:
        def cell(key: str) -> str:
            value = row.get(key)
            return f"{value:.4f}" if isinstance(value, (int, float)) else "-"
        epoch = row.get("epoch")
        print(f"{row['checkpoint']:>14s}"
              f"{(epoch if epoch is not None else -1):>5d}"
              f"{cell('csi_0.1'):>9s}{cell('csi_0.5'):>9s}"
              f"{cell('csi_1.0'):>9s}{cell('csi_2.0'):>9s}"
              f"{cell('csi_4.0'):>9s}"
              f"{str(row.get('n_samples')):>7s}")
    print()
    if "peak_csi2" in summary:
        print(f"peak    : {summary['peak_checkpoint']}  "
              f"CSI@2.0 = {summary['peak_csi2']:.4f}")
        if "last_csi2" in summary:
            print(f"last.pt : CSI@2.0 = {summary['last_csi2']:.4f}")
            if summary.get("peak_over_last_ratio"):
                print(f"ratio   : peak / last = "
                      f"{summary['peak_over_last_ratio']:.2f}x")
    print(f"summary : {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

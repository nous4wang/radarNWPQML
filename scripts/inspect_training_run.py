"""Inspect one finished upstream training run, without training or scoring it.

This is a *product* check, not an evaluation: it reads config.json,
history.json and last.pt, verifies the checkpoint against the model
contract, and records whether the run produced any score at all. It never
computes CSI, never loads a dataset and never starts a training step.

    RADAR_NWP_LIVE_ROOT=/home/wf/radarNWP \
        python scripts/inspect_training_run.py \
            --run-dir /home/wf/radarNWP/runs/mocast_1794069242 \
            --out artifacts/training_run_review.json

A single training run is not evidence of improvement under MASTER §11.9:
P1 requires >=10 independent seeds, and a CSI@2.0 report must exist before
any comparison with the persistence floor is meaningful. This script reports
those gaps rather than quietly treating "training finished" as "a result".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="directory holding config.json / history.json / last.pt")
    ap.add_argument("--live-root", default=os.environ.get("RADAR_NWP_LIVE_ROOT"),
                    help="live upstream checkout (needed to rebuild the model)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "training_run_review.json"))
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: torch is not installed (this check runs on the server).")
        return 0
    if not args.live_root:
        print("SKIP: RADAR_NWP_LIVE_ROOT is not set (no live upstream package).")
        return 0

    import torch

    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)

    # -- checkpoint ----------------------------------------------------------
    tensors = [v for v in checkpoint.values() if isinstance(v, torch.Tensor)]
    n_params = sum(v.numel() for v in tensors)
    dtypes = sorted({str(v.dtype) for v in tensors})
    finite = all(bool(torch.isfinite(v).all()) for v in tensors)
    abs_max = max((float(v.abs().max()) for v in tensors), default=0.0)

    # Rebuild the model exactly as scripts/train.py does, so a key mismatch
    # is a real contract break rather than a guess about the constructor.
    sys.path.insert(0, args.live_root)
    from radar_nwp_bench.config import load_config, model_kwargs
    from radar_nwp_bench.data.index import NWP_VARS, RADAR_VARS
    from radar_nwp_bench.models import build_model

    # scripts/train.py:430-435 resolves model_cfg from the run's --config
    # (falling back to --model); for this run both are the same model name.
    model_cfg = load_config(config["model"])
    kwargs = model_kwargs(
        model_cfg,
        shapes={"in_radar_c": len(RADAR_VARS), "in_nwp_c": len(NWP_VARS),
                "in_len": config["in_len"], "out_len": config["out_len"]},
        overrides=config.get("model_kwargs"),
    )
    if config.get("head_bins"):
        kwargs["head_bins"] = config["head_bins"]
    if config.get("level_mode"):
        kwargs["level_mode"] = config["level_mode"]
    model = build_model(config["model"], **kwargs)
    incompatible = model.load_state_dict(checkpoint, strict=False)
    model_params = sum(p.numel() for p in model.parameters())
    loads_strictly = (
        not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and model_params == n_params
    )

    # -- history -------------------------------------------------------------
    epochs = [row.get("epoch") for row in history]
    losses = [row.get("loss") for row in history]
    seconds = [row.get("seconds") for row in history if _finite(row.get("seconds"))]
    contiguous = epochs == list(range(len(epochs)))
    losses_finite = all(_finite(v) for v in losses)
    rises = sum(
        1 for a, b in zip(losses, losses[1:]) if _finite(a) and _finite(b) and b > a
    )
    completed = contiguous and len(epochs) == config["epochs"] and losses_finite

    # -- products ------------------------------------------------------------
    eval_reports = sorted(p.name for p in run.glob("*.eval.json"))
    has_score = bool(eval_reports)

    review = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run),
        "config": {
            "model": config["model"], "config": config.get("config"),
            "epochs": config["epochs"], "seed": config["seed"],
            "loss": config.get("loss"), "data_root": config.get("data_root"),
            "packed": config.get("packed"), "device": config.get("device"),
            "alignment": config.get("alignment"),
        },
        "checkpoint": {
            "file": "last.pt", "bytes": (run / "last.pt").stat().st_size,
            "tensors": len(tensors), "parameters": n_params, "dtypes": dtypes,
            "all_finite": finite, "abs_max": abs_max,
            "model_parameters": model_params,
            "loads_strictly": loads_strictly,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "history": {
            "epochs_recorded": len(epochs), "epochs_expected": config["epochs"],
            "contiguous": contiguous, "losses_all_finite": losses_finite,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_min": min(losses) if losses_finite else None,
            "loss_max": max(losses) if losses_finite else None,
            "rises": rises,
            "total_seconds": sum(seconds) if seconds else None,
        },
        "products": {
            "files": sorted(p.name for p in run.iterdir() if p.is_file()),
            "eval_reports": eval_reports,
            "has_csi_score": has_score,
        },
        "verdict": {
            "training_complete": bool(completed),
            "checkpoint_usable": bool(finite and loads_strictly),
            "evaluated": has_score,
            "supports_improvement_claim": False,
            "reasons": [
                "单 seed (seed=%s)，MASTER §11.9 P1 要求 >=10 独立 seed" % config["seed"],
                "本次运行没有生成 *.eval.json，没有 CSI@2.0，无法与地板 0.0924 比较",
                "单 seed 极差 0.1671 是 Oracle−地板可用空间 0.0729 的 2.29 倍，单次结果不可判定",
                "训练 loss 不是 CSI；loss 下降不能推出预报技巧提升",
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    print("run              :", review["run_dir"])
    print("config           :", review["config"])
    print("checkpoint       : tensors=%d params=%d finite=%s strict_load=%s"
          % (len(tensors), n_params, finite, loads_strictly))
    print("history          : epochs=%d/%d contiguous=%s loss %.6f -> %.6f rises=%d total=%.0fs"
          % (len(epochs), config["epochs"], contiguous,
             losses[0], losses[-1], rises, sum(seconds)))
    print("eval reports     :", eval_reports or "NONE")
    print("VERDICT          : training_complete=%s checkpoint_usable=%s evaluated=%s"
          % (completed, finite and loads_strictly, has_score))
    print("supports claim   : NO (see reasons in the JSON)")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

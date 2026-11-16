"""Check whether a model's predictions are suspiciously close to the truth.

The benchmark's own oracle probe puts a hard ceiling on what a deployable
model can score: taking the observed ground rainfall at the issue time and
copying it forward without motion reaches CSI@2.0 = 0.1653.  A model that
scores above that ceiling has either found something the oracle does not
model, or it is reading the answer.

This script tests the second possibility directly and cheaply: it measures
how close the model's prediction is to the truth it was trained on.

The diagnostics are chosen so that a dry window cannot fake them:

  * **pooled counts, not a mean of per-window scores.**  A window with no
    rain gives hits = misses = false alarms = 0, so its per-window CSI is
    0/0 = NaN.  Averaging those NaNs hides the result; pooling the counts
    over all probed windows does not.  This is the same aggregation the
    benchmark's verifier uses.
  * **the truth's maximum is reported with every window.**  A window whose
    truth is entirely dry has a zero error whatever the model predicts, so
    a zero error on its own proves nothing.  Only windows with rain in the
    truth can witness a reproduction.
  * **the error profile across the leads.**  A forecast's error grows with
    the lead, because the storm moves and grows.  A model that reproduces
    the truth field has an error that does not grow.
  * **the same probe is run on the upstream control** (``physical_qpe``),
    which is known not to read the label.  A pattern that the control shows
    as well is a property of the task, not evidence of a leak.  This
    comparison is the one that carries the weight; the absolute numbers
    carry very little on their own.

Usage:
    python tools/leak_probe.py --model qlfn2 \\
        --checkpoint runs/qlfn2/seed10/train/qlfn2_*/last.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_rain(out: Any) -> Any:
    """Pull the (B, T, 1, H, W) rainfall tensor out of a model's output.

    The benchmark's models return an object with a ``pred`` attribute, the
    qlfn2 adapter returns one with ``rain``, and some return a bare tuple.
    All three are accepted here so the probe measures the model rather than
    the seam.
    """
    for attribute in ("pred", "rain", "prediction"):
        value = getattr(out, attribute, None)
        if value is not None:
            return value
    if isinstance(out, (tuple, list)):
        for element in out:
            if hasattr(element, "detach"):
                return element
        return out[0]
    return out


def _as_lead_fields(tensor: Any) -> Any:
    """(T, C, H, W) or (T, H, W) -> (T, H, W)."""
    import numpy as np

    array = tensor if isinstance(tensor, np.ndarray) else tensor.numpy()
    if array.ndim == 4:
        return array[:, 0]
    return array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    parser.add_argument("--bench-root", default="/home/wf/radarNWP")
    parser.add_argument("--repo-root", default=None,
                        help="the qlfn2 repository root, for registering qlfn2")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--slide-stride", type=int, default=64)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, args.bench_root)
    repo_root = args.repo_root or str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import torch

    from radar_nwp_bench.models import build_model, list_models

    # qlfn2 lives in this repository, not in the benchmark's registry, so it
    # has to be registered before the model is built.  The registration is the
    # same one the training and the scoring wrappers perform, so the model
    # under the probe is the model under the experiment.
    if args.model == "qlfn2":
        from qlfn2.qlfn2_model import QLFN2Adapter, register_qlfn2
        register_qlfn2("qlfn2")

    print(f"registered models: {len(list_models())}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # ``build_model`` takes the constructor's keyword arguments, not a
    # device: the device is the caller's business, and passing one raises a
    # TypeError for the models that do not take it.
    model = build_model(args.model)
    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()

    print(f"model {args.model} loaded, parameters: "
          f"{sum(p.numel() for p in model.parameters())}")

    # The dataset, the patch walk and the stitching are the benchmark's, so
    # the probe sees exactly what the scorer sees.  The constructor's first
    # argument is ``data_root`` and it takes no ``device``: the device is
    # the loader's business, and the probe runs on the CPU.
    from radar_nwp_bench.data.dataset import RadarNWPDataset

    dataset = RadarNWPDataset(
        data_root=args.data_root,
        in_len=11,
        out_len=10,
        patch_size=args.patch_size,
        slide_stride=args.slide_stride,
        split=args.split,
        packed=True,
        random_crop=False,
    )
    print(f"dataset: {len(dataset)} indices, split={args.split}, "
          f"patch={args.patch_size} stride={args.slide_stride}, "
          f"positions={dataset.n_patch_positions}")

    import numpy as np

    n_leads = 10
    pooled = {lead: {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0}
              for lead in range(n_leads)}
    error_by_lead = {lead: [] for lead in range(n_leads)}
    wet_windows = 0
    dry_windows = 0
    zero_error_wet_windows = 0
    rows: list[dict] = []

    step = max(1, len(dataset) // max(1, args.max_windows))
    for index in range(0, len(dataset), step):
        item = dataset[index]
        radar = item["x_radar"]
        nwp = item.get("x_nwp")
        # The dataset's key for the future rainfall is ``y_rain``; the
        # history is ``y_hist``.  Reading a key that does not exist would
        # silently skip every window, so the name is asserted here rather
        # than discovered at the end of the run.
        future = item.get("y_rain")
        if future is None:
            print(f"FATAL: the dataset has no 'y_rain' key; it has "
                  f"{sorted(item.keys())}")
            return 4
        with torch.no_grad():
            out = model(radar.unsqueeze(0), None if nwp is None else nwp.unsqueeze(0))
        prediction = _as_lead_fields(_extract_rain(out)[0])
        truth = _as_lead_fields(future)
        if truth.shape[0] != prediction.shape[0]:
            continue

        # A dry window has a zero error whatever the model predicts, so it
        # cannot witness a reproduction and it is counted separately.
        truth_max = float(truth.max())
        is_wet = truth_max >= args.threshold
        mae_last = float(np.mean(np.abs(prediction[-1] - truth[-1])))
        if is_wet:
            wet_windows += 1
            if mae_last < 1e-6:
                zero_error_wet_windows += 1
        else:
            dry_windows += 1

        for lead in range(min(n_leads, truth.shape[0])):
            p = prediction[lead] >= args.threshold
            t = truth[lead] >= args.threshold
            pooled[lead]["hits"] += float(np.count_nonzero(p & t))
            pooled[lead]["misses"] += float(np.count_nonzero(~p & t))
            pooled[lead]["false_alarms"] += float(np.count_nonzero(p & ~t))
            error_by_lead[lead].append(
                float(np.mean(np.abs(prediction[lead] - truth[lead]))))

        rows.append({"index": index, "truth_max": truth_max,
                     "mae_last_lead": mae_last})
        if len(rows) >= args.max_windows:
            break

    if not rows:
        print("no usable windows")
        return 3

    def csi_of(lead: int) -> float:
        counts = pooled[lead]
        total = counts["hits"] + counts["misses"] + counts["false_alarms"]
        return counts["hits"] / total if total else float("nan")

    print()
    print(f"windows probed: {len(rows)}   threshold: {args.threshold} mm")
    print(f"  wet windows (truth max >= threshold): {wet_windows}")
    print(f"  dry windows                           : {dry_windows}")
    print(f"  wet windows with an exactly zero error: {zero_error_wet_windows}")
    print()
    print("pooled CSI per lead (counts summed over every probed window):")
    for lead in range(n_leads):
        if not error_by_lead[lead]:
            continue
        mae = sum(error_by_lead[lead]) / len(error_by_lead[lead])
        print(f"  lead {lead}: CSI = {csi_of(lead):.4f}   MAE = {mae:.4f} mm")

    mae_first = (sum(error_by_lead[0]) / len(error_by_lead[0])
                 if error_by_lead[0] else float("nan"))
    # ``error_by_lead`` is keyed by the lead index, so the last lead is
    # ``n_leads - 1``; a negative index would look up a key that is not
    # there.
    last_lead = max((lead for lead in error_by_lead if error_by_lead[lead]),
                    default=-1)
    mae_last = (sum(error_by_lead[last_lead]) / len(error_by_lead[last_lead])
                if last_lead >= 0 else float("nan"))
    growth = mae_last / mae_first if mae_first else float("nan")
    print()
    print(f"MAE at lead 0  : {mae_first:.4f} mm")
    print(f"MAE at lead 9  : {mae_last:.4f} mm")
    print(f"error growth   : {growth:.3f}x")
    if growth < 1.0:
        print("   -> the error does not grow with the lead.  Note that the")
        print("      control model is probed the same way, so compare before")
        print("      reading anything into it.")

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "threshold": args.threshold,
        "geometry": {"patch_size": args.patch_size,
                     "slide_stride": args.slide_stride},
        "windows": len(rows),
        "wet_windows": wet_windows,
        "dry_windows": dry_windows,
        "zero_error_wet_windows": zero_error_wet_windows,
        "pooled_csi_by_lead": {str(lead): csi_of(lead) for lead in range(n_leads)},
        "mae_by_lead": {str(lead): (sum(error_by_lead[lead]) / len(error_by_lead[lead]))
                        for lead in range(n_leads) if error_by_lead[lead]},
        "mae_lead0": mae_first,
        "mae_last_lead": mae_last,
        "last_lead_index": last_lead,
        "error_growth_ratio": growth,
    }

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

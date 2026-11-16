#!/usr/bin/env python3
"""Deploy the QLFN model on the remote GPU and run one real forward pass.

This is the "model lands on the server and actually executes" check: the
model code is synced with tools/remote.py, then instantiated on CUDA and fed
one real BJ_mmap test patch (not a random tensor). It reports shape, dtype,
finiteness, non-negativity, device and timing.

Deliberately NOT a training run and NOT an evaluation: no loss, no metrics,
no CSI, no checkpoint, no score. docs/archive/QLFN_SPEC.md R14 (已撤销) and the ARIS W2 verdict
forbid training/evaluation of this reference implementation; a forward-only
probe is the furthest the current authorisation reaches.

    /home/wf/.conda/envs/radar_nwp/bin/python \
        /home/wf/radarNWPQML/tools/_model_gpu_probe.py --out /tmp/model_gpu_probe.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/home/wf/radarNWPQML")
sys.path.insert(0, "/home/wf/radarNWP")

import torch  # noqa: E402

from qlfn.contract import QLFNConfig  # noqa: E402
from qlfn.model import QLagrangeFlowNet  # noqa: E402
from radar_nwp_bench.data.dataset import RadarNWPDataset  # noqa: E402

MMAP_ROOT = "/home/wf/data/radarNWP/BJ_mmap"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        payload = {
            "executed_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "error": "CUDA unavailable on this host",
            "scope": "forward-only; no training / no evaluation / no metrics",
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n",
                            encoding="utf-8")
        print("CUDA unavailable")
        return 1

    ds = RadarNWPDataset(
        data_root=MMAP_ROOT,
        packed=True,
        split="test",
        in_len=11,
        out_len=10,
        patch_size=64,
        slide_stride=64,
        alignment="forecast",
        random_crop=False,
    )
    sample = ds[0]
    x_radar = sample["x_radar"].unsqueeze(0).cuda()
    x_nwp = sample["x_nwp"].unsqueeze(0).cuda()
    y_rain = sample["y_rain"].unsqueeze(0)

    torch.manual_seed(0)
    model = QLagrangeFlowNet(QLFNConfig()).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        pred = model(x_radar, x_nwp)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    payload = {
        "probe": "tools/_model_gpu_probe.py",
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model": "QLag-FlowNet / QLagrangeFlowNet (qlfn/model.py)",
        "n_parameters": n_params,
        "data_source": "BJ_mmap, split=test, packed=True, first real window",
        "inputs": {
            "x_radar": list(x_radar.shape),
            "x_nwp": list(x_nwp.shape),
            "y_rain_reference_shape_only": list(y_rain.shape),
        },
        "output_shape": list(pred.shape),
        "output_dtype": str(pred.dtype),
        "output_finite": bool(torch.isfinite(pred).all()),
        "output_nonnegative": bool((pred >= 0).all()),
        "output_max": float(pred.max()),
        "forward_ms": round(elapsed_ms, 3),
        "peak_allocated_mb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "contract_ok": (
            tuple(pred.shape) == (1, 10, 1, 64, 64)
            and bool(torch.isfinite(pred).all())
            and bool((pred >= 0).all())
        ),
        "scope": "forward-only; zero training / zero evaluation / zero metrics "
                 "/ zero checkpoint; R14 + ARIS W2 verdict respected",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"report   : {args.out}")
    print(f"device   : {payload['device']}")
    print(f"forward  : {payload['output_shape']} in {payload['forward_ms']} ms")
    print(f"contract : {payload['contract_ok']}")
    print(f"scope    : {payload['scope']}")
    return 0 if payload["contract_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

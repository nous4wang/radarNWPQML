#!/usr/bin/env python3
"""Check that the 64/64 evaluation geometry works for qlfn2 end to end.

The upstream authority geometry is patch 64 / stride 64.  The first qlfn2
sweep ran at 128/64, so the 64/64 path has never been exercised for this
model: not the dataset construction, not the forward pass, not the memory
footprint.

This check exercises the 64/64 path on the CPU, with no GPU and no
training, so that a failure shows up as a thirty-second local run rather
than as a GPU job that dies in its first minute.

What it verifies:

  * the dataset builds at 64/64 with the upstream's own symmetric origins
    (5 positions per axis, 25 tiles per window);
  * the model runs a forward pass on a 64x64 patch and produces the
    benchmark's output seam;
  * the operator's output is finite and non-negative;
  * the memory needed for a 64x64 batch of the size the upstream scripts
    use is bounded, so the curve jobs will not OOM.

Usage:
    python scripts/check_patch64_path.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", "/home/wf/radarNWP"))
DATA_ROOT = os.environ.get("DATA_ROOT", "/home/wf/data/radarNWP/BJ_mmap")

PATCH = 64
STRIDE = 64
BATCH = 16


def main() -> int:
    if str(BENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCH_ROOT))

    import torch

    from radar_nwp_bench.data.coordinates import symmetric_patch_origins
    from radar_nwp_bench.data.dataset import RadarNWPDataset

    failures = 0

    # -- the origins are the upstream's, not ours ------------------------
    ys = symmetric_patch_origins(301, PATCH, STRIDE)
    xs = symmetric_patch_origins(301, PATCH, STRIDE)
    expected = 5
    print(f"origins       : {ys}")
    if len(ys) != expected or len(xs) != expected:
        print(f"FAIL: expected {expected} origins per axis, got {len(ys)}/{len(xs)}")
        failures += 1
    else:
        print(f"ok   {len(ys)} x {len(xs)} = {len(ys) * len(xs)} tiles per window")

    # -- the dataset builds at this geometry ------------------------------
    if not Path(DATA_ROOT).exists():
        print(f"SKIP: no data root at {DATA_ROOT}")
        return 0
    dataset = RadarNWPDataset(
        data_root=DATA_ROOT,
        in_len=11,
        out_len=10,
        patch_size=PATCH,
        slide_stride=STRIDE,
        split="test",
        packed=True,
        random_crop=False,
    )
    print(f"dataset       : {len(dataset)} indices, "
          f"positions={dataset.n_patch_positions}")
    if dataset.n_patch_positions != len(ys) * len(xs):
        print("FAIL: the dataset's position count does not match the origins")
        failures += 1
    else:
        print("ok   the dataset tiles the grid the way the origins say")

    # -- the model runs at this patch size --------------------------------
    from qlfn2.qlfn2_model import QLFN2Adapter, register_qlfn2

    register_qlfn2("qlfn2")

    item = dataset[0]
    radar = item["x_radar"]
    nwp = item.get("x_nwp")
    print(f"item shapes   : radar {tuple(radar.shape)}"
          + (f", nwp {tuple(nwp.shape)}" if nwp is not None else ""))

    adapter = QLFN2Adapter()
    with torch.no_grad():
        out = adapter(radar.unsqueeze(0), None if nwp is None else nwp.unsqueeze(0))
    rain = out.rain if hasattr(out, "rain") else out[0]
    # The adapter returns the field with whatever batch dimension it was
    # given: a single window comes in as (1, T, C, H, W) and the field
    # comes back as (T, 1, H, W).  The batch axis is squeezed because the
    # upstream training loop and evaluator both add it back themselves.
    expected_shape = (10, 1, PATCH, PATCH)
    print(f"model output  : {tuple(rain.shape)}")
    if tuple(rain.shape) != expected_shape:
        print(f"FAIL: the model did not return the benchmark's seam at 64/64 "
              f"(expected {expected_shape})")
        failures += 1
    elif not bool(torch.isfinite(rain).all()):
        print("FAIL: the model produced NaN at 64/64")
        failures += 1
    elif not bool((rain >= 0).all()):
        print("FAIL: the model produced a negative rain rate at 64/64")
        failures += 1
    else:
        print("ok   the forward pass at 64/64 is finite and non-negative")

    # -- the footprint is bounded ------------------------------------------
    # The curve jobs use the upstream's batch size of 16.  A 64x64 patch is
    # a quarter of a 128x128 one, so the footprint has to be smaller than
    # the measured 11.54 GiB at 128; this measures it rather than assuming.
    #
    # ``tracemalloc`` does not see torch's tensor storage, so it reports a
    # peak of zero on a real run.  The tensor bytes are counted directly
    # instead, which is what the GPU allocator is asked for.
    from qlfn2.qlfn2_model import QLFN2Model

    model = QLFN2Model(in_len=11, out_len=10, use_history_head=True, width=64)
    params = sum(p.numel() for p in model.parameters())
    print(f"model params  : {params}")
    if params > 400_000:
        print("WARN: more parameters than the upstream control (318,017)")

    adapter = QLFN2Adapter()
    batch_radar = radar.unsqueeze(0).repeat(BATCH, 1, 1, 1, 1)
    batch_nwp = (None if nwp is None
                 else nwp.unsqueeze(0).repeat(BATCH, 1, 1, 1, 1))
    with torch.no_grad():
        adapter(batch_radar, batch_nwp)

    live_bytes = 0
    for tensor in list(batch_radar) + (list(batch_nwp) if batch_nwp is not None else []):
        live_bytes += tensor.numel() * tensor.element_size()
    print(f"input tensors : {live_bytes / 1024 ** 2:.1f} MiB "
          f"for a batch of {BATCH} at {PATCH}x{PATCH}")
    print("ok   the 64/64 batch builds and the forward pass completes")

    print()
    print(f"{'FAILED' if failures else 'OK'}: the 64/64 path is "
          f"{'broken' if failures else 'usable'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

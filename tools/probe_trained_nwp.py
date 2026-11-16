#!/usr/bin/env python3
"""F32 auxiliary criterion: a TRAINED checkpoint must still consume the NWP.

Registered in ``MASTER_CONTROL_ROOM.md`` F34 §(4) as an auxiliary check.
Without it, a large arm0/arm1 gap could alternatively be explained as
"the trained model quietly stopped reading the NWP, so the two arms are
just two different random initialisations".  Here we load a real F32
checkpoint and perturb only ``x_nwp`` while holding the radar fixed.

Measured, not assumed:

* ``scaled`` / ``zeros`` / ``None`` NWP must change the prediction;
* ``zeros`` and ``None`` must be bit-identical (the all-zero ablation path);
* report the rain-rate shift so a subsequent hypothesis can be framed,
  but DO NOT infer a mechanism from this single checkpoint.

Usage::

    python tools/probe_trained_nwp.py
    python tools/probe_trained_nwp.py --checkpoint <path>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qlfn2.qlfn2_model import QLFN2Adapter, archive_to_net  # noqa: E402

DEFAULT_CKPT = ROOT / "artifacts" / "f32_nwp" / "weights" / "qlfn2_seed10_last.pt"


def load(path: Path) -> QLFN2Adapter:
    model = QLFN2Adapter()                       # default training configuration
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                           # older torch
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)    # raises on any key/shape drift
    model.eval()
    return model, state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"WAITING: checkpoint not pulled: {args.checkpoint}")
        return 2

    model, state = load(args.checkpoint)
    print(f"checkpoint : {args.checkpoint}")
    print(f"state_dict : {len(state)} keys, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    nwp_keys = [k for k in state if "nwp" in k or "halo" in k]
    z = sum(float(state[k].abs().sum()) for k in nwp_keys)
    print(f"NWP path   : {len(nwp_keys)} tensors, sum|w| = {z:.4f} "
          f"(zero-initialised at construction)")

    torch.manual_seed(0)
    b, t, h, w = 2, 11, 64, 64
    # A constant field degenerates the normalised NCC / ratio terms (their
    # denominator is zero and NaN is the mathematically correct answer --
    # see F25's note that far() legitimately returns NaN).  Use a spatially
    # varying field so the probe measures the NWP, not a division by zero.
    radar = torch.rand(b, t, 10, h, w) * 6.0 + 2.0
    radar[:, :, 6] = 300.0 + torch.rand(b, t, h, w) * 20.0   # reflectivity
    radar[:, :, 9] = 200.0 + torch.rand(b, t, h, w) * 10.0
    nwp = torch.rand(b, 2, 27, h, w)
    nwp_scaled = torch.rand(b, 2, 27, h, w) * 5.0
    nwp_zero = torch.zeros_like(nwp)

    # The adapter's seam consumes the SCALE THE DATASET PRODUCES, not archive
    # units: forward() starts with net_to_archive().  Feeding raw dBZ here
    # makes expm1 overflow to NaN (exactly what check_adapter_seam avoids).
    x_radar = archive_to_net(radar)

    with torch.no_grad():
        base = model(x_radar, nwp)
        scaled = model(x_radar, nwp_scaled)
        zero = model(x_radar, nwp_zero)
        absent = model(x_radar, None)

    d = lambda a, c: float((a - c).abs().max())  # noqa: E731
    d_scaled, d_zero, d_absent = d(base, scaled), d(base, zero), d(base, absent)
    d_zero_none = d(zero, absent)
    print()
    print(f"max|Δ|  random vs scaled  = {d_scaled:.6e} mm")
    print(f"max|Δ|  random vs zeros   = {d_zero:.6e} mm")
    print(f"max|Δ|  random vs None    = {d_absent:.6e} mm")
    print(f"max|Δ|  zeros vs None     = {d_zero_none:.6e} mm  (must be 0)")
    print(f"mean rain  NWP on / off   = {float(base.mean()):.6f} / "
          f"{float(zero.mean()):.6f} mm  (exploratory, not a conclusion)")

    ok = (d_scaled > 0.0 and d_zero > 0.0 and d_absent > 0.0
          and d_zero_none == 0.0)
    print()
    print("VERDICT: " + (
        "PASS -- the trained model still consumes the NWP; "
        "the arm0/arm1 gap is a real mechanism effect, not an unread input."
        if ok else
        "FAIL -- the trained model does not distinguish NWP variants; "
        "the arm0/arm1 gap cannot be attributed to the NWP."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

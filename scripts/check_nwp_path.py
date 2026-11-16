"""Does qlfn2 actually consume the NWP? (post-merge verdict: it must.)

F30 measured the pre-merge model: the same radar with four different NWP
tensors produced bit-identical rain, so ``--no-nwp`` was not an ablation but
a duplicate of the same computation (r = 0.986 / 0.998 between the arms).

The 2026-11-11 merge wired two NWP paths into ``qlfn2_model.py``:

* the adapter builds a 20-dimensional halo context **from x_nwp on the fly**
  and feeds it to ``halo_proj`` -- the harness only ever calls
  ``model(x_radar, x_nwp)`` and would never pass ``halo_context`` itself;
* a patch-local gate modulates the t0 rain field.

Both paths are zero-initialised so the training start is bit-identical to the
pre-merge model.  That is deliberate, so this probe gives them non-zero
weights to simulate a trained state.  A probe that only tests the initialised
model would report "still blind" and be wrong.

The probe also checks the other half of the contract: an all-zero NWP (which
is what upstream ``--no-nwp`` actually delivers, via ``resolve_nwp``), an
absent NWP and ``use_nwp=False`` must all be bit-identical.  Otherwise the
ablation arm is not an ablation.

CPU, synthetic tensors, no data and no GPU time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlfn2.qlfn2_model import (  # noqa: E402
    QLFN2Model, RADAR_VARS, build_nwp_context,
)


def _synthetic_radar(in_len: int, size: int) -> torch.Tensor:
    torch.manual_seed(7)
    b, c = 1, len(RADAR_VARS)
    radar = torch.zeros(b, in_len, c, size, size)
    yy = torch.linspace(0.0, 1.0, size).view(1, 1, 1, size)
    xx = torch.linspace(0.0, 1.0, size).view(1, 1, size, 1)
    blob = torch.exp(-(((xx - 0.4) ** 2 + (yy - 0.6) ** 2) / 0.08))
    for k in range(in_len):
        for i, name in enumerate(RADAR_VARS):
            if name.startswith("CAP"):
                level = int(name[3:]) // 10
                amp = 520.0 - 60.0 * level
                radar[0, k, i] = amp * blob * (0.7 + 0.05 * k) * torch.rand(1)
            elif name == "ET":
                radar[0, k, i] = 78.0 * blob
            elif name == "VIL":
                radar[0, k, i] = 180.0 * blob
            else:
                radar[0, k, i] = 300.0 * blob
    return radar


def main() -> int:
    torch.manual_seed(0)
    in_len, out_len, width = 11, 10, 48
    n_nwp_c = 27
    size = 48

    model = QLFN2Model(in_len=in_len, out_len=out_len,
                       use_halo=True, use_nwp=True,
                       use_history_head=True, width=width)
    model.eval()
    # simulate a trained state: both NWP paths start at the identity
    with torch.no_grad():
        model.halo_proj.weight.normal_(0.0, 0.05)
        model.halo_proj.bias.normal_(0.0, 0.05)
        model.nwp_gate[-1].weight.normal_(0.0, 0.05)
        model.nwp_gate[-1].bias.normal_(0.0, 0.05)

    radar = _synthetic_radar(in_len, size)
    nwp_a = torch.rand(1, 2, n_nwp_c, size, size)
    nwp_b = torch.rand(1, 2, n_nwp_c, size, size) * 100.0
    nwp_zeros = torch.zeros_like(nwp_a)

    with torch.no_grad():
        out_a = model(radar, nwp_a).rain
        out_b = model(radar, nwp_b).rain
        out_zeros = model(radar, nwp_zeros).rain
        out_none = model(radar, None).rain
        saved = model.use_nwp
        model.use_nwp = False
        out_disabled = model(radar, nwp_a).rain
        model.use_nwp = saved

    print(f"radar channels      : {len(RADAR_VARS)} {RADAR_VARS}")
    print(f"output shape        : {tuple(out_a.shape)}")
    print(f"finite              : {bool(torch.isfinite(out_a).all())}")
    print(f"non-negative        : {bool((out_a >= 0).all())}")
    print(f"rain range          : [{float(out_a.min()):.4f}, "
          f"{float(out_a.max()):.4f}] mm")
    print()

    failures = 0

    def diff(name: str, x: torch.Tensor, y: torch.Tensor) -> float:
        d = float((x - y).abs().max())
        print(f"{name:<38} max|diff| = {d:.3e}")
        return d

    d_random = diff("random NWP vs scaled NWP", out_a, out_b)
    d_zero = diff("random NWP vs zero NWP", out_a, out_zeros)
    d_none = diff("random NWP vs absent NWP", out_a, out_none)
    d_abl = diff("zero NWP vs use_nwp=False", out_zeros, out_disabled)
    d_zero_none = diff("zero NWP vs absent NWP", out_zeros, out_none)

    print()
    if d_random > 0.0 and d_zero > 0.0 and d_none > 0.0:
        print("VERDICT A: the model DOES read the NWP.")
        print("          The --no-nwp ablation arm is a real contrast now.")
    else:
        print("VERDICT A: the model is STILL BLIND to the NWP (F30 unfixed).")
        failures += 1
    if d_abl == 0.0 and d_zero_none == 0.0:
        print("VERDICT B: zero / absent / disabled are bit-identical --")
        print("          a real ablation, not a constant 0.5 gate.")
    else:
        print("VERDICT B: the ablation arm is contaminated.")
        failures += 1

    # -- halo context is built from x_nwp by the adapter ------------------
    print()
    ctx, source = build_nwp_context(nwp_a)
    print(f"halo context        : shape {tuple(ctx.shape)}, source={source}")
    _ok_ctx = source == "in_patch" and tuple(ctx.shape) == (1, 20)
    if not _ok_ctx:
        print("VERDICT C: the harness-shaped NWP did not yield a 20-d context.")
        failures += 1
    else:
        print("VERDICT C: the 20-d halo context is built on the fly from x_nwp.")
    diag = model(radar, nwp_a).diagnostics
    print(f"diagnostics         : nwp_active={diag['nwp_active']}, "
          f"source={diag['nwp_source']}, "
          f"halo_connected={diag['halo_connected']}")
    if not (diag["nwp_active"] and diag["halo_connected"]):
        failures += 1

    # -- static cross-check ------------------------------------------------
    src = (Path(__file__).resolve().parents[1] / "qlfn2" / "qlfn2_model.py"
           ).read_text(encoding="utf-8")
    body = src.split("    def forward(", 1)[-1]
    n_body = body.count("x_nwp")
    print()
    print(f"static check        : 'x_nwp' occurs {n_body} time(s) in the "
          f"model forward body (signature + docstring only would be <= 2)")
    if n_body <= 2:
        print("          the NWP is not referenced in the forward body.")
        failures += 1

    print()
    print(f"RESULT: {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s))")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

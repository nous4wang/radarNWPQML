"""Self-checks for the qlfn2 library.

Every check states the fact it is checking and where that fact came from, so
a failure points at a specific measurement rather than at a vague shape
mismatch.  Run them with::

    python -m qlfn2.checks

The checks are pure CPU tensor work: they need neither the archive nor a
GPU, so they belong in the local test loop.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from typing import Callable

import torch

from .qlfn2_model import (
    HALO_CONTEXT_INDICES, NWP_VARS, nwp_context_from_patch,
    CAPPI_INDICES, HALO_EXPECTED, HALO_CONTEXT_VARS, HaloGeometry,
    Motion, PHASE_STATES, QLFN2Model, RADAR_VARS,
    PhaseRoutedZROperator, StructuredRainHead,
    advect_sequence, archive_to_net, bilinear_upsample, build_masks,
    center_crop, estimate_motion, net_to_archive, nwp_context_from_native,
    nwp_to_radar, reachable_distance_px, symmetric_patch_origins,
    vertical_quantities, warp,
)
from .qlfn2_losses import (
    ThresholdAwareLoss, lead_weights, threshold_logits,
)


class CheckFailure(AssertionError):
    """Raised when a self-check does not hold."""


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise CheckFailure(message)


# ---------------------------------------------------------------------------
def check_halo_geometry() -> str:
    """The halo geometry matches the measured numbers.

    Fact: the centre crop throws away 100 px on each side, which is 33.33
    native cells or 1.00 degree, and 63.9 % of the area.
    """
    g = HaloGeometry()
    _require(g.upsampled == 501, f"upsampled size {g.upsampled} != 501")
    _require(g.crop == 100, f"crop {g.crop} != 100")
    _require(abs(g.halo_degrees_per_side - 1.0) < 1e-6,
             f"halo width {g.halo_degrees_per_side} != 1.00 degree")
    _require(abs(g.halo_area_fraction - HALO_EXPECTED["halo_area_fraction"]) < 1e-3,
             f"halo area {g.halo_area_fraction} != {HALO_EXPECTED['halo_area_fraction']}")
    return (f"crop={g.crop}px side={g.halo_degrees_per_side}deg "
            f"area={g.halo_area_fraction:.4f}")


def _add_bench_to_path() -> bool:
    """Put the benchmark on the import path, so the comparison can run.

    The benchmark lives in its own tree: ``/home/wf/radarNWP`` on the
    remote, and a sibling checkout locally.
    """
    candidates = [
        os.environ.get("RADAR_NWP_BENCH"),
        "/home/wf/radarNWP",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "1radarNWPprediction", "radar-nwp-bench"),
        "R:/1radarNWPprediction/radar-nwp-bench",
    ]
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "radar_nwp_bench")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return True
    return False


def check_registration_matches_upstream() -> str:
    """Our registration equals the benchmark's own numpy registration.

    Fact: the dataset pipeline registers NWP by
    ``coordinates.bilinear_upsample_center_crop``; anything built on a
    different convention would not be comparable.
    """
    field = torch.randn(2, 167, 167)
    ours = nwp_to_radar(field)
    _require(ours.shape == (2, 301, 301), f"shape {tuple(ours.shape)} != (2, 301, 301)")
    if not _add_bench_to_path():
        return "skipped (the benchmark is not importable here)"
    try:
        from radar_nwp_bench.data.coordinates import bilinear_upsample_center_crop
    except ImportError:
        return "skipped (the benchmark is not importable here)"
    ref = torch.stack([
        torch.from_numpy(
            np.asarray(bilinear_upsample_center_crop(field[i].numpy()), dtype=np.float32)
        )
        for i in range(2)
    ])
    diff = float((ours - ref).abs().max())
    _require(diff < 2e-3, f"registration differs from upstream by {diff:.2e}")
    return f"max|diff| = {diff:.2e}"


def check_patch_origins() -> str:
    """The sliding-window origins span the field evenly."""
    origins = symmetric_patch_origins(301, 128, 64)
    _require(origins[0] == 0, "the first origin must be 0")
    _require(origins[-1] == 301 - 128, f"the last origin must be {301 - 128}")
    steps = {origins[i + 1] - origins[i] for i in range(len(origins) - 1)}
    _require(len(steps) <= 2, f"the origin spacing is not even: {sorted(steps)}")
    return f"{len(origins)} origins, spacing {sorted(steps)}"


def check_cappi_vertical_quantities() -> str:
    """A synthetic column gives the expected vertical quantities.

    The synthetic column carries signal at 2, 3 and 4 km with a maximum at
    3 km, so the top level is 4 km, the column maximum sits at 3 km and the
    bright band (max minus lowest) is the difference between 3 km and 2 km.
    """
    radar = torch.zeros(1, 10, 8, 8)
    radar[:, CAPPI_INDICES[0]] = 30.0     # 2 km, 30 dBZ
    radar[:, CAPPI_INDICES[1]] = 45.0     # 3 km, 45 dBZ
    radar[:, CAPPI_INDICES[2]] = 35.0     # 4 km, 35 dBZ
    vq = vertical_quantities(radar)
    top = float(vq["cap_top_km"][0, 0, 0, 0])
    arg = float(vq["cap_argmax_km"][0, 0, 0, 0])
    nlev = float(vq["cap_nz_levels"][0, 0, 0, 0])
    bright = float(vq["cap_bright_dbz"][0, 0, 0, 0])
    _require(abs(top - 4.0) < 1e-5, f"cap_top_km = {top}, expected 4.0")
    _require(abs(arg - 3.0) < 1e-5, f"cap_argmax_km = {arg}, expected 3.0")
    _require(abs(nlev - 3.0) < 1e-5, f"cap_nz_levels = {nlev}, expected 3.0")
    _require(abs(bright - 1.5) < 1e-4, f"cap_bright_dbz = {bright}, expected 1.5")
    return f"top={top} argmax={arg} nz={nlev} bright={bright}"


def check_operator_is_a_power_law_at_init() -> str:
    """With uniform phase weights the operator is a single power law."""
    op = PhaseRoutedZROperator()
    op.eval()
    radar = torch.zeros(1, 10, 8, 8)
    radar[:, 6] = 400.0                     # 40 dBZ composite reflectivity
    with torch.no_grad():
        out = op(radar)
    weights = out.weights[0, :, 0, 0]
    _require(float(weights.sum()) == 1.0 or abs(float(weights.sum()) - 1.0) < 1e-5,
             f"the phase weights do not sum to one: {float(weights.sum())}")
    _require(float(out.rain[0, 0, 0]) >= 0.0, "the operator produced a negative rain rate")
    return f"weights={weights.tolist()} rain@40dBZ={float(out.rain[0,0,0]):.4f}"


def check_operator_matches_the_measured_relation() -> str:
    """The operator starts on the relation the archive actually shows.

    Measured on the remote, over 366676 wet pixels of the first ten
    cases: the log-log fit between composite reflectivity and the 6-minute
    rainfall is

        log y = -3.3517 + 0.6600 * log(CR_raw)

    which is ``y = 0.1600 * z_dBZ ** 0.660`` in millimetres per 6-minute
    accumulation.  The operator is initialised on exactly that relation,
    so it starts at the physical scale of the answer rather than an order
    of magnitude away from it.

    This is the check that catches the amplitude bug: an operator whose
    intercept is off by a factor of ten cannot reach the 2 mm threshold at
    all, and every score it produces is a statement about the threshold
    rather than about the model.
    """
    op = PhaseRoutedZROperator()
    op.eval()
    worst = 0.0
    for z in (10.0, 20.0, 30.0, 45.0, 60.0):
        radar = torch.zeros(1, 10, 1, 1)
        radar[:, 6] = z * 10.0               # raw units: 0.1 dBZ
        with torch.no_grad():
            got = float(op(radar).rain.flatten()[0])
        measured = 0.1600 * z ** 0.660
        worst = max(worst, abs(got - measured) / measured)
    _require(worst < 0.02,
             f"the operator is off the measured relation by {worst:.4f}")
    return f"max relative deviation from the measured relation = {worst:.2e}"


def check_head_can_reach_the_scored_threshold() -> str:
    """The head must be able to emit 2 mm per 6 minutes.

    CSI@2.0 is the benchmark's primary score, so a head that cannot reach
    2 mm whatever it learns cannot score on it.  The operator alone tops
    out near 1.97 mm at 45 dBZ, so the learned correction has to be able to
    add to it; this check drives the correction to its bound and asks
    whether 2 mm is within reach.
    """
    head = StructuredRainHead(48)
    head.eval()
    radar = torch.zeros(1, 10, 8, 8)
    radar[:, 6] = 450.0                     # 45 dBZ
    features = torch.zeros(1, 48, 8, 8)
    with torch.no_grad():
        head.residual[-1].bias.fill_(10.0)
        top = float(head(radar, features).rain.flatten()[0])
    _require(top >= 2.0,
             f"the head tops out at {top:.3f} mm/6min, below the 2 mm threshold")
    return f"head ceiling at 45 dBZ = {top:.3f} mm/6min (threshold 2.0)"


def check_operator_is_zero_without_echo() -> str:
    """No reflectivity means no rain, whatever the phase weights say."""
    op = PhaseRoutedZROperator()
    op.eval()
    radar = torch.zeros(1, 10, 8, 8)
    with torch.no_grad():
        out = op(radar)
    _require(float(out.rain.abs().max()) == 0.0,
             f"the operator produced rain without echo: {float(out.rain.abs().max())}")
    return "no echo -> no rain"


def check_motion_recovery() -> str:
    """A translating blob is recovered with the right sign and size."""
    torch.manual_seed(0)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    vx, vy = 2.0, -1.0
    frames = []
    for k in range(8):
        cx, cy = 30 + vx * k, 30 + vy * k
        blob = 400.0 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 20.0)
        frame = torch.zeros(10, h, w)
        frame[6] = blob
        frames.append(frame)
    radar = torch.stack(frames, dim=0)[None]
    motion = estimate_motion(radar)
    _require(motion.usable, "the motion was not estimated")
    _require(abs(motion.vx - vx) < 0.35, f"vx = {motion.vx:.3f}, expected {vx}")
    _require(abs(motion.vy - vy) < 0.35, f"vy = {motion.vy:.3f}, expected {vy}")
    return f"vx={motion.vx:.3f} vy={motion.vy:.3f}"


def check_warp_alignment() -> str:
    """Warping a translating field by its own motion removes the shift."""
    torch.manual_seed(1)
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    vx, vy = 2.0, -1.0
    cx, cy = 30.0, 30.0
    field = (400.0 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 20.0))[None, None]
    moved = warp(field, vx, vy, 1.0)
    expect = 400.0 * torch.exp(-((xx - (cx + vx)) ** 2 + (yy - (cy + vy)) ** 2) / 20.0)
    diff = float((moved[0, 0] - expect).abs().max())
    _require(diff < 1.5, f"the warped field is off by {diff:.3f} dBZ")
    return f"max|diff| = {diff:.3f} dBZ"


def check_halo_masks_point_upstream() -> str:
    """The upstream annulus is on the side the weather comes from."""
    motion = Motion(vx=2.0, vy=0.0)          # the system moves east
    masks = build_masks(motion, max_distance_px=40.0)
    n = masks.centre.shape[0]
    c0 = (n - 301) // 2
    # the upstream side is west, i.e. the smaller column indices
    upstream_cols = masks.upstream[:, :c0].sum()          # the west columns
    downstream_cols = masks.downstream[:, c0 + 301:].sum()  # the east columns
    _require(int(upstream_cols) > 0, "the upstream annulus is empty on the upstream side")
    _require(int(masks.upstream[:, c0 + 301:].sum()) == 0,
             "the upstream annulus leaked into the downstream side")
    _require(int(downstream_cols) > 0,
             "the downstream annulus is empty on the downstream side")
    total_ring = int((~masks.centre).sum())
    _require(int(masks.upstream.sum()) < total_ring,
             "the narrow annulus is not narrower than the whole ring")
    return (f"upstream px={int(masks.upstream.sum())} of ring {total_ring}, "
            f"reach={reachable_distance_px(motion):.1f}px")


def check_model_shapes() -> str:
    """The model returns the benchmark's seam and no NaN."""
    torch.manual_seed(2)
    b, t, h, w = 2, 11, 64, 64
    x = torch.zeros(b, t, 10, h, w)
    x[:, :, 6] = 300.0
    x[:, :, 9] = 200.0
    model = QLFN2Model(in_len=t, out_len=10, use_history_head=True, width=8)
    with torch.no_grad():
        out = model(x)
    _require(out.rain.shape == (b, 10, 1, h, w),
             f"rain shape {tuple(out.rain.shape)} != {(b, 10, 1, h, w)}")
    _require(out.rain_history is not None and out.rain_history.shape == (b, t, 1, h, w),
             "the history head did not produce the zero-advection supervision")
    _require(bool(torch.isfinite(out.rain).all()), "the model produced NaN")
    _require(bool((out.rain >= 0).all()), "the model produced a negative rain rate")
    return f"rain {tuple(out.rain.shape)}, history {tuple(out.rain_history.shape)}"


def check_copy_forward_branch() -> str:
    """The copy-forward arm is runnable, and it really copies.

    Fact: ``motion_mode="none"`` is the ablation that switches the advection
    off, so its output has to be the t0 rain field repeated across the ten
    leads.  The arm exists to answer whether the advection carries
    anything, so it has to be *runnable*, and it has to be a copy: a branch
    that silently returned the wrong shape, or that advected anyway, would
    make the arm meaningless while looking perfectly healthy.

    This check is here because that branch had never been executed before
    the experiment reached it.  The seed-10 arm-2 job died on its first
    forward pass with a shape error, and that error had been sitting in the
    code the whole time -- every earlier run had used the advection branch.
    """
    torch.manual_seed(11)
    b, t, h, w = 2, 11, 64, 64
    x = torch.zeros(b, t, 10, h, w)
    x[:, :, 6] = 300.0
    x[:, :, 9] = 200.0
    model = QLFN2Model(in_len=t, out_len=10, use_history_head=True, width=8,
                       motion_mode="none")
    with torch.no_grad():
        out = model(x)
    _require(out.rain.shape == (b, 10, 1, h, w),
             f"rain shape {tuple(out.rain.shape)} != {(b, 10, 1, h, w)}")
    _require(bool(torch.isfinite(out.rain).all()),
             "the copy-forward arm produced NaN")
    _require(float(out.rain.max()) > 0.0,
             "the copy-forward arm produced no rain at all")
    lead0 = out.rain[:, 0]
    _require(all(bool(torch.equal(out.rain[:, k], lead0)) for k in range(1, 10)),
             "the copy-forward arm is not a copy: the leads differ")
    _require(float(out.motion.vx.abs().max()) == 0.0,
             "the copy-forward arm moved the field")
    return f"copy-forward {tuple(out.rain.shape)}, 10 identical leads"


def check_halo_branch_runs() -> str:
    """The halo branch executes, and the context reaches the prediction.

    Fact: the halo projection used to project onto ``width`` channels while
    the offset was added to the ``n_features``-channel feature stack, so
    the first execution of this branch raised a ``RuntimeError``.  The
    branch had never run before, because the benchmark's harness never
    passes a halo context -- the upstream tree contains no occurrence of
    ``halo_context`` at all.

    Two things are checked, and the difference between them matters.  The
    halo offset enters the *trunk*, so it must change the trunk's output
    immediately.  It reaches the *rain field* only through the residual
    branch, whose last convolution is zero-initialised on purpose -- the
    model starts as a pure physical operator and learns the correction
    later.  So the end-to-end effect is checked after that convolution is
    given a non-zero weight.
    """
    torch.manual_seed(13)
    b, t, h, w = 2, 11, 32, 32
    x = torch.zeros(b, t, 10, h, w)
    x[:, :, 6] = 300.0
    x[:, :, 9] = 200.0
    model = QLFN2Model(in_len=t, out_len=10, use_halo=True,
                       use_history_head=True, width=8)
    # 合并后 halo_proj 零初始化（训练起点与合并前逐位一致）；这里给出
    # 非零权重，模拟训练后的状态，否则两个上下文都投影成同一个零偏移。
    with torch.no_grad():
        model.halo_proj.weight.normal_(0.0, 0.05)
        model.halo_proj.bias.normal_(0.0, 0.05)
    ctx_a = torch.zeros(b, 20)
    ctx_b = torch.ones(b, 20) * 2.0

    trunk_out: list[torch.Tensor] = []

    def _capture(_module, _inputs, output):
        trunk_out.append(output.detach().clone())

    handle = model.trunk.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            out_a = model(x, halo_context=ctx_a)
            trunk_a = trunk_out[0]
            trunk_out.clear()
            out_b = model(x, halo_context=ctx_b)
            trunk_b = trunk_out[0]
    finally:
        handle.remove()

    _require(bool(torch.isfinite(out_a.rain).all()),
             "the halo branch produced NaN")
    _require(out_a.rain.shape == (b, 10, 1, h, w),
             f"halo rain shape {tuple(out_a.rain.shape)} != {(b, 10, 1, h, w)}")
    trunk_gap = float((trunk_a - trunk_b).abs().max())
    _require(trunk_gap > 0.0,
             "the halo context does not reach the trunk, so the projection "
             "is not wired to anything")

    # give the residual branch a non-zero last convolution, as training would
    with torch.no_grad():
        model.head.residual[-1].weight.normal_(0.0, 0.05)
        model.head.residual[-1].bias.normal_(0.0, 0.05)
        rain_a = model(x, halo_context=ctx_a).rain
        rain_b = model(x, halo_context=ctx_b).rain
    rain_gap = float((rain_a - rain_b).abs().max())
    _require(rain_gap > 0.0,
             "the halo context does not reach the rain field even with a "
             "non-zero residual")
    return (f"halo branch runs; trunk gap {trunk_gap:.3e}, "
            f"rain gap {rain_gap:.3e} mm")


def check_nwp_is_consumed() -> str:
    """合并后模型必须**真的**消费 NWP —— 判据相对 F30 整体反转。

    F30 实测：合并前 qlfn2 对四种 NWP（随机 / x100 / 全零 / 缺席）输出
    bit-identical，臂 0 与臂 1 的逐轮曲线 r = 0.986 / 0.998、mean|delta|
    = 0.0019 / 0.0018 ⇒ ``--no-nwp`` 不是消融，是重复。本自检确保接线后
    不会再悄悄退回那个状态。

    两条 NWP 通路都做零初始化（训练起点与合并前逐位一致），所以这里
    显式给非零权重来模拟训练后的状态；这是"按分支覆盖"而不是"按函数
    覆盖" —— 只测起点会看到恒等，误判成仍然全盲。
    """
    torch.manual_seed(17)
    b, t, h, w = 2, 11, 32, 32
    x = torch.zeros(b, t, 10, h, w)
    x[:, :, 6] = 300.0
    x[:, :, 9] = 200.0
    model = QLFN2Model(in_len=t, out_len=10, use_history_head=True, width=8)
    model.eval()
    with torch.no_grad():
        model.halo_proj.weight.normal_(0.0, 0.05)
        model.halo_proj.bias.normal_(0.0, 0.05)
        model.nwp_gate[-1].weight.normal_(0.0, 0.05)
        model.nwp_gate[-1].bias.normal_(0.0, 0.05)
    nwp_a = torch.rand(b, 2, 27, h, w)
    nwp_b = torch.rand(b, 2, 27, h, w) * 100.0
    with torch.no_grad():
        out_a = model(x, nwp_a).rain
        out_b = model(x, nwp_b).rain
        out_z = model(x, torch.zeros_like(nwp_a)).rain
        out_none = model(x, None).rain
    for name, other in (("scaled", out_b), ("zero", out_z), ("absent", out_none)):
        gap = float((out_a - other).abs().max())
        _require(gap > 0.0,
                 f"the model is still blind to the NWP ({name} changes the "
                 f"output by {gap:.3e}); the F30 wiring has regressed")
    _require(float((out_z - out_none).abs().max()) == 0.0,
             "an all-zero NWP and an absent NWP must behave identically")

    # --no-nwp 是一次真消融：全零 NWP 与显式关掉 NWP 逐位相同。
    # 若只判 `x_nwp is not None`，sigmoid 门控会输出 0.5，消融臂变成
    # "整体折半"，而不是"没有 NWP"。
    saved = model.use_nwp
    model.use_nwp = False
    with torch.no_grad():
        out_disabled = model(x, nwp_a).rain
    model.use_nwp = saved
    gap_abl = float((out_z - out_disabled).abs().max())
    _require(gap_abl == 0.0,
             f"the zero-NWP ablation differs from use_nwp=False by {gap_abl:.3e}")
    return (f"NWP changes the rain by {float((out_a - out_b).abs().max()):.3e} mm; "
            "zero/None/disabled are bit-identical")


def check_loss_threshold_alignment() -> str:
    """事件项的 logit 必须以每个阈值各自居中（用户审查第 3 条）。

    修复前 ``score = log(p)`` 把决策边界钉死在 1.0 mm：给 p = 1.5 mm、
    y 全为 0，阈值取 0.5 与取 2.0 会得到**完全相同**的事件损失。
    修复后 ``score = log(p) - log(thr)``，前者为正（错判正例、损失大）、
    后者为负（正确判负、损失小），两者必须显著不同。
    """
    p = torch.full((1, 2, 2, 2), 1.5)
    y = torch.zeros_like(p)
    for thr in (0.5, 1.0, 2.0, 4.0):
        at_thr = float(threshold_logits(torch.tensor([float(thr)]), thr)[0])
        _require(abs(at_thr) < 1e-6,
                 f"logit at p == {thr} is {at_thr:.3e}, not zero")
    _require(float(threshold_logits(p, 0.5)[0, 0, 0, 0]) > 0.0,
             "p=1.5 should be a positive at the 0.5 mm threshold")
    _require(float(threshold_logits(p, 2.0)[0, 0, 0, 0]) < 0.0,
             "p=1.5 should be a negative at the 2.0 mm threshold")

    kw = dict(weight_amount=0.0, weight_event=1.0, event_positive_weight=1.0)
    e05 = float(ThresholdAwareLoss(thresholds=(0.5,), **kw)(p, y).event)
    e20 = float(ThresholdAwareLoss(thresholds=(2.0,), **kw)(p, y).event)
    _require(e05 - e20 > 1e-3,
             f"the 0.5 and 2.0 mm event losses are too close "
             f"({e05:.6f} vs {e20:.6f}); the thresholds are still collapsed")

    # lead_decay 必须逐时效生效，而不只是乘一个标量 decay.mean()
    y2 = torch.zeros(1, 2, 2, 2)
    y2[:, 0] = 3.0
    d1 = float(ThresholdAwareLoss(thresholds=(2.0,), lead_decay=1.0, **{
        k: v for k, v in kw.items()})(p, y2).event)
    d2 = float(ThresholdAwareLoss(thresholds=(2.0,), lead_decay=0.2, **{
        k: v for k, v in kw.items()})(p, y2).event)
    _require(abs(d1 - d2) > 1e-6,
             "lead_decay has no per-lead effect on the event term")
    return f"logit(thr)=0; event@0.5={e05:.4f} > event@2.0={e20:.4f}"


def check_adapter_wires_halo_context() -> str:
    """adapter 从 x_nwp 现场生成 20 维上下文 —— F30 断联修复的回归判据。

    上游 harness 只调 ``model(x_radar, x_nwp)``，从不传 ``halo_context``。
    合并前 adapter 也不生成它 ⇒ halo 投影永远收不到输入。本自检要求
    adapter 在普通两参调用下就把上下文接上，并写进 diagnostics 供日志核对。
    """
    torch.manual_seed(29)
    b, t, h, w = 2, 11, 32, 32
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    compressed = archive_to_net(raw)
    nwp = torch.rand(b, 2, 27, h, w)
    from qlfn2.qlfn2_model import QLFN2Adapter, build_nwp_context

    ctx, source = build_nwp_context(nwp)
    _require(source == "in_patch",
             f"the harness-shaped NWP took the {source!r} path, not 'in_patch'")
    _require(tuple(ctx.shape) == (b, 20),
             f"context shape {tuple(ctx.shape)} != {(b, 20)}")

    model = QLFN2Adapter(in_radar_c=10, in_nwp_c=27, in_len=t, out_len=10,
                         width=8, use_history_head=True)
    with torch.no_grad():
        pred = model(compressed, nwp)
    diag = model.last_diagnostics
    _require(pred.shape == (b, 10, 1, h, w),
             f"adapter returned {tuple(pred.shape)}")
    _require(diag.get("nwp_active") is True,
             "the adapter did not activate the NWP path")
    _require(diag.get("nwp_source") == "in_patch",
             f"nwp_source is {diag.get('nwp_source')!r}")
    _require(diag.get("halo_connected") is True,
             "the halo projection still receives no context")
    _require(model.last_qpe_pred is not None,
             "the R1 history supervision was lost")
    return (f"two-arg harness call -> source={diag['nwp_source']}, "
            f"context {tuple(ctx.shape)}, halo_connected=True")


def check_native_halo_ring_contract() -> str:
    """原生 167x167 外环路径产出的上下文与 halo_proj 的 20 维契约一致。

    这条分支在当前上游 harness 下不可达（dataset 已把 NWP 裁到 patch），
    但必须可执行、可测量：将来数据层开放外环后，同一套 halo_proj 权重
    可以直接接收真正的上游半平面统计，而不需要换模型。
    """
    torch.manual_seed(31)
    native = torch.randn(1, 27, 167, 167)
    motion = Motion(vx=2.0, vy=1.0, usable=True)
    ctx = nwp_context_from_native(native, motion)
    _require(tuple(ctx.shape) == (1, 20),
             f"native ring context {tuple(ctx.shape)} != (1, 20)")
    _require(bool(torch.isfinite(ctx).all()), "native ring context has NaN")
    _require(len(HALO_CONTEXT_VARS) * 4 == 20,
             "5 variables x 4 statistics must equal the halo_proj input width")
    return f"native ring -> {tuple(ctx.shape)} context, 5 vars x 4 stats"


def check_unit_bridge() -> str:
    """The network scale and the archive's units convert back and forth.

    Fact: the harness hands the network ``log1p(clip(v, 0)) / 6`` for the
    dBZ variables and ``/ 4`` for ET and VIL, and that transform is exactly
    invertible.  If it were not, the qlfn2 physics would be reading the
    wrong units and every score would be meaningless.
    """
    torch.manual_seed(3)
    raw = torch.rand(2, 3, 10, 16, 16) * 600.0
    back = net_to_archive(archive_to_net(raw))
    diff = float((back - raw).abs().max())
    _require(diff < 1e-2 * max(1.0, float(raw.abs().max())),
             f"the unit bridge does not round-trip: max|diff| = {diff:.3e}")
    if not _add_bench_to_path():
        return f"max|diff| = {diff:.2e} (the benchmark is not importable here)"
    try:
        from radar_nwp_bench.data.transforms import radar_log_transform
    except ImportError:
        return f"max|diff| = {diff:.2e} (the benchmark is not importable here)"
    physical = raw / 10.0
    # the upstream transform works per variable, so it is compared one
    # variable at a time
    worst = 0.0
    for c, name in enumerate(RADAR_VARS):
        ours = archive_to_net(raw[:, :, c:c + 1], (name,)).squeeze(2)
        theirs = torch.from_numpy(np.asarray(
            radar_log_transform(physical[:, :, c].numpy(), name), dtype=np.float32))
        worst = max(worst, float((ours - theirs).abs().max()))
    _require(worst < 1e-5,
             f"the unit bridge differs from the benchmark's transform by {worst:.2e}")
    return f"max|diff| vs upstream = {worst:.2e}"


def check_motion_is_tile_consistent() -> str:
    """The correlation estimator reports the same velocity on every tile.

    Fact: the benchmark slides a 128x128 patch over the 301x301 grid and
    stitches the tiles back together, so a statistic that depends on where
    the storm sits inside the tile puts a discontinuity at the seams.  The
    normalised cross-correlation is invariant to the crop; the echo
    centroid is not.  This check records both numbers.
    """
    torch.manual_seed(4)
    s = 256
    yy, xx = torch.meshgrid(torch.arange(s, dtype=torch.float32),
                            torch.arange(s, dtype=torch.float32), indexing="ij")
    frames = []
    for k in range(11):
        frame = torch.zeros(10, s, s)
        frame[6] = 300.0 * torch.exp(
            -((xx - (120.0 + 2.0 * k)) ** 2 + (yy - 128.0) ** 2) / 300.0)
        frames.append(frame)
    full = torch.stack(frames, dim=0)[None]
    tiles = {
        "origin": full[:, :, :, :128, :128],
        "shifted": full[:, :, :, 64:192, 64:192],
    }
    corr = [estimate_motion(t, mode="corr") for t in tiles.values()]
    centroid = [estimate_motion(t, mode="centroid") for t in tiles.values()]
    for m in corr:
        _require(m.usable, "the correlation estimator found no motion")
        _require(abs(m.vx - 2.0) < 0.51, f"vx = {m.vx:.3f}, expected 2.0")
    spread = max(abs(m.vx - 2.0) for m in corr)
    centroid_spread = max(abs(m.vx - 2.0) for m in centroid)
    _require(spread < 0.51,
             f"the correlation estimator disagrees across tiles by {spread:.3f}")
    _require(centroid_spread > spread,
             "the centroid estimator was expected to disagree across tiles")
    return (f"corr spread = {spread:.2f} px, centroid spread = "
            f"{centroid_spread:.2f} px")


def check_adapter_seam() -> str:
    """The adapter returns the benchmark's contract on a real batch shape.

    Fact: the harness calls ``model(x_radar, x_nwp)`` and expects
    ``(B, T_out, 1, H, W)`` in millimetres per 6-minute accumulation.  The
    check runs the adapter on the compressed scale the dataset produces.
    """
    torch.manual_seed(5)
    b, t, h, w = 2, 11, 64, 64
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    compressed = archive_to_net(raw)
    if not _add_bench_to_path():
        return "skipped (the benchmark is not importable here)"
    try:
        from qlfn2.qlfn2_model import QLFN2Adapter
    except RuntimeError as exc:
        return f"skipped ({exc})"
    model = QLFN2Adapter(in_radar_c=10, in_nwp_c=27, in_len=t, out_len=10,
                         width=8, use_history_head=True)
    with torch.no_grad():
        pred = model(compressed, None)
    _require(pred.shape == (b, 10, 1, h, w),
             f"the adapter returned {tuple(pred.shape)}, expected {(b, 10, 1, h, w)}")
    _require(bool(torch.isfinite(pred).all()), "the adapter produced NaN")
    _require(bool((pred >= 0).all()), "the adapter produced a negative rain rate")
    _require(model.last_qpe_pred is not None
             and model.last_qpe_pred.shape == (b, t, 1, h, w),
             "the adapter did not publish the history-rainfall supervision")
    _require(model.last_diagnostics.get("nwp_active") is False,
             "x_nwp=None must deactivate the NWP path completely")
    return f"pred {tuple(pred.shape)}, history {tuple(model.last_qpe_pred.shape)}"


def check_runs_at_both_evaluation_geometries() -> str:
    """The model runs at the patch sizes both evaluation geometries use.

    Fact: the upstream authority scripts score with ``--patch-size 64
    --slide-stride 64``, and the qlfn2 sweep scored with ``--patch-size 128
    --slide-stride 64``.  The two disagree by 13.8 % on the benchmark's own
    ``physical_qpe`` epoch_00 (0.1226 under 64/64, 0.1395 under 128/128 --
    same weights, same protocol), so the geometry is a measurement
    decision, not a detail.

    A model that only runs at one patch size would make one of those two
    numbers unreachable, and it would be reachable only by accident -- the
    failure would show up as a shape error deep inside the training loop,
    after the GPU time has been spent.  This check runs the model at both
    sizes and requires that the interior of the 128-pixel prediction agrees
    with the 64-pixel one, which is what says the model is size-agnostic
    rather than accidentally shape-bound.
    """
    torch.manual_seed(23)
    b, t = 1, 11
    big = torch.zeros(b, t, 10, 128, 128)
    big[:, :, 6] = 300.0
    big[:, :, 9] = 200.0
    small = big[:, :, :, :64, :64].contiguous()

    model = QLFN2Model(in_len=t, out_len=10, use_history_head=True, width=8)
    model.eval()
    with torch.no_grad():
        out_big = model(big)
        out_small = model(small)

    _require(out_big.rain.shape == (b, 10, 1, 128, 128),
             f"rain at 128 is {tuple(out_big.rain.shape)}")
    _require(out_small.rain.shape == (b, 10, 1, 64, 64),
             f"rain at 64 is {tuple(out_small.rain.shape)}")
    _require(bool(torch.isfinite(out_big.rain).all()),
             "the model produced NaN at patch 128")
    _require(bool(torch.isfinite(out_small.rain).all()),
             "the model produced NaN at patch 64")

    # The interior of the 64-pixel prediction has to be close to the
    # corresponding interior of the 128-pixel one.  The edges are excluded
    # on purpose: a convolutional model sees a different neighbourhood at
    # the boundary, so the border pixels are allowed to disagree.
    interior_big = out_big.rain[..., 16:48, 16:48]
    interior_small = out_small.rain[..., 16:48, 16:48]
    scale = max(float(interior_big.abs().max()), 1e-6)
    rel = float((interior_big - interior_small).abs().max()) / scale
    _require(rel < 0.5,
             f"the two patch sizes disagree by {rel:.3f} (relative) in the "
             f"interior, so the model is not size-agnostic")
    return f"128 and 64 both run; interior relative gap {rel:.3f}"


def check_loss_decay_broadcast_and_direction() -> str:
    """5D 输入不得裂变通道轴；逐时效权重必须从 1.0 单调衰减。

    实测（修复前，t=10, lead_decay=0.2）：4D ``(2,10,8,8)`` 得 0.114381，
    5D ``(2,10,1,8,8)`` 静默得 0.114000 —— ``decay.view(1,t,1,1)`` 的
    ``t`` 落在通道维，把 1 个通道广播成 10 个；通道数为 3 时才抛
    ``RuntimeError``。也就是说上游契约 ``(B,T,1,H,W)`` 恰好是**唯一不报错
    但结果错误**的形状。

    第二个判据：修复前权重为 ``[5e-7,...,0.2,1.0]``（越近越小），
    修复后必须是 ``[1.0,...,0.2]``（越远越小）。
    """
    torch.manual_seed(31)
    b, t, h, w = 2, 10, 6, 6
    p4 = torch.rand(b, t, h, w) + 0.5
    y4 = torch.zeros_like(p4)
    y4[:, 0] = 3.0
    kw = dict(thresholds=(2.0,), weight_amount=0.5, weight_event=1.0,
              event_positive_weight=1.0, lead_decay=0.2)
    loss = ThresholdAwareLoss(**kw)
    got4 = loss(p4, y4)
    got5 = loss(p4.unsqueeze(2), y4.unsqueeze(2))   # (B,T,1,H,W)
    gap = float((got4.total - got5.total).abs().max())
    _require(gap == 0.0,
             f"4D and 5D losses differ by {gap:.3e}; the lead axis is "
             f"broadcast onto the channel axis")

    # 修复前 3 通道会 RuntimeError；修复后通道维不该被当成时效轴。
    p5c = p4.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
    y5c = y4.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
    got5c = loss(p5c, y5c)
    _require(bool(torch.isfinite(got5c.total)),
             "a 3-channel 5D prediction must not crash or produce NaN")

    weights = lead_weights(t, 0.2)
    _require(abs(weights[0] - 1.0) < 1e-12,
             f"the first lead must have weight 1.0, got {weights[0]:.6g}")
    _require(abs(weights[-1] - 0.2) < 1e-12,
             f"the last lead must have weight 0.2, got {weights[-1]:.6g}")
    _require(all(weights[i] >= weights[i + 1] for i in range(t - 1)),
             f"the lead weights are not monotonically decreasing: {weights}")

    # 默认配方（decay=1.0）必须逐时效都是 1，即完全不加权。
    unit = lead_weights(t, 1.0)
    _require(all(abs(v - 1.0) < 1e-12 for v in unit),
             "lead_decay=1.0 must be a no-op on every lead")
    return (f"4D==5D (gap {gap:.1e}); weights {weights[0]:.2f} -> "
            f"{weights[-1]:.2f} decreasing; 3-channel 5D finite")


def check_history_last_frame_is_gated() -> str:
    """rain_history 的最后一帧 = 已门控的 rain_t0，且梯度能直达 nwp_gate。

    上游 R1 用 ``last_qpe_pred``（= rain_history）直接监督 ``y_hist``，
    最后一帧对应 t0 地面降水。修复前该帧由 ``head`` 重算，不含
    ``rain_t0 * (1 + gate)`` ⇒ NWP 门控只能从未来 10 帧经平流间接拿梯度。
    """
    torch.manual_seed(37)
    b, t, h, w = 2, 11, 24, 24
    x = torch.zeros(b, t, 10, h, w)
    x[:, :, 6] = 300.0
    x[:, :, 9] = 200.0
    model = QLFN2Model(in_len=t, out_len=10, width=8,
                       use_history_head=True).eval()
    with torch.no_grad():
        plain = model(x)
    gap = float((plain.rain_history[:, -1]
                 - plain.rain_t0).abs().max())
    _require(gap == 0.0,
             f"without NWP the last history frame differs from rain_t0 "
             f"by {gap:.3e}; the merge equivalence is broken")

    # 模拟训练后的门控（零初始化下梯度仍存在，但幅度太小不便判读）
    with torch.no_grad():
        model.nwp_gate[-1].weight.normal_(0.0, 0.1)
        model.nwp_gate[-1].bias.normal_(0.0, 0.1)
    nwp = torch.rand(b, 2, 27, h, w)
    out = model(x, nwp)
    out.rain_history[:, -1].sum().backward()
    grad = model.nwp_gate[-1].weight.grad
    _require(grad is not None and float(grad.abs().max()) > 0.0,
             "the t0 history supervision gives no gradient to the NWP gate; "
             "the last frame is still an un-gated head call")
    changed = float((out.rain_history[:, -1]
                     - out.rain_t0).abs().max().detach())
    _require(changed == 0.0,
             "the last history frame and rain_t0 diverged after gating")
    return (f"last frame == gated rain_t0 (gap {gap:.1e}); "
            f"gate grad max|g| = {float(grad.abs().max()):.3e}")


def check_adapter_halo_follows_nwp() -> str:
    """构造参数里没有 NWP 时，halo_proj / nwp_gate 都不该被实例化。

    同时确认上游 ``--no-nwp`` 的真实语义：``apply_ablation`` 给的是
    **同形状全零张量**，``in_nwp_c`` 仍为 27 ⇒ 默认 state_dict 不变，
    消融仍是"参数量相同的受控实验"；只是 forward 里跳过 NWP 分支。
    """
    from qlfn2.qlfn2_model import QLFN2Adapter

    full = QLFN2Adapter(in_radar_c=10, in_nwp_c=27, width=8)
    _require(full.core.halo_proj is not None
             and full.core.nwp_gate is not None,
             "the default adapter must keep both NWP paths")
    blank = QLFN2Adapter(in_radar_c=10, in_nwp_c=0, width=8,
                         use_halo=True, use_nwp=True)
    _require(blank.core.halo_proj is None and blank.core.nwp_gate is None,
             "use_halo did not follow use_nwp / in_nwp_c")
    _require(len(blank.state_dict()) < len(full.state_dict()),
             "the no-NWP adapter should not carry dead NWP weights")

    b, t, h, w = 2, 11, 16, 16
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    comp = archive_to_net(raw)
    with torch.no_grad():
        p_zero = full(comp, torch.zeros(b, 2, 27, h, w))
        p_none = full(comp, None)
    gap = float((p_zero - p_none).abs().max())
    _require(gap == 0.0,
             f"--no-nwp (all-zero tensor) differs from x_nwp=None by {gap:.3e}")
    _require(full.last_diagnostics.get("nwp_active") is False,
             "an all-zero NWP must deactivate the NWP path")
    return (f"halo/nwp_gate off when in_nwp_c=0; zero-NWP == None "
            f"(gap {gap:.1e}); default state_dict unchanged")


def check_compat_aliases() -> str:
    """三个旧模块名是**纯别名**，必须指向同一个实现对象。"""
    import importlib

    impl_m = importlib.import_module("qlfn2.qlfn2_model")
    impl_l = importlib.import_module("qlfn2.qlfn2_losses")
    a = importlib.import_module("qlfn2.adapter")
    m = importlib.import_module("qlfn2.model")
    l = importlib.import_module("qlfn2.losses")
    _require(a.QLFN2Adapter is impl_m.QLFN2Adapter,
             "qlfn2.adapter.QLFN2Adapter is a second implementation")
    _require(m.QLFN2Model is impl_m.QLFN2Model
             and m.ModelOutput is impl_m.ModelOutput,
             "qlfn2.model is not the merged implementation")
    _require(l.ThresholdAwareLoss is impl_l.ThresholdAwareLoss
             and l.lead_weights is impl_l.lead_weights,
             "qlfn2.losses is not the merged implementation")
    _require(set(l.__all__) == set(impl_l.__all__),
             "the losses shim's __all__ drifted from the implementation")
    return "adapter/model/losses aliases point at the two merged files"


# NOTE(F36, 2026-11-14): CHECKS / run_all / __main__ live at the END of this
# file.  The two F36 checks were originally appended *after* the __main__
# block, so the CHECKS tuple referenced names that did not exist yet and
# ``python -m qlfn2.checks`` died with NameError at import time.  Any new
# check must be defined ABOVE the CHECKS tuple, and ``__main__`` stays last.


def check_nwp_channel_semantics() -> str:
    """HALO_CONTEXT_VARS 必须精确对齐上游 NWP_VARS 的真实通道索引 (2,3,0,14,23)。

    F36 修复 D1 回归判据：合并版曾用 range(5)/enumerate 盲取 0..4 通道，
    把 DVG200(1) 与 HT0(4) 当成 PWAT(14) 与 TTdMean74(23)。
    本自检验证：(1) 索引表精确等于 (2, 3, 0, 14, 23)，且如上游可导入则与
    上游 data.index.NWP_VARS 完全一致；(2) 扰动通道 14/23 必须改变上下文，
    而扰动未被选中的通道 1 (DVG200) 和 4 (HT0) 上下文必须严格不变 (0.000e+00)。
    """
    _require(HALO_CONTEXT_INDICES == (2, 3, 0, 14, 23),
             f"HALO_CONTEXT_INDICES = {HALO_CONTEXT_INDICES}, expected (2, 3, 0, 14, 23)")
    # 不依赖上游 checkout 的硬校验：F36 初版曾用另一套自造的 27 变量名表，
    # 把 TTdMean74 放到 22、实际取到 TTdMax74，索引仍"看起来"修好了。
    _require(len(NWP_VARS) == 27, f"NWP_VARS has {len(NWP_VARS)} entries")
    _require(NWP_VARS[22] == "TTdMax74" and NWP_VARS[23] == "TTdMean74",
             f"NWP_VARS[22:24] = {NWP_VARS[22:24]}; the variable table drifted "
             f"from radar_nwp_bench/data/index.py")
    if _add_bench_to_path():
        try:
            from radar_nwp_bench.data.index import NWP_VARS as UP_NWP_VARS
            _require(tuple(UP_NWP_VARS) == NWP_VARS,
                     "local NWP_VARS drifted from upstream radar_nwp_bench.data.index.NWP_VARS")
        except ImportError:
            pass

    base = torch.zeros(2, 2, 27, 16, 16)
    ctx_base = nwp_context_from_patch(base)

    # 扰动未选中通道 1 (DVG200)、4 (HT0)、22 (TTdMax74) -> 输出必须逐位不变。
    # 22 这一条专门锁住「把 TTdMax74 当成 TTdMean74」的错位。
    wrong = base.clone()
    wrong[:, 0, 1] = 5.0
    wrong[:, 0, 4] = -3.0
    wrong[:, 0, 22] = 7.0
    gap_wrong = float((nwp_context_from_patch(wrong) - ctx_base).abs().max())
    _require(gap_wrong == 0.0,
             f"perturbing unselected channels (DVG200/HT0) changed context by {gap_wrong:.3e}")

    # 扰动真实选中通道 14 (PWAT) 和 23 (TTdMean74) -> 输出必须改变
    right = base.clone()
    right[:, 0, 14] = 2.5
    right[:, 0, 23] = -1.5
    gap_right = float((nwp_context_from_patch(right) - ctx_base).abs().max())
    _require(gap_right > 0.0,
             "perturbing PWAT(14)/TTdMean74(23) did not change context at all")
    return f"indices={HALO_CONTEXT_INDICES}; unused ch gap={gap_wrong:.1e}, target ch gap={gap_right:.2e}"


def check_nwp_orthogonal_subpaths_and_sample_mask() -> str:
    """halo_only 与 gate_only 可独立开关，且混合 batch 下全零样本不被污染 (D2/M1)。"""
    from qlfn2.qlfn2_model import QLFN2Adapter

    b, t, h, w = 2, 11, 16, 16
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    comp = archive_to_net(raw)
    nwp = torch.rand(b, 2, 27, h, w)

    halo_only = QLFN2Adapter(width=8, use_halo=True, use_nwp_gate=False).eval()
    gate_only = QLFN2Adapter(width=8, use_halo=False, use_nwp_gate=True).eval()
    _require(halo_only.core.halo_proj is not None and halo_only.core.nwp_gate is None,
             "halo_only failed to isolate halo_proj")
    _require(gate_only.core.halo_proj is None and gate_only.core.nwp_gate is not None,
             "gate_only failed to isolate nwp_gate")

    with torch.no_grad():
        p_h = halo_only(comp, nwp)
        p_g = gate_only(comp, nwp)
    _require(p_h.shape == (b, 10, 1, h, w) and p_g.shape == (b, 10, 1, h, w),
             "subpath shapes mismatch")

    # 混合 batch 测试 (D2)：样本 0 有 NWP，样本 1 全零，样本 1 必须等于纯雷达输出
    full = QLFN2Adapter(width=8, use_halo=True, use_nwp_gate=True).eval()
    with torch.no_grad():
        full.core.halo_proj.weight.normal_(0.0, 0.1)
        full.core.halo_proj.bias.normal_(0.0, 0.5)
        full.core.nwp_gate[-1].weight.normal_(0.0, 0.1)
        full.core.nwp_gate[-1].bias.normal_(0.0, 0.5)
        mixed_nwp = nwp.clone()
        mixed_nwp[1].zero_()
        out_mixed = full(comp, mixed_nwp)
        out_zero = full(comp, torch.zeros_like(nwp))
    gap_s1 = float((out_mixed[1] - out_zero[1]).abs().max())
    gap_s0 = float((out_mixed[0] - out_zero[0]).abs().max())
    _require(gap_s1 == 0.0,
             f"zero-NWP sample in a mixed batch was polluted by bias: gap={gap_s1:.3e}")
    _require(gap_s0 > 0.0, "active-NWP sample in a mixed batch did not change")
    return f"halo_only/gate_only ok; mixed-batch zero sample gap={gap_s1:.1e}"


def check_subpath_state_dict_contract() -> str:
    """F36 三个正交臂的 state_dict 键数不同 ⇒ 跨臂权重不可互载。

    ``use_halo`` / ``use_nwp_gate`` 是**构造参数**：关掉子通路会真的不注册
    那几个参数，与 ``motion_mode`` 那种"只换 forward 分支、state_dict 不变"
    的运行时开关完全不同。因此 train 与 eval **必须同传同一份
    ``--model-kwargs``**，否则 checkpoint 要么加载失败，要么静默用错模型。

    同时钉死 D2 的消融语义：任一单通路臂在收到上游 ``--no-nwp`` 的
    **同形状全零张量**时，必须与 ``x_nwp=None`` 以及与"没有 NWP 权重"的
    参考模型逐位相同。三个模型用同一个 seed 分别构造，保证 trunk/head
    权重一致，比较才有意义。
    """
    from qlfn2.qlfn2_model import QLFN2Adapter

    torch.manual_seed(101)
    full = QLFN2Adapter(width=8)
    torch.manual_seed(101)
    halo_only = QLFN2Adapter(width=8, use_nwp_gate=False)
    torch.manual_seed(101)
    gate_only = QLFN2Adapter(width=8, use_halo=False)

    n_full = len(full.state_dict())
    n_halo = len(halo_only.state_dict())
    n_gate = len(gate_only.state_dict())
    _require(n_full > n_halo and n_full > n_gate,
             f"single-path arms did not drop NWP weights: "
             f"full={n_full}, halo_only={n_halo}, gate_only={n_gate}")
    # halo_proj 有 2 个参数、nwp_gate 有 4 个，所以两个单通路臂键数本来就不同；
    # 真正要锁的是「各自只丢掉自己那一组键」，而不是键数相等。
    full_keys = set(full.state_dict())
    dropped_halo = full_keys - set(halo_only.state_dict())
    dropped_gate = full_keys - set(gate_only.state_dict())
    _require(len(dropped_halo) == 4
             and all("nwp_gate" in k for k in dropped_halo),
             f"halo_only dropped the wrong keys: {sorted(dropped_halo)}")
    _require(len(dropped_gate) == 2
             and all("halo_proj" in k for k in dropped_gate),
             f"gate_only dropped the wrong keys: {sorted(dropped_gate)}")

    b, t, h, w = 2, 11, 16, 16
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    comp = archive_to_net(raw)
    zeros = torch.zeros(b, 2, 27, h, w)
    with torch.no_grad():
        ref = full(comp, None)
        for name, model in (("halo_only", halo_only), ("gate_only", gate_only)):
            p_zero = model(comp, zeros)
            p_none = model(comp, None)
            _require(float((p_zero - p_none).abs().max()) == 0.0,
                     f"{name}: --no-nwp (all-zero) != x_nwp=None")
            _require(float((p_zero - ref).abs().max()) == 0.0,
                     f"{name}: all-zero NWP != pure-radar reference")
            _require(model.last_diagnostics.get("nwp_active") is False,
                     f"{name}: an all-zero NWP still activated the NWP path")
    return (f"keys full/halo_only/gate_only = {n_full}/{n_halo}/{n_gate}; "
            f"all-zero NWP == None == pure radar")


def check_model_kwargs_json_contract() -> str:
    """作业脚本里逐字使用的 ``--model-kwargs`` JSON 必须构造出预期的臂。

    上游 ``config.model_kwargs`` 把该 JSON 直接 ``update`` 进 ``build_model``
    的关键字参数；``QLFN2Adapter`` 有 ``**unused``，拼错一个键会被静默吞掉，
    正交归因臂会悄悄变回 ``both_fixed`` 而日志毫无异常。本自检用与
    ``scripts/sbatch_qlfn2_f36.sbatch`` 完全相同的三份 JSON 构造模型，
    并核对 diagnostics 里的两条通路状态。
    """
    import json
    from qlfn2.qlfn2_model import QLFN2Adapter

    specs = {
        "both_fixed": ('{"use_halo": true, "use_nwp_gate": true}', True, True),
        "halo_only": ('{"use_halo": true, "use_nwp_gate": false}', True, False),
        "gate_only": ('{"use_halo": false, "use_nwp_gate": true}', False, True),
    }
    b, t, h, w = 2, 11, 16, 16
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 300.0
    raw[:, :, 9] = 200.0
    comp = archive_to_net(raw)
    nwp = torch.rand(b, 2, 27, h, w)
    seen = []
    for name, (payload, want_halo, want_gate) in specs.items():
        kwargs = json.loads(payload)
        torch.manual_seed(103)
        model = QLFN2Adapter(width=8, **kwargs)
        _require((model.core.halo_proj is not None) == want_halo,
                 f"{name}: halo_proj mismatch for {payload}")
        _require((model.core.nwp_gate is not None) == want_gate,
                 f"{name}: nwp_gate mismatch for {payload}")
        with torch.no_grad():
            model(comp, nwp)
        diag = model.last_diagnostics
        _require(diag.get("halo_connected") is want_halo,
                 f"{name}: halo_connected={diag.get('halo_connected')}")
        _require(diag.get("gate_connected") is want_gate,
                 f"{name}: gate_connected={diag.get('gate_connected')}")
        seen.append(f"{name}={diag.get('nwp_source')}")
    return "; ".join(seen)


def check_f37_lagrangian_two_slice_and_lead0_guard() -> str:
    """F37 回归自检：lead 0 不受演进项污染；第 2 片 NWP 真被消费；双时效张量契约。

    用户给的 F37 规格第 29 条（本工作区实为第 31 条：F36 后基线已是 30 条）。
    覆盖 4 个不变量：
      (1) ``lead 0`` 与纯雷达逐位恒等（``k=0`` 时 vel/source 权重均为 0）；
      (2) ``lead 1..9`` 必须响应 ``x_nwp[:, 1]``（forecast_hour 1h 演变趋势）；
      (3) ``extract_nwp_evolution_tensor`` 为 (B,20,H,W)，且第 2 片只进
          差分半边、不改变分析片半边；通道表对齐上游 27 通道顺序；
      (4) F37 两条通路默认关闭时 state_dict 仍是 F36 的 32 键
          （``check_merge_equivalence.py`` 零修改通过的承重契约）。

    **边界**：文档所说「生消项空间零均值」只在
    ``zm_s = raw_s - raw_s.mean(dim=(-2,-1))`` 这一步成立；乘上湿区包络
    ``wet_env`` 与 ``scale_k`` 后进入雨量场的量不再严格零均值。因此本自检
    **不**把「全图雨量均值不变」写成不变量，避免把结构性约束误报成物理保证。
    """
    from qlfn2.qlfn2_model import (
        QLFN2Adapter, NWP_VARS, NWP_EVOL_INDICES,
        extract_nwp_evolution_tensor,
    )

    torch.manual_seed(41)
    b, t, h, w = 2, 11, 24, 24
    raw = torch.zeros(b, t, 10, h, w)
    raw[:, :, 6] = 350.0
    raw[:, :, 9] = 220.0
    comp = archive_to_net(raw)
    model = QLFN2Adapter(
        width=8, use_halo=False, use_nwp_gate=False,
        use_nwp_vel=True, use_nwp_source=True,
    ).eval()
    with torch.no_grad():
        model.core.nwp_vel_net[-1].weight.normal_(0.0, 0.2)
        model.core.nwp_vel_net[-1].bias.normal_(0.0, 0.2)
        model.core.nwp_source_net[-1].weight.normal_(0.0, 0.2)
        model.core.nwp_source_net[-1].bias.normal_(0.0, 0.5)

    nwp_a = torch.rand(b, 2, 27, h, w)
    nwp_b = nwp_a.clone()
    nwp_b[:, 1] = nwp_a[:, 1] + 5.0
    with torch.no_grad():
        pred_a = model(comp, nwp_a)
        pred_b = model(comp, nwp_b)
        pred_zero = model(comp, torch.zeros_like(nwp_a))

    # (1) lead 0 (+6 min) 必须与纯雷达逐位恒等
    lead0_gap = float((pred_a[:, 0] - pred_zero[:, 0]).abs().max())
    _require(lead0_gap == 0.0,
             f"F37 violated the lead-0 zero-harm guard: gap={lead0_gap:.3e}")
    # (2) 远时效 lead 1..9 必须响应第 2 片 NWP
    late_gap = float((pred_a[:, 1:] - pred_b[:, 1:]).abs().max())
    _require(late_gap > 0.0,
             "F37 is still blind to the second NWP slice x_nwp[:, 1]")
    _require(bool(torch.isfinite(pred_a).all()),
             "the F37 evolution path produced NaN")

    # (3) 双时效演变张量：分析片半边不受 forecast_hour 影响，趋势半边必须变化
    evol_a = extract_nwp_evolution_tensor(nwp_a)
    evol_b = extract_nwp_evolution_tensor(nwp_b)
    _require(tuple(evol_a.shape) == (b, 20, h, w),
             f"unexpected evol shape {tuple(evol_a.shape)}")
    _require(float((evol_a[:, :10] - evol_b[:, :10]).abs().max()) == 0.0,
             "the analysis half of evol must not depend on forecast_hour")
    _require(float((evol_a[:, 10:] - evol_b[:, 10:]).abs().max()) > 0.0,
             "the 1h trend half of evol did not consume x_nwp[:, 1]")
    _require(len(NWP_EVOL_INDICES) == 10
             and NWP_VARS.index("TTdMean74") == 23,
             "the F37 evolution channel table drifted from the upstream order")

    # (4) 默认关闭 F37 通路时必须与 F36 的 state_dict 契约逐位一致
    default = QLFN2Adapter(width=8)
    _require(default.core.nwp_vel_net is None
             and default.core.nwp_source_net is None,
             "F37 paths must be OFF by default (F36 merge equivalence)")
    _require(len(default.state_dict()) == 32,
             f"default state_dict has {len(default.state_dict())} keys, "
             f"expected the F36 32-key contract")
    return (f"lead0 guard gap={lead0_gap:.1e}; slice-1 late gap={late_gap:.3e}; "
            f"evol=(B,20,H,W); default keys=32")


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("halo geometry", check_halo_geometry),
    ("registration matches upstream", check_registration_matches_upstream),
    ("sliding window origins", check_patch_origins),
    ("CAPPI vertical quantities", check_cappi_vertical_quantities),
    ("operator starts as a power law", check_operator_is_a_power_law_at_init),
    ("operator matches the measured relation",
     check_operator_matches_the_measured_relation),
    ("head can reach the 2 mm threshold",
     check_head_can_reach_the_scored_threshold),
    ("operator gives no rain without echo", check_operator_is_zero_without_echo),
    ("motion recovery", check_motion_recovery),
    ("semi-Lagrangian warp", check_warp_alignment),
    ("halo masks point upstream", check_halo_masks_point_upstream),
    ("model shapes", check_model_shapes),
    ("copy-forward branch", check_copy_forward_branch),
    ("halo branch runs", check_halo_branch_runs),
    ("the NWP is actually consumed", check_nwp_is_consumed),
    ("unit bridge matches the benchmark", check_unit_bridge),
    ("motion is tile-consistent", check_motion_is_tile_consistent),
    ("adapter honours the benchmark seam", check_adapter_seam),
    ("adapter wires the halo context", check_adapter_wires_halo_context),
    ("native halo ring contract", check_native_halo_ring_contract),
    ("loss logits are threshold-aligned", check_loss_threshold_alignment),
    ("loss decay: 4D==5D and weights decrease with lead",
     check_loss_decay_broadcast_and_direction),
    ("history last frame is gated (t0 supervision reaches NWP)",
     check_history_last_frame_is_gated),
    ("adapter halo follows NWP availability",
     check_adapter_halo_follows_nwp),
    ("legacy import aliases are pure shims", check_compat_aliases),
    ("runs at both evaluation geometries",
     check_runs_at_both_evaluation_geometries),
    ("NWP channel semantics match (2,3,0,14,23)",
     check_nwp_channel_semantics),
    ("NWP orthogonal subpaths and per-sample mask",
     check_nwp_orthogonal_subpaths_and_sample_mask),
    ("subpath state_dict contract and all-zero ablation",
     check_subpath_state_dict_contract),
    ("--model-kwargs JSON constructs the intended arm",
     check_model_kwargs_json_contract),
    ("F37 two-slice Lagrangian / lead-0 guard / 32-key default",
     check_f37_lagrangian_two_slice_and_lead0_guard),
)


def run_all(verbose: bool = True) -> int:
    """Run every check; return the number of failures."""
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            continue
        if verbose:
            print(f"ok    {name}: {detail}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if run_all() else 0)
    
    

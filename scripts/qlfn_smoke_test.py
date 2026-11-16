#!/usr/bin/env python3
"""Runtime smoke test for QLFN (requires PyTorch; run on the server).

    /home/wf/.conda/envs/radar_nwp/bin/python scripts/qlfn_smoke_test.py

Checks R2-R10 of ``docs/archive/QLFN_SPEC.md`` v1.1 on CPU with tiny patches. It does
NOT train a model, score a dataset, or produce a CSI number: it only verifies
that shapes, gradients, masked statistics and diagnostic isolation behave as
specified.

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the mechanism families are rejected upstream; this is an engineering
self-test of a reference implementation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import qlfn.model as model_module  # noqa: E402
from qlfn.checks import check, render  # noqa: E402
from qlfn.contract import (  # noqa: E402
    KM_PER_PIXEL,
    MAX_DRIFT_KM,
    QLFNConfig,
    drift_scale,
    validate_batch_shapes,
)
from qlfn.data_audit import masked_contingency  # noqa: E402
from qlfn.flow_head import (  # noqa: E402
    LatentConditionalFlowHead,
    conditional_flow_matching_loss,
)
from qlfn.losses import (  # noqa: E402
    masked_contingency_torch,
    masked_coverage_extreme_loss,
)
from qlfn.model import QLagrangeFlowNet, QLFN  # noqa: E402
from qlfn.scale_bridge import DualScaleBridge, plan_for  # noqa: E402

FORWARD_CASES = ((66, 66), (64, 64), (128, 128))
FOLD_CASES = FORWARD_CASES + ((301, 301),)


def main() -> int:
    """Run every runtime check and render the report.

    Returns:
        Process exit code: 0 when every check passed, 1 otherwise.
    """
    torch.manual_seed(0)
    cfg = QLFNConfig()
    rows: list[tuple[str, bool, str]] = []
    rows.append(check("alias QLFN is the model class",
                      QLFN is QLagrangeFlowNet))

    # -- R5: forward shape / non-negativity on real training patches --------
    for height, width in FORWARD_CASES:
        x_radar = torch.randn(1, 11, 10, height, width)
        x_nwp = torch.randn(1, 2, 27, height, width)
        model = QLagrangeFlowNet(cfg)
        model.eval()
        with torch.no_grad():
            pred = model(x_radar, x_nwp)
        expected = (1, 10, 1, height, width)
        rows.append(check(
            f"R5 forward {height}x{width} shape",
            tuple(pred.shape) == expected, str(tuple(pred.shape)),
        ))
        rows.append(check(
            f"R5 forward {height}x{width} finite+nonneg",
            bool(torch.isfinite(pred).all() and (pred >= 0).all()),
        ))
        rows.append(check(
            f"R10 entropy diagnostic in [0,1] {height}x{width}",
            0.0 <= model.last_diagnostics["mean_norm_entropy"] <= 1.0,
            f"{model.last_diagnostics['mean_norm_entropy']:.4f}",
        ))

    # -- R3: the 301x301 scoring grid, with a bounded CPU budget ------------
    light = QLagrangeFlowNet(QLFNConfig(flow_steps=1))
    light.eval()
    with torch.no_grad():
        pred_301 = light(
            torch.randn(1, 11, 10, 301, 301),
            torch.randn(1, 2, 27, 301, 301),
        )
    rows.append(check(
        "R3 forward 301x301 (flow_steps=1 CPU budget)",
        tuple(pred_301.shape) == (1, 10, 1, 301, 301)
        and bool(torch.isfinite(pred_301).all() and (pred_301 >= 0).all()),
        str(tuple(pred_301.shape)),
    ))

    with torch.no_grad():
        none_pred = QLagrangeFlowNet(cfg).eval()(
            torch.randn(1, 11, 10, 66, 66), None
        )
    rows.append(check("x_nwp=None ablation path",
                      tuple(none_pred.shape) == (1, 10, 1, 66, 66)))
    rows.append(check(
        "R2 bad NWP grid rejected",
        bool(validate_batch_shapes((1, 11, 10, 66, 66),
                                  (1, 2, 27, 167, 167), config=cfg)),
    ))
    rows.append(check(
        "R2 missing required target = FAIL",
        bool(validate_batch_shapes((1, 11, 10, 66, 66),
                                   (1, 2, 27, 66, 66),
                                   require_target=True)),
    ))

    # -- R4: REAL fold_channels -> decode_subpixel numerical roundtrip ------
    # v1.0 tested PixelUnshuffle(PixelShuffle(x)) on a synthetic tensor, which
    # is a vacuous test: it never touched either production method.
    bridge = DualScaleBridge(cfg)
    for height, width in FOLD_CASES:
        plan = plan_for(height, width, cfg)
        original = torch.randn(1, cfg.t_out, 1, height, width)
        folded = bridge.fold_channels(original.squeeze(2), plan)
        restored = bridge.decode_subpixel(folded, plan)
        error = float((restored - original).abs().max())
        rows.append(check(
            f"R4 real fold roundtrip {height}x{width} < 1e-6",
            error < 1e-6 and tuple(restored.shape) == original.shape,
            f"max_abs={error:.2e}, folded={tuple(folded.shape)}",
        ))
    bad_channels = torch.randn(1, cfg.subpixel_channels - 9, 22, 22)
    try:
        bridge.decode_subpixel(bad_channels, plan_for(66, 66, cfg))
        channel_rejected = False
    except ValueError:
        channel_rejected = True
    rows.append(check("R6 wrong sub-pixel channels rejected", channel_rejected))

    # -- R9: grid-derived drift scale ---------------------------------------
    rows.append(check(
        "R9 drift scale uses 1.11195 km/px and scales as 1/width",
        abs(drift_scale(66) * 66 - 2.0 * MAX_DRIFT_KM / KM_PER_PIXEL) < 1e-9
        and abs(drift_scale(66) / drift_scale(128) - 128.0 / 66.0) < 1e-9,
        f"66px={drift_scale(66):.6f}, 301px={drift_scale(301):.6f}",
    ))

    # -- R10: diagnostic entropy must not influence the forecast ------------
    r10_model = QLagrangeFlowNet(cfg).eval()
    x_r = torch.randn(1, 11, 10, 66, 66)
    x_n = torch.randn(1, 2, 27, 66, 66)
    original_forward = r10_model.angle_lift.forward

    def perturbed_forward(feat_r, feat_n):
        """Replace the diagnostic entropy map with a very different one."""
        z_fused, _entropy, diagnostics = original_forward(feat_r, feat_n)
        return z_fused, torch.ones_like(_entropy) * 0.99, diagnostics

    with torch.no_grad():
        baseline_pred = r10_model(x_r, x_n)
    r10_model.angle_lift.forward = perturbed_forward
    with torch.no_grad():
        perturbed_pred = r10_model(x_r, x_n)
    r10_model.angle_lift.forward = original_forward
    rows.append(check(
        "R10 entropy perturbation leaves output bit-identical",
        torch.equal(baseline_pred, perturbed_pred),
        "diagnostic quantity is not wired into pred_rain",
    ))

    # -- R8 / T3 / T4: masked composite loss --------------------------------
    pred = torch.rand(2, 10, 1, 66, 66, requires_grad=True)
    target = torch.rand(2, 10, 1, 66, 66) * 3.0
    mask = torch.ones(2, 10)
    mask[:, 5:] = 0.0
    loss, diagnostics = masked_coverage_extreme_loss(pred, target, mask, cfg)
    loss.backward()
    rows.append(check(
        "T4 composite loss finite", bool(torch.isfinite(loss)),
        f"loss={diagnostics['loss_total']:.4f}",
    ))
    rows.append(check(
        "masked gradient reaches prediction",
        pred.grad is not None and bool(torch.isfinite(pred.grad).all())
        and float(pred.grad.abs().sum()) > 0.0,
    ))
    rows.append(check(
        "T3/T4 caveats exposed",
        diagnostics["soft_csi_is_authoritative_csi"] is False
        and diagnostics["coverage_invariance_guaranteed"] is False,
    ))

    p = torch.rand(1, 4, 1, 1, 1)
    t = torch.tensor([[[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]]])
    zero_mask = torch.zeros(1, 4, 1, 1, 1)
    tp, fp, fn = masked_contingency_torch(p, t, zero_mask)
    rows.append(check(
        "R8 fully masked batch yields zero counts",
        float(tp) == float(fp) == float(fn) == 0.0,
    ))
    p_list = [0.9, 0.2, 0.7, 0.4]
    t_list = [1.0, 0.0, 1.0, 1.0]
    m_list = [1, 1, 0, 1]
    pure = masked_contingency(p_list, t_list, m_list)
    torch_counts = masked_contingency_torch(
        torch.tensor(p_list).view(1, 4, 1, 1, 1),
        torch.tensor(t_list).view(1, 4, 1, 1, 1),
        torch.tensor(m_list, dtype=torch.float32).view(1, 4, 1, 1, 1),
    )
    rows.append(check(
        "R8 torch matches canonical pure-python formula",
        all(abs(a - float(b)) < 1e-6 for a, b in zip(pure, torch_counts)),
        f"pure={pure} torch={tuple(float(v) for v in torch_counts)}",
    ))
    try:
        masked_contingency_torch(p, t, torch.full_like(zero_mask, 0.5))
        soft_rejected = False
    except ValueError:
        soft_rejected = True
    rows.append(check("R8 soft mask rejected", soft_rejected))
    try:
        masked_coverage_extreme_loss(pred.detach(), target,
                                     torch.full((2, 10), 0.5), cfg)
        composite_soft_rejected = False
    except ValueError:
        composite_soft_rejected = True
    rows.append(check(
        "R8 composite rejects soft mask", composite_soft_rejected,
    ))

    # -- R7: CFM objective, prior consistency, reachable cond, residual -----
    head = LatentConditionalFlowHead(cfg)
    cond = torch.randn(2, cfg.latent_dim, 22, 22)
    x1 = torch.randn(2, cfg.subpixel_channels, 22, 22)
    cfm_loss = conditional_flow_matching_loss(head, x1, cond)
    cfm_loss.backward()
    grads = [p.grad for p in head.velocity_net.parameters()
             if p.grad is not None]
    rows.append(check(
        "R7 CFM objective finite + gradients",
        bool(torch.isfinite(cfm_loss)) and len(grads) > 0,
        f"loss={float(cfm_loss):.4f}, grad_tensors={len(grads)}",
    ))

    zero_velocity_head = LatentConditionalFlowHead(cfg)
    zero_velocity_head.forward_velocity = (
        lambda z_t, s, c: torch.zeros_like(z_t)
    )
    with torch.no_grad():
        integrated = zero_velocity_head.integrate_ode(cond, n_steps=4)
    rows.append(check(
        "R7 inference starts from the same zero prior as training",
        int(torch.count_nonzero(integrated)) == 0,
        "a randn prior would leave non-zero samples",
    ))
    try:
        zero_velocity_head.integrate_ode(cond, n_steps=0)
        steps_rejected = False
    except ValueError:
        steps_rejected = True
    rows.append(check("R7 invalid ODE step count rejected", steps_rejected))

    cfm_model = QLagrangeFlowNet(cfg)
    cfm_model.extreme_head.zero_grad(set_to_none=True)
    x_r2 = torch.randn(1, 11, 10, 66, 66)
    x_n2 = torch.randn(1, 2, 27, 66, 66)
    captured: dict[str, torch.Tensor] = {}
    original_loss_fn = model_module.conditional_flow_matching_loss

    def spy(head_, x1_, cond_):
        """Capture the training target so its semantics can be verified."""
        captured["x1"] = x1_.detach().clone()
        return original_loss_fn(head_, x1_, cond_)

    model_module.conditional_flow_matching_loss = spy
    zero_target = torch.zeros(1, 10, 1, 66, 66)
    model_loss = cfm_model.flow_training_loss(x_r2, x_n2, zero_target)
    model_loss.backward()
    model_module.conditional_flow_matching_loss = original_loss_fn
    rows.append(check(
        "R7 model CFM path has reachable cond + gradients",
        bool(torch.isfinite(model_loss))
        and any(p.grad is not None
                for p in cfm_model.extreme_head.velocity_net.parameters()),
        f"loss={float(model_loss):.4f}",
    ))
    rows.append(check(
        "R7 target is a non-negative body residual (zero target -> x1=0)",
        int(torch.count_nonzero(captured["x1"])) == 0
        and captured["x1"].shape[1] == cfg.subpixel_channels,
        f"x1={tuple(captured['x1'].shape)}",
    ))
    captured.clear()
    model_module.conditional_flow_matching_loss = spy
    cfm_model.flow_training_loss(x_r2, x_n2,
                                 torch.full((1, 10, 1, 66, 66), 8.0))
    model_module.conditional_flow_matching_loss = original_loss_fn
    rows.append(check(
        "R7 residual target responds to a large rain target",
        int(torch.count_nonzero(captured["x1"])) > 0,
    ))
    try:
        cfm_model.flow_training_loss(x_r2, x_n2, torch.zeros(1, 9, 1, 66, 66))
        bad_target_rejected = False
    except ValueError:
        bad_target_rejected = True
    rows.append(check("R7 malformed target rejected", bad_target_rejected))

    return render(
        rows, "SMOKE_OK (no training, no evaluation, no scores produced)"
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Masked coverage / extreme loss with correct lead-step masking (torch layer).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): losses are engineering support only, not a research claim.

RETRACTED: v1.0 claimed to "fix blueprint defect T2" (masked-out lead steps
counted as false alarms). That was a **hard false positive**. The blueprint's
``p_hit`` and ``t_hit`` are already multiplied by a binary ``mask``, so for
``m in {0,1}`` (``m**2 == m``) the two formulations are algebraically
identical:

    blueprint  fp = p*m * (1 - t*m) = p*m*(1-t)
    this file  fp = p_m * (m - t_m) = p*m*(1-t)

No defect was fixed; this is an equivalent rewrite. What IS added here is a
real contract: the mask must be binary 0/1 (the equivalence does NOT hold for
soft masks), plus a fully-masked regression test. The canonical pure-Python
specification is ``qlfn.data_audit.masked_contingency``; the smoke test
asserts the two implementations agree. See
``docs/QLFN_BLUEPRINT_REVIEW.md`` §1.1.

Caveat (T3/T4): the soft sigmoid CSI is NOT the authoritative hard CSI@2.0,
and the soft wet mask does NOT guarantee coverage invariance. Any claim that
CSI@0.1 / 0.5 or BIAS@0.5 is unchanged by the extreme branch is invalid.
Units are mm per 6-min accumulation, never mm/h.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .contract import QLFNConfig, THRESHOLDS_MM_PER_6MIN

# Source: blueprint §5.4 ``tier_cfg`` (a reference reproduction, NOT the
# authoritative hard CSI@2.0 metric). Thresholds are checked against the
# upstream contract at import time; see CODING_STANDARDS.md §2.
@dataclass(frozen=True)
class TierWeight:
    """One soft-CSI tier.

    Args:
        threshold: Rain threshold in mm per 6-min accumulation.
        weight: Loss weight for this tier.
        temperature: Sigmoid temperature for the soft hit relaxation.
    """

    threshold: float
    weight: float
    temperature: float


TIER_CFG: tuple[TierWeight, ...] = (
    TierWeight(0.1, 0.15, 0.05),
    TierWeight(0.5, 0.25, 0.10),
    TierWeight(2.0, 0.35, 0.25),
    TierWeight(4.0, 0.25, 0.40),
)
EXTREME_THRESHOLD = 2.0  # matches THRESHOLDS_MM_PER_6MIN[3]


def assert_binary_mask(mask: torch.Tensor) -> None:
    """Reject non-binary masks: the two contingency forms differ for them.

    Args:
        mask: Candidate mask tensor.

    Returns:
        None.

    Raises:
        ValueError: If any element is not exactly 0 or 1.
    """
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError(
            "valid_step_mask must be binary 0/1; the blueprint and QLFN "
            "contingency formulations are NOT equivalent for soft masks"
        )


def masked_contingency_torch(
    pred_hit: torch.Tensor, truth_hit: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft contingency counts confined to observed lead steps.

    Args:
        pred_hit: Raw soft hits in ``[0, 1]`` (NOT pre-multiplied by mask).
        truth_hit: Raw hard hits, 0 or 1 (NOT pre-multiplied by mask).
        mask: Binary 1 for observed lead steps, 0 otherwise.

    Returns:
        ``(tp, fp, fn)`` scalars. A fully masked batch returns zeros.

    Raises:
        ValueError: If ``mask`` is not binary 0/1.
    """
    assert_binary_mask(mask)
    m = mask
    p_m = pred_hit * m
    t_m = truth_hit * m
    tp = torch.sum(p_m * t_m)
    fp = torch.sum(p_m * (m - t_m))
    fn = torch.sum((m - p_m) * t_m)
    return tp, fp, fn


def masked_coverage_extreme_loss(
    pred_rain: torch.Tensor,
    target_rain: torch.Tensor,
    valid_step_mask: torch.Tensor | None = None,
    config: QLFNConfig = QLFNConfig(),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Composite loss with masked regression, soft CSI and wet-area regulariser.

    Args:
        pred_rain: ``(B, T_out, 1, H, W)``, mm per 6min.
        target_rain: Same shape as ``pred_rain``.
        valid_step_mask: ``(B, T_out)``; 0 means no ground truth for that step.
            When ``None``, all steps are treated as observed (never invented).
        config: Immutable configuration.

    Returns:
        ``(total_loss, diagnostics)`` where diagnostics are detached floats.

    Raises:
        ValueError: If shapes disagree or the mask has the wrong shape.
    """
    if pred_rain.shape != target_rain.shape:
        raise ValueError("pred_rain and target_rain must have the same shape")
    batch, t_out, _, height, width = pred_rain.shape
    if t_out != config.t_out:
        raise ValueError(f"expected {config.t_out} lead steps, got {t_out}")
    if valid_step_mask is None:
        mask = torch.ones(batch, t_out, 1, 1, 1, device=pred_rain.device,
                          dtype=pred_rain.dtype)
    else:
        if tuple(valid_step_mask.shape) != (batch, t_out):
            raise ValueError("valid_step_mask must be (B, T_out)")
        mask = valid_step_mask.view(batch, t_out, 1, 1, 1).to(
            device=pred_rain.device, dtype=pred_rain.dtype
        )
    assert_binary_mask(mask)
    if not all(t.threshold in THRESHOLDS_MM_PER_6MIN for t in TIER_CFG):
        raise ValueError("soft-CSI tier thresholds must be upstream thresholds")
    valid_denom = torch.clamp(mask.sum() * height * width, min=1.0)

    weights = (
        1.0
        + 4.0 * (target_rain >= 0.5).float()
        + 15.0 * (target_rain >= EXTREME_THRESHOLD).float()
    )
    huber = F.huber_loss(
        torch.log1p(pred_rain), torch.log1p(target_rain),
        reduction="none", delta=0.5,
    )
    loss_reg = torch.sum(mask * weights * huber) / valid_denom

    csi_terms: list[torch.Tensor] = []
    for tier in TIER_CFG:
        p_hit = torch.sigmoid((pred_rain - tier.threshold) / tier.temperature)
        t_hit = (target_rain >= tier.threshold).float()
        tp, fp, fn = masked_contingency_torch(p_hit, t_hit, mask)
        soft_csi = (tp + 1e-5) / (tp + fp + fn + 1e-5)
        csi_terms.append(tier.weight * (1.0 - soft_csi))
    loss_csi = torch.stack(csi_terms).sum()

    wet_pred = torch.sum(torch.sigmoid((pred_rain - 0.5) / 0.1) * mask)
    wet_true = torch.sum((target_rain >= 0.5).float() * mask)
    loss_coverage = torch.abs(wet_pred - wet_true) / (wet_true + 100.0)

    total = loss_reg + 0.8 * loss_csi + 0.2 * loss_coverage
    diagnostics: dict[str, Any] = {
        "loss_total": float(total.detach()),
        "loss_reg": float(loss_reg.detach()),
        "loss_csi": float(loss_csi.detach()),
        "loss_coverage": float(loss_coverage.detach()),
        "soft_csi_is_authoritative_csi": False,
        "coverage_invariance_guaranteed": False,
    }
    return total, diagnostics

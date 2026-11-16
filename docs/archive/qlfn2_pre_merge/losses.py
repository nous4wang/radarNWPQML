"""A loss that is aligned with the score, not with the mean.

The measured reason this exists: the upstream benchmark rescaled the Z-R
field with the MSE-optimal factor and the CSI at 2 mm collapsed to 0.0009,
because MSE and CSI are different objectives.  The same thing happens with a
mean regressor: it predicts E[y | x], which smooths the field, so almost
nothing crosses the 2 mm threshold.  In this project's own probe the mean
regression operator reached a bias of 0.05 at the 2 mm threshold, i.e. it
predicted one twentieth of the observed heavy-rain rate.

The loss here therefore has two parts:

* an **amount** term (Poisson negative log-likelihood), which keeps the
  magnitude calibrated;
* an **event** term (binary cross-entropy on ``y >= tau`` at several
  thresholds), which is what the contingency table actually measures.

The event term is what teaches the model where the heavy rain *is*, and it is
where the phase-routed operator's contribution shows up.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    """The parts of the loss, for logging."""

    total: torch.Tensor
    amount: torch.Tensor
    event: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "loss_amount": float(self.amount.detach()),
            "loss_event": float(self.event.detach()),
        }


class ThresholdAwareLoss(nn.Module):
    """Poisson amount loss plus a weighted event loss.

    Args:
        thresholds: the rainfall thresholds, in mm per 6-minute accumulation.
            2.0 is the load-bearing one: the persistence floor is 0.0924 and
            the oracle ceiling 0.1653 at that threshold.
        weight_amount: the weight of the amount term.
        weight_event: the weight of the event term.
        event_positive_weight: the positive-class weight of the event term.
            None means "derive it from the batch", which keeps the rare
            heavy-rain events from being drowned out.
        lead_decay: the per-lead discount.  A lead of one step gets 1.0 and
            the last lead gets ``lead_decay``.
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        weight_amount: float = 0.5,
        weight_event: float = 1.0,
        event_positive_weight: float | None = None,
        lead_decay: float = 1.0,
    ) -> None:
        super().__init__()
        self.thresholds = tuple(thresholds)
        self.weight_amount = float(weight_amount)
        self.weight_event = float(weight_event)
        self.event_positive_weight = event_positive_weight
        self.lead_decay = float(lead_decay)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossBreakdown:
        """Compute the loss.

        Args:
            pred: (B, T, H, W) the predicted rainfall, mm per 6 minutes.
            target: (B, T, H, W) the truth, the same units.

        Returns:
            :class:`LossBreakdown`.
        """
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        t = pred.shape[1]
        decay = torch.tensor(
            [self.lead_decay ** (t - 1 - k) for k in range(t)],
            dtype=pred.dtype, device=pred.device,
        ).view(1, t, 1, 1)

        p = pred.clamp(min=1e-4)
        y = target.clamp(min=0.0)
        # Poisson negative log-likelihood, up to a constant
        amount = ((p - y * torch.log(p)) * decay).mean()

        event = pred.new_zeros(())
        for thr in self.thresholds:
            # a monotone score from the prediction: the log of the rain rate
            score = torch.log(p)
            pos_weight = self.event_positive_weight
            if pos_weight is None:
                n_pos = (y >= thr).sum()
                n_neg = (y < thr).sum()
                pos_weight = float((n_neg / n_pos.clamp(min=1)).item()) if n_pos > 0 else 1.0
            bce = F.binary_cross_entropy_with_logits(
                score, (y >= thr).to(p.dtype), pos_weight=torch.tensor(
                    pos_weight, dtype=p.dtype, device=p.device),
            )
            event = event + bce * decay.mean()
        event = event / len(self.thresholds)

        total = self.weight_amount * amount + self.weight_event * event
        return LossBreakdown(total=total, amount=amount, event=event)

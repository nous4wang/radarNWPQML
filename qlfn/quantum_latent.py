"""Angle-lift cross-modal interaction, diagnostic only (torch layer).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the quantum-inspired phase / tensor-network family is rejected
upstream. This module is an engineering reproduction of the blueprint's
arithmetic with the physical claims removed.

Scope of the downgrade (defects P1-P5): this module is diagnostic only: it is
not a density matrix, and ``norm_entropy`` is not entanglement entropy in the
quantum sense. No partial trace, no Schmidt decomposition and no Hermitian
observable is constructed. ``prob_k`` is a normalised convolution-response
spectrum; the rank-M projection is an approximation with no error bound.

Scope discipline (aihero Spec review): v1.0 added a learnable ``phase_r`` /
``phase_n`` and a three-channel angle lift that the blueprint never had.
That was scope creep on top of a SEALED mechanism, so it has been removed.
The lift is again the blueprint's two-channel ``[cos(theta), sin(theta)]``;
``readout_weights`` is only a de-physicalised rename of the blueprint's
``schmidt_weights``. ``norm_entropy`` is written to diagnostics and MUST NOT
be used to gate or scale any model output (R10).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .contract import QLFNConfig


class AngleLiftInteraction(nn.Module):
    """Fuse radar and NWP latents through a low-rank angle-lift interaction.

    Args:
        config: Immutable QLFN configuration.
    """

    def __init__(self, config: QLFNConfig = QLFNConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.latent_dim
        self.norm_r = nn.GroupNorm(8, width)
        self.norm_n = nn.GroupNorm(8, width)
        lift_ch = width * 2
        self.u_real = nn.Conv2d(lift_ch, config.rank, 1, bias=False)
        self.u_imag = nn.Conv2d(lift_ch, config.rank, 1, bias=False)
        self.v_real = nn.Conv2d(lift_ch, config.rank, 1, bias=False)
        self.v_imag = nn.Conv2d(lift_ch, config.rank, 1, bias=False)
        self.readout_weights = nn.Parameter(
            torch.randn(config.n_observables, config.rank)
            / math.sqrt(config.rank)
        )
        self.spatial_coupler = nn.Conv2d(
            config.n_observables, width, 3, padding=1
        )

    @staticmethod
    def _angle_lift(x: torch.Tensor) -> torch.Tensor:
        """Map real features to the blueprint's real amplitude pair.

        Args:
            x: Normalised features ``(B, D, H, W)``.

        Returns:
            ``(B, 2D, H, W)`` = ``[cos(theta), sin(theta)]`` with
            ``theta = (pi/2) tanh(x)``. This is a real two-vector, NOT a
            complex amplitude with a phase ``e^{i phi}`` (blueprint defect P4).
        """
        theta = (math.pi / 2.0) * torch.tanh(x)
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)

    def forward(
        self, feat_r: torch.Tensor, feat_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return fused latent, a diagnostic entropy map, and diagnostics.

        Args:
            feat_r: Radar latent ``(B, D, Hc, Wc)``.
            feat_n: NWP latent with the same shape.

        Returns:
            ``(z_fused, norm_entropy, diagnostics)``. ``norm_entropy`` is a
            diagnostic soft mask in ``[0, 1]``, NOT a physical observable.
            It must not be used as evidence for convective triggering.
        """
        psi_r = self._angle_lift(self.norm_r(feat_r))
        psi_n = self._angle_lift(self.norm_n(feat_n))
        ur, ui = self.u_real(psi_r), self.u_imag(psi_r)
        vr, vi = self.v_real(psi_n), self.v_imag(psi_n)
        prod_real = ur * vr - ui * vi
        prod_imag = ur * vi + ui * vr
        energy = prod_real ** 2 + prod_imag ** 2 + 1e-8
        prob_k = energy / energy.sum(dim=1, keepdim=True)
        entropy = -torch.sum(prob_k * torch.log(prob_k + 1e-8), dim=1,
                            keepdim=True)
        norm_entropy = entropy / math.log(self.config.rank)

        obs_real = torch.einsum(
            "brhw,mr->bmhw", prod_real, self.readout_weights
        )
        obs_imag = torch.einsum(
            "brhw,mr->bmhw", prod_imag, self.readout_weights
        )
        readout = obs_real ** 2 + obs_imag ** 2
        z_fused = feat_r + feat_n + self.spatial_coupler(readout)
        diagnostics: dict[str, Any] = {
            "mean_norm_entropy": float(norm_entropy.mean().detach()),
            "top1_mode_dominance": float(prob_k.max(dim=1)[0].mean().detach()),
            "physical_interpretation": "none; diagnostic quantities only",
        }
        return z_fused, norm_entropy, diagnostics

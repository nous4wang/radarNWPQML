"""Reversible spatial folding / unfolding for QLFN (torch layer).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): folding is an engineering operation, not one of the rejected mechanism
families. No research claim is made here.

Blueprint defect S2 is fixed: padding is computed from the real H/W via
``contract.fold_plan`` so any positive training patch works; nothing is
hard-coded to 301 / 303 / 101 / 167.

Defect S5 is fixed by an explicit channel assertion in ``decode_subpixel``
before any ``view``/``PixelShuffle``. NOTE: an earlier revision of this file
claimed to "fix S1" (a supposed 96-vs-90 channel crash). That claim was a
**hard false positive** -- the blueprint's ``out_ch = t_out * 9 = 90`` and
``nn.Identity()`` is shape-legal. It has been retracted; see
``docs/QLFN_BLUEPRINT_REVIEW.md`` §1.1. The real blockers were T1 (no CFM
training objective) and the missing channel assertion.

D3 is respected: x_nwp is already on the radar grid (``models/base.py:14-15``).
There is no 167x167 branch and no outer-ring recycling (``MASTER.md:224``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import QLFNConfig, FoldPlan, fold_plan


class DualScaleBridge(nn.Module):
    """Pad, fold to a coarser latent grid, and decode back losslessly.

    Args:
        config: Immutable QLFN configuration.
    """

    def __init__(self, config: QLFNConfig = QLFNConfig()) -> None:
        super().__init__()
        self.config = config
        self.unshuffle = nn.PixelUnshuffle(config.fold)
        self.shuffle = nn.PixelShuffle(config.fold)
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(config.folded_radar_channels, config.latent_dim, 1,
                      bias=False),
            nn.GroupNorm(8, config.latent_dim),
            nn.GELU(),
            nn.Conv2d(config.latent_dim, config.latent_dim, 3, padding=1),
        )
        self.nwp_encoder = nn.Sequential(
            nn.Conv2d(config.c_nwp * 3, config.latent_dim, 1, bias=False),
            nn.GroupNorm(8, config.latent_dim),
            nn.GELU(),
            nn.Conv2d(config.latent_dim, config.latent_dim, 3, padding=1),
        )

    @staticmethod
    def pad_to_fold(x: torch.Tensor, plan: FoldPlan) -> torch.Tensor:
        """Apply the plan's reflect padding.

        Args:
            x: Tensor ``(B, C, H, W)`` on the original grid.
            plan: Folding plan for that H/W.

        Returns:
            Padded tensor whose H/W are divisible by ``plan.fold``.
        """
        return F.pad(
            x, (plan.pad_left, plan.pad_right, plan.pad_top, plan.pad_bottom),
            mode="reflect",
        )

    def fold_channels(self, x: torch.Tensor, plan: FoldPlan) -> torch.Tensor:
        """Losslessly fold ``(B, C, H, W)`` into ``(B, C*r^2, H/r, W/r)``.

        Args:
            x: Original-grid tensor.
            plan: Folding plan.

        Returns:
            Folded tensor. No learned layer is applied here, so this is
            exactly invertible by :meth:`decode_subpixel`.
        """
        return self.unshuffle(self.pad_to_fold(x, plan))

    def decode_subpixel(
        self, subpixel_logits: torch.Tensor, plan: FoldPlan
    ) -> torch.Tensor:
        """Unfold ``(B, T_out*r^2, Hc, Wc)`` back to ``(B, T_out, 1, H, W)``.

        Args:
            subpixel_logits: Must have exactly ``config.subpixel_channels``
                channels, ordered lead-step by lead-step.
            plan: Folding plan used on the encoder side.

        Returns:
            Full-resolution tensor cropped back to the original H/W.

        Raises:
            ValueError: If the channel count does not match the contract.
        """
        if subpixel_logits.shape[1] != self.config.subpixel_channels:
            raise ValueError(
                f"expected {self.config.subpixel_channels} sub-pixel channels, "
                f"got {subpixel_logits.shape[1]}"
            )
        batch = subpixel_logits.shape[0]
        r2 = self.config.fold ** 2
        folded = subpixel_logits.reshape(
            batch * self.config.t_out, r2,
            subpixel_logits.shape[2], subpixel_logits.shape[3],
        )
        out = self.shuffle(folded)
        top, bottom, left, right = plan.crop
        out = out[:, :, top:bottom, left:right]
        return out.reshape(
            batch, self.config.t_out, 1, plan.height, plan.width
        )

    def decode(
        self, feat: torch.Tensor, head: nn.Module, plan: FoldPlan
    ) -> torch.Tensor:
        """Apply a head then unfold; the head must emit sub-pixel channels.

        Args:
            feat: Latent tensor ``(B, D, Hc, Wc)``.
            head: Module producing ``config.subpixel_channels`` channels.
            plan: Folding plan.

        Returns:
            Full-resolution ``(B, T_out, 1, H, W)`` tensor.
        """
        return self.decode_subpixel(head(feat), plan)

    def encode_radar(
        self, r_frame: torch.Tensor, geo_prior: torch.Tensor, plan: FoldPlan
    ) -> torch.Tensor:
        """Fold one aligned radar frame plus static prior into latent space.

        Args:
            r_frame: ``(B, C_radar, H, W)`` aligned radar features.
            geo_prior: ``(B, geo_channels, H, W)`` static prior.
            plan: Folding plan.

        Returns:
            ``(B, D, Hc, Wc)`` latent tensor.
        """
        folded = self.fold_channels(torch.cat([r_frame, geo_prior], dim=1), plan)
        return self.radar_encoder(folded)

    def encode_nwp(
        self, x_nwp: torch.Tensor, plan: FoldPlan
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Co-locate NWP with the folded radar grid and expose drift surrogates.

        The two NWP valid times and their difference are stacked, padded, and
        area-pooled by the fold factor. Area pooling (not bilinear
        upsampling) keeps the co-location free of interpolation artefacts.

        Args:
            x_nwp: ``(B, 2, C_nwp, H, W)`` already on the radar grid.
            plan: Folding plan.

        Returns:
            ``(core, surrogate)`` where core is ``(B, D, Hc, Wc)`` and
            surrogate is ``(B, 2, H, W)`` built from declared scalar indices.
            The surrogate is NOT a U/V wind vector (``MASTER.md:1186``).
        """
        t0, t1 = x_nwp[:, 0], x_nwp[:, -1]
        stack = torch.cat([t0, t1, t1 - t0], dim=1)
        pooled = F.avg_pool2d(
            self.pad_to_fold(stack, plan),
            kernel_size=self.config.fold, stride=self.config.fold,
        )
        core = self.nwp_encoder(pooled)
        indices = list(self.config.wind_surrogate_indices)
        surrogate = x_nwp[:, 0][:, indices]
        return core, surrogate


def plan_for(height: int, width: int, config: QLFNConfig = QLFNConfig()) -> FoldPlan:
    """Convenience wrapper so callers never compute padding themselves.

    Args:
        height: Input height.
        width: Input width.
        config: Immutable configuration.

    Returns:
        A :class:`FoldPlan` for ``config.fold``.
    """
    return fold_plan(height, width, config.fold)

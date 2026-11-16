"""QLagrangeFlowNet (display name QLag-FlowNet, code id QLFN).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the Lagrangian-alignment, quantum-inspired and generative
flow-matching families were rejected upstream. This is an **auditable
reference implementation** only: it does not train, evaluate, modify the
upstream repository, or claim any score. See ``docs/QLFN_SPEC.md`` and
``docs/QLFN_BLUEPRINT_REVIEW.md``.

Engineering changes relative to the blueprint, stated precisely:

* D2 / S2 - any positive H/W is padded to a foldable grid via
  ``contract.fold_plan``; nothing is hard-coded to 301.
* S4 / D1 - the two drift inputs are *declared scalar surrogates*
  (``DVG850``, ``WS925``), never U/V wind components.
* S5 / R6 - ``decode_subpixel`` asserts the sub-pixel channel count.
* S6 / R9 - drift is scaled by ``contract.drift_scale(width)``, which uses the
  upstream grid spacing 1.11195 km/px (``MASTER.md:1556``), not the
  blueprint's ``12.0 / 150.0`` or a 1 km/px approximation.
* T1 / R7 - a CFM training objective exists in ``flow_head`` and is reachable
  through :meth:`QLagrangeFlowNet.flow_training_loss`, with a consistent zero
  prior and a *residual* target matching the inference composition.

RETRACTED: an earlier revision claimed to fix "S1" (a 96-vs-90 channel crash)
and "T2" (masked false alarms). Both were hard false positives; see
``docs/QLFN_BLUEPRINT_REVIEW.md`` §1.1. No such fixes are claimed here.

R10: the angle-lift entropy is written to ``last_diagnostics`` ONLY. It is
never interpolated, gated, or multiplied into ``pred_rain``. The physical
reading of that quantity was invalidated upstream (P1-P5).

T3 is NOT fixed as a claim: the wet mask is soft and coverage invariance is
explicitly disclaimed in :meth:`QLagrangeFlowNet.forward`'s docstring.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import (
    T_NWP,
    FoldPlan,
    QLFNConfig,
    drift_scale,
    validate_batch_shapes,
)
from .flow_head import (
    LatentConditionalFlowHead,
    conditional_flow_matching_loss,
)
from .geo_prior import StaticGeoRadarPrior
from .quantum_latent import AngleLiftInteraction
from .scale_bridge import DualScaleBridge, plan_for


class MultiFrameAccumulator(nn.Module):
    """Learnable temporal integration with a grid-scaled drift prior.

    Args:
        config: Immutable QLFN configuration.
    """

    def __init__(self, config: QLFNConfig = QLFNConfig()) -> None:
        super().__init__()
        self.config = config
        self.temporal_weights = nn.Parameter(
            torch.zeros(1, config.t_in, 1, 1, 1)
        )
        with torch.no_grad():
            self.temporal_weights[0, -3:, 0, 0, 0] = torch.tensor(
                [0.2, 0.5, 0.3]
            )
        self.motion = nn.Sequential(
            nn.Conv2d(config.t_in * config.c_radar, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, config.c_radar, 3, padding=1),
        )
        self.drift_net = nn.Sequential(
            nn.Conv2d(config.c_radar + 2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(
        self, x_radar: torch.Tensor, surrogate: torch.Tensor
    ) -> torch.Tensor:
        """Integrate the history frames and apply a bounded drift warp.

        Args:
            x_radar: ``(B, T_in, C_radar, H, W)``.
            surrogate: ``(B, 2, H, W)`` declared scalar drift prior. These
                are NOT U/V wind components.

        Returns:
            ``(B, C_radar, H, W)`` aligned radar feature frame.
        """
        batch, _, channels, height, width = x_radar.shape
        weights = F.softmax(self.temporal_weights, dim=1)
        accumulated = torch.sum(x_radar * weights, dim=1)
        motion = self.motion(
            x_radar.reshape(batch, self.config.t_in * channels, height, width)
        )
        combined = accumulated + motion
        # Normalised-grid bound for MAX_DRIFT_KM at the actual patch width and
        # the upstream 1.11195 km/px spacing (see contract.drift_scale).
        scale = drift_scale(width)
        delta = torch.tanh(
            self.drift_net(torch.cat([combined, surrogate], dim=1))
        ) * scale
        yy = torch.linspace(-1.0, 1.0, height, device=x_radar.device,
                            dtype=x_radar.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=x_radar.device,
                            dtype=x_radar.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )
        sample_grid = base - delta.permute(0, 2, 3, 1)
        return F.grid_sample(
            combined, sample_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )


class QLagrangeFlowNet(nn.Module):
    """Reference implementation of the blueprint architecture.

    Args:
        config: Immutable QLFN configuration.
    """

    def __init__(self, config: QLFNConfig = QLFNConfig()) -> None:
        super().__init__()
        self.config = config
        self.geo_prior = StaticGeoRadarPrior(config.geo_channels)
        self.scale_bridge = DualScaleBridge(config)
        self.accumulator = MultiFrameAccumulator(config)
        self.angle_lift = AngleLiftInteraction(config)
        self.body_head = nn.Conv2d(
            config.latent_dim, config.subpixel_channels, 1
        )
        self.extreme_head = LatentConditionalFlowHead(config)
        self.last_diagnostics: dict[str, Any] = {}

    def _encode(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None,
        target_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, FoldPlan]:
        """Run the shared input-to-latent path for forward and CFM training.

        The diagnostic entropy map returned by ``AngleLiftInteraction`` is
        deliberately discarded: it is summarised into ``last_diagnostics``
        only and can never influence the forecast (R10).

        Args:
            x_radar: ``(B, 11, 10, H, W)``.
            x_nwp: ``(B, 2, 27, H, W)`` or ``None`` for an all-zero ablation.
            target_shape: Shape of ``y_rain``. When supplied, a missing or
                malformed target is a hard failure (missing evidence = FAIL).

        Returns:
            ``(fused_latent, fold_plan)``.

        Raises:
            ValueError: If the IO contract is violated.
        """
        batch, _, _, height, width = x_radar.shape
        if x_nwp is None:
            x_nwp = torch.zeros(
                batch, T_NWP, self.config.c_nwp, height, width,
                device=x_radar.device, dtype=x_radar.dtype,
            )
        problems = validate_batch_shapes(
            tuple(x_radar.shape), tuple(x_nwp.shape), target_shape,
            config=self.config, require_target=target_shape is not None,
        )
        if problems:
            raise ValueError("; ".join(problems))

        plan = plan_for(height, width, self.config)
        nwp_core, surrogate = self.scale_bridge.encode_nwp(x_nwp, plan)
        aligned = self.accumulator(x_radar, surrogate)
        geo = self.geo_prior(x_radar[:, -1])
        radar_core = self.scale_bridge.encode_radar(aligned, geo, plan)
        fused, _entropy, diagnostics = self.angle_lift(radar_core, nwp_core)
        self.last_diagnostics = diagnostics
        return fused, plan

    def forward(
        self, x_radar: torch.Tensor, x_nwp: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map inputs to a non-negative rain forecast.

        Args:
            x_radar: ``(B, 11, 10, H, W)`` on an arbitrary patch grid.
            x_nwp: ``(B, 2, 27, H, W)`` on the SAME grid, or ``None`` for
                an all-zero ablation input.

        Returns:
            ``(B, 10, 1, H, W)`` in mm per 6-min accumulation, clamped >= 0.

        Raises:
            ValueError: If the IO contract is violated.

        Note:
            The wet mask is soft, so this composition does NOT guarantee that
            CSI@0.1 / CSI@0.5 / BIAS@0.5 are unchanged by the extreme branch
            (blueprint defect T3). No improvement claim is associated with
            this output. The angle-lift entropy is diagnostic only (R10).
        """
        fused, plan = self._encode(x_radar, x_nwp)
        body = F.softplus(
            self.scale_bridge.decode(fused, self.body_head, plan)
        )
        wet_mask = torch.sigmoid((body - 0.25) * 12.0)
        extreme_subpixel = self.extreme_head.integrate_ode(
            fused, n_steps=self.config.flow_steps
        )
        extreme = F.relu(
            self.scale_bridge.decode_subpixel(extreme_subpixel, plan)
        )
        pred_rain = body + wet_mask * extreme
        return torch.clamp(pred_rain, min=0.0)

    def flow_training_loss(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None,
        target_rain: torch.Tensor,
    ) -> torch.Tensor:
        """CFM velocity-matching loss on the residual above the body forecast.

        The conditioning latent is produced by the model's own encoding path,
        so an external caller never has to fabricate a latent tensor. ``x1``
        is ``clamp(target_rain - body, min=0)`` folded to sub-pixel space,
        which matches the inference composition
        ``pred_rain = body + wet_mask * relu(decode(z))``. Both the target
        and the conditioning are detached: this objective trains the extreme
        velocity head only. Calling it is REQUIRED for the extreme branch to
        learn anything, but this reference implementation never calls it
        itself (no training is performed here).

        Args:
            x_radar: ``(B, 11, 10, H, W)``.
            x_nwp: ``(B, 2, 27, H, W)`` or ``None``.
            target_rain: ``(B, 10, 1, H, W)``, mm per 6-min accumulation.

        Returns:
            Scalar velocity-matching loss.

        Raises:
            ValueError: If the IO contract (including ``y_rain``) is violated.
        """
        fused, plan = self._encode(
            x_radar, x_nwp, tuple(target_rain.shape)
        )
        with torch.no_grad():
            body = F.softplus(
                self.scale_bridge.decode(fused, self.body_head, plan)
            )
            residual = torch.clamp(target_rain - body, min=0.0)
            x1 = self.scale_bridge.fold_channels(residual.squeeze(2), plan)
        return conditional_flow_matching_loss(
            self.extreme_head, x1, fused.detach()
        )


QLFN = QLagrangeFlowNet

__all__ = ["QLagrangeFlowNet", "QLFN", "MultiFrameAccumulator"]

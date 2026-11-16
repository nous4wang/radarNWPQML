"""Conditional flow-matching head with an explicit training objective.

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the generative / flow-matching family is rejected upstream. This module
is an engineering reference only and makes no research claim.

Defect T1 fixed: the blueprint shipped ``forward_velocity`` and
``integrate_ode`` but no training objective, so the ODE output was an
untrained random field. :func:`conditional_flow_matching_loss` implements the
rectified-flow target ``u = x_1 - z_0`` on the linear path
``z_s = (1-s) z_0 + s x_1``. Without calling it, this head learns nothing.

Prior consistency (aihero/ARIS review): v1.0 trained with ``z0 ~ N(0, I)``
but integrated from a zero prior at inference, so the velocity field was
trained and evaluated on disjoint paths. Both now use the SAME zero prior.
Consequence, stated honestly: with a Dirac prior at zero this is a
*deterministic* rectified path, not a stochastic generative sampler. It
produces a single conditional residual and makes no distributional claim.
The speculative ``deterministic_eval=False`` branch was removed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import QLFNConfig


class LatentConditionalFlowHead(nn.Module):
    """Velocity field over sub-pixel residual space.

    Args:
        config: Immutable QLFN configuration.
    """

    def __init__(self, config: QLFNConfig = QLFNConfig()) -> None:
        super().__init__()
        self.config = config
        self.out_ch = config.subpixel_channels
        width = config.latent_dim
        self.velocity_net = nn.Sequential(
            nn.Conv2d(self.out_ch + width + 1, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, self.out_ch, 1),
        )

    def forward_velocity(
        self, z_t: torch.Tensor, s: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ``v_theta(z_s, s, cond)``.

        Args:
            z_t: State ``(B, out_ch, Hc, Wc)``.
            s: Time ``(B, 1)`` in ``[0, 1]``.
            cond: Conditioning latent ``(B, D, Hc, Wc)``.

        Returns:
            Velocity with the same shape as ``z_t``.
        """
        batch = z_t.shape[0]
        s_map = s.view(batch, 1, 1, 1).expand(
            batch, 1, z_t.shape[2], z_t.shape[3]
        )
        return self.velocity_net(torch.cat([z_t, cond, s_map], dim=1))

    def integrate_ode(
        self, cond: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """Midpoint integration from the zero prior used during training.

        Args:
            cond: Conditioning latent ``(B, D, Hc, Wc)``.
            n_steps: Positive number of ODE steps. Callers use
                ``config.flow_steps``; there is no hidden default.

        Returns:
            ``(B, out_ch, Hc, Wc)`` sub-pixel residual tensor.

        Raises:
            ValueError: If ``n_steps`` is not positive.

        Note:
            The zero prior is shared with
            :func:`conditional_flow_matching_loss`. A fixed prior gives
            bit-exact replay, not an independent sample
            (see ``analyze_seed_sensitivity.py``).
        """
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        steps = n_steps
        batch, _, height, width = cond.shape
        z = torch.zeros(batch, self.out_ch, height, width,
                        device=cond.device, dtype=cond.dtype)
        ds = 1.0 / float(steps)
        for i in range(steps):
            s0 = torch.full((batch, 1), i * ds, device=cond.device,
                            dtype=cond.dtype)
            v0 = self.forward_velocity(z, s0, cond)
            z_mid = z + 0.5 * ds * v0
            s_mid = torch.full((batch, 1), (i + 0.5) * ds,
                               device=cond.device, dtype=cond.dtype)
            z = z + ds * self.forward_velocity(z_mid, s_mid, cond)
        return z


def conditional_flow_matching_loss(
    head: LatentConditionalFlowHead,
    x1: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    """Rectified-flow training objective for the velocity field.

    Args:
        head: Velocity head whose parameters should receive gradients.
        x1: Target sub-pixel residual ``(B, out_ch, Hc, Wc)``. For QLFN this
            must be the folded *residual above the body forecast*, not the
            full rain target (see ``QLagrangeFlowNet.flow_training_loss``).
        cond: Conditioning latent ``(B, D, Hc, Wc)``.

    Returns:
        Scalar mean-squared velocity error.

    Raises:
        ValueError: If ``x1`` channel count differs from ``head.out_ch``.

    Note:
        ``z0`` is the zero prior, exactly as in :meth:`integrate_ode`. With a
        Dirac prior ``target_velocity`` reduces to ``x1`` and the path is
        deterministic; no generative/distributional claim is attached.
    """
    if x1.shape[1] != head.out_ch:
        raise ValueError(
            f"x1 needs {head.out_ch} channels, got {x1.shape[1]}"
        )
    z0 = torch.zeros_like(x1)
    s = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype)
    s_map = s.view(-1, 1, 1, 1)
    z_s = (1.0 - s_map) * z0 + s_map * x1
    target_velocity = x1 - z0
    predicted = head.forward_velocity(z_s, s, cond)
    return torch.mean((predicted - target_velocity) ** 2)

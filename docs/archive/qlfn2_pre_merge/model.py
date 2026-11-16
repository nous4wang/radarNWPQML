"""The qlfn2 model: a structured t0 rain field, advected onto the ten leads.

The seam is the benchmark's: ``model(x_radar, x_nwp) -> (B, T_out, 1, H, W)``,
so ``scripts/evaluate.py`` can score it with no special case, on the same
full-grid Hann stitching and the same verifier as every other arm.

What is different from the upstream single-frame QPE arm:

* the rain field is produced by the **phase-routed operator** of
  :mod:`qlfn2.operator`, whose weights come from the observed CAPPI vertical
  structure rather than from a generic ten-channel convolution;
* the operator reads the **whole eleven-frame history**, not only the issue
  frame: the 6-minute and 30-minute reflectivity trends are what separate a
  growing cell from a decaying one, and the upstream arm that reaches
  CSI@2.0 = 0.1226 uses the history window too;
* the motion that carries the field to the ten leads is **measured from the
  radar** (:mod:`qlfn2.motion`), not proxied by NWP wind speed;
* the optional halo context is the **narrow upstream annulus** of
  :mod:`qlfn2.halo`, not the whole discarded ring;
* the same RainHead can be applied to the history frames, which gives the
  zero-advection supervision the upstream route R1 asks for -- the archive has
  the label at every history timestamp, so the supervision is available
  without touching the test-time input contract.

Units
-----
The core works on the **archive's raw units** (0.1 dBZ, 0.1 km, 0.1 kg m-2),
which is the scale every measured number in ``MASTER_CONTROL_ROOM.md``
section 2 was computed on.  The benchmark's harness, on the other hand,
hands the network a *compressed* field; :mod:`qlfn2.units` converts between
the two and :mod:`qlfn2.adapter` is the only caller that crosses the line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .cappi import scaled_vertical_quantities
from .geometry import RADAR_VARS
from .motion import Motion, MotionField, advect_sequence, estimate_motion_field
from .operator import StructuredRainHead
from .units import archive_to_net


def _residual_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(1, channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(1, channels),
    )


@dataclass
class ModelOutput:
    """The model's output and the diagnostics worth logging.

    Attributes:
        rain: (B, T_out, 1, H, W) the predicted rainfall, mm per 6 minutes.
        rain_t0: (B, 1, H, W) the field at the issue time.
        rain_history: (B, T_in, 1, H, W) or None, the same head applied to
            the history frames -- the zero-advection supervision of route R1.
        motion: the motion the field was advected with.
        diagnostics: a dict of scalars and tensors for the log.
    """

    rain: torch.Tensor
    rain_t0: torch.Tensor
    rain_history: torch.Tensor | None = None
    motion: "MotionField | None" = None
    diagnostics: dict = field(default_factory=dict)


class QLFN2Model(nn.Module):
    """The structured QPE plus advection model.

    Args:
        in_len: how many history frames are given (11).
        out_len: how many forecast frames are produced (10).
        use_halo: whether the NWP halo context is fed to the trunk.
        use_history_head: whether the same head is also applied to the
            history frames, which produces the zero-advection supervision.
        z_min: the reflectivity floor of the operator, in dBZ.
        width: the trunk's channel width.
        motion_mode: how the advection velocity is measured -- ``"corr"``
            (the cross-correlation peak, which gives the same answer on
            every tile of a sliding window) or ``"centroid"`` (the echo
            centroid, which does not) -- or ``"none"`` to switch the
            advection off and score the copy-forward arm.
        max_shift: the largest displacement the correlation search covers,
            in pixels per 6-minute step.
    """

    def __init__(
        self,
        in_len: int = 11,
        out_len: int = 10,
        use_halo: bool = False,
        use_history_head: bool = True,
        z_min: float = 5.0,
        width: int = 48,
        motion_mode: str = "corr",
        max_shift: int = 8,
    ) -> None:
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.use_halo = bool(use_halo)
        self.use_history_head = bool(use_history_head)
        self.motion_mode = str(motion_mode)
        self.max_shift = int(max_shift)

        n_raw = len(RADAR_VARS)
        n_vertical = 10                       # qlfn2.cappi.VERTICAL_QUANTITY_NAMES
        # the issue frame, the 6-minute trend, the 30-minute trend and the
        # ten named vertical quantities
        self.n_features = n_raw * 3 + n_vertical

        self.trunk = nn.Sequential(
            nn.Conv2d(self.n_features, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            _residual_block(width),
            nn.GELU(),
        )
        # the head consumes the trunk output, so it sees ``width`` channels
        self.head = StructuredRainHead(width, z_min=z_min, hidden=width)
        # the halo context arrives as a handful of scalars per frame; it is
        # injected as a constant offset, one value per feature channel, so
        # that it can modulate the trunk without pretending to be a spatial
        # field.
        #
        # The projection must land on ``n_features`` and not on ``width``:
        # the offset is added to the *feature* stack, which has
        # ``n_features`` channels, while ``width`` is the trunk's own
        # channel count.  Projecting to ``width`` produced a
        # ``RuntimeError`` the first time the halo branch was ever
        # executed -- the branch had never run, because the benchmark's
        # harness never passes a halo context (see
        # ``qlfn2.checks.check_halo_branch_runs``).
        self.halo_proj = nn.Linear(20, self.n_features) if use_halo else None

    # -- features ----------------------------------------------------------
    @staticmethod
    def causal_features(net: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
        """The feature stack of every frame, built causally.

        For frame ``k`` the stack is

            [ net(k), net(k) - net(k-1), net(k) - net(k-5), vq(k) ]

        where ``net`` is the compressed radar field and ``vq`` the scaled
        vertical quantities.  The two differences are the 6-minute and the
        30-minute reflectivity trend: they are what tells a growing cell
        from a decaying one, and the upstream arm that reaches
        CSI@2.0 = 0.1226 reads the history window rather than only the issue
        frame.

        Args:
            net: (B, T, 10, H, W) the compressed radar field.
            vq: (B, T, 10, H, W) the scaled vertical quantities.

        Returns:
            (B, T, 40, H, W).
        """
        b, t = net.shape[:2]
        prev1 = torch.cat([net[:, :1], net[:, :-1]], dim=1)
        trend_6 = net - prev1
        idx = torch.arange(t, device=net.device)
        prev5 = net[:, (idx - 5).clamp(min=0)]
        trend_30 = net - prev5
        return torch.cat([net, trend_6, trend_30, vq], dim=2)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None = None,
        halo_context: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Run the model.

        Args:
            x_radar: (B, T_in, 10, H, W) the history, in the archive's raw
                units (0.1 dBZ, 0.1 km, 0.1 kg m-2).
            x_nwp: (B, T_nwp, 27, H, W) the NWP, already registered onto the
                radar grid by the benchmark's loader.  Unused unless
                ``use_halo`` is on; the halo context is passed in separately
                so that the model never has to redo the registration.
            halo_context: (B, 20) the halo scalars, when ``use_halo`` is on.

        Returns:
            :class:`ModelOutput`.
        """
        if x_radar.dim() != 5:
            raise ValueError(f"expected (B, T, 10, H, W), got {tuple(x_radar.shape)}")
        b, t = x_radar.shape[:2]
        if t < 2:
            raise ValueError("the motion estimate needs at least two frames")

        net = archive_to_net(x_radar)
        vq = scaled_vertical_quantities(x_radar)
        feats = self.causal_features(net, vq)

        t0_feats = feats[:, -1]
        if self.halo_proj is not None and halo_context is not None:
            cond = self.halo_proj(halo_context)                  # (B, width)
            t0_feats = t0_feats + cond[:, :, None, None]

        trunk = self.trunk(t0_feats)
        op = self.head(x_radar[:, -1], trunk)
        rain_t0 = op.rain.unsqueeze(1)                          # (B, 1, H, W)

        motion = estimate_motion_field(x_radar, mode=self.motion_mode,
                                       max_shift=self.max_shift)
        if self.motion_mode == "none":
            # The copy-forward arm: the t0 rain field is repeated across the
            # ten leads, with no displacement at all.
            #
            # The output contract is (B, T_out, C, H, W).  ``rain_t0`` is
            # (B, C, H, W) -- its single axis is the *channel*, not a lead --
            # so the lead axis has to be inserted before the field is
            # expanded.  Expanding a 4-D tensor into a 5-D target puts the
            # channel count on the lead axis, which is the bug this comment
            # exists to prevent.
            rain = rain_t0.unsqueeze(1).expand(-1, self.out_len, -1, -1, -1).contiguous()
        else:
            rain = advect_sequence(rain_t0, motion, self.out_len)

        rain_hist = None
        if self.use_history_head:
            # one batched pass over the whole window: the trunk sees
            # (B*T) images, which is the same arithmetic as T separate calls
            # but one kernel launch.
            hist_trunk = self.trunk(feats.reshape(b * t, *feats.shape[2:]))
            hist_trunk = hist_trunk.reshape(b, t, *hist_trunk.shape[1:])
            hist = []
            for k in range(t):
                o = self.head(x_radar[:, k], hist_trunk[:, k])
                hist.append(o.rain.unsqueeze(1).unsqueeze(1))
            rain_hist = torch.cat(hist, dim=1)

        diagnostics = {
            "motion_vx": float(motion.vx[0]),
            "motion_vy": float(motion.vy[0]),
            "motion_usable": bool(motion.usable[0]),
            "phase_weights_mean": op.weights.detach().mean(dim=(0, 2, 3)).tolist(),
            "operator_a": op.params["a"].tolist(),
            "operator_b": op.params["b"].tolist(),
        }
        return ModelOutput(
            rain=rain,
            rain_t0=rain_t0,
            rain_history=rain_hist,
            motion=motion,
            diagnostics=diagnostics,
        )

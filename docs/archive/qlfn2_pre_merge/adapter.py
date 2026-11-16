"""The upstream seam: qlfn2 behind the benchmark's model interface.

The benchmark's harness is the thing that makes a comparison meaningful:
every model is an adapter behind one interface, so the training loop, the
metrics and the data pipeline never learn a model's particularities.  This
module puts qlfn2 behind that interface, so the benchmark's own
``scripts/train.py`` and ``scripts/evaluate.py`` can train and score it
without a single line being changed in the benchmark.

What the adapter has to reconcile
---------------------------------
* **Units.**  The harness hands the model a *compressed* radar field
  (``log1p``-scaled, see :mod:`qlfn2.units`); the qlfn2 physics needs the
  archive's raw units.  The conversion happens here and nowhere else.
* **The output contract.**  ``(B, T_out, 1, H, W)`` in millimetres per
  6-minute accumulation.
* **The history supervision.**  The upstream training loop reads
  ``model.last_qpe_pred`` and compares it with the observed history rainfall
  when ``--qpe-weight`` is set (route R1).  The adapter publishes the
  qlfn2 head's estimate of the history frames, so that route comes for
  free.

What the adapter deliberately does **not** do
---------------------------------------------
It does not touch the data pipeline, the stitching, the verifier or the
split.  Those stay the benchmark's, which is what makes the numbers
comparable with the ones in the benchmark's own ``MASTER.md``.
"""
from __future__ import annotations

import torch

from .bench import require_bench
from .model import QLFN2Model
from .units import net_to_archive

require_bench()

from radar_nwp_bench.models.base import BasePrecipitationModel  # noqa: E402


class QLFN2Adapter(BasePrecipitationModel):
    """The qlfn2 core behind the benchmark's model seam.

    Args:
        in_radar_c: radar channels the harness will deliver (10).
        in_nwp_c: NWP channels the harness will deliver (27).
        in_len: history frames (11).
        out_len: forecast frames (10).
        out_channels: output channels per frame (1, the rainfall).
        width: the trunk's channel width.
        z_min: the reflectivity floor of the operator, in dBZ.
        motion_mode: ``"corr"``, ``"centroid"`` or ``"none"``.
        use_history_head: whether the head is also applied to the history
            frames, which publishes the R1 supervision.
    """

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        out_channels: int = 1,
        width: int = 48,
        z_min: float = 5.0,
        motion_mode: str = "corr",
        max_shift: int = 8,
        use_halo: bool = False,
        use_history_head: bool = True,
        **unused: object,
    ) -> None:
        super().__init__(
            in_radar_c=in_radar_c,
            in_nwp_c=in_nwp_c,
            in_len=in_len,
            out_len=out_len,
            out_channels=out_channels,
        )
        self.core = QLFN2Model(
            in_len=in_len,
            out_len=out_len,
            use_halo=use_halo,
            use_history_head=use_history_head,
            z_min=z_min,
            width=width,
            motion_mode=motion_mode,
            max_shift=max_shift,
        )
        #: The upstream training loop reads this attribute when
        #: ``--qpe-weight`` is set, and compares it with the observed
        #: history rainfall.  It is refreshed on every forward pass.
        self.last_qpe_pred: torch.Tensor | None = None

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: torch.Tensor | None = None,
        halo_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the ten forecast frames.

        Args:
            x_radar: (B, T_in, C_radar, H, W) the compressed radar history.
            x_nwp: (B, T_nwp, C_nwp, H, W) the registered NWP, or ``None`` --
                the base class has already replaced ``None`` with a zero
                tensor of the right shape.
            halo_context: (B, 20) the halo scalars, when the halo arm is on.

        Returns:
            (B, T_out, 1, H, W), millimetres per 6-minute accumulation.
        """
        raw = net_to_archive(x_radar)
        out = self.core(raw, x_nwp=x_nwp, halo_context=halo_context)
        self.last_qpe_pred = out.rain_history
        return out.rain


__all__ = ["QLFN2Adapter"]

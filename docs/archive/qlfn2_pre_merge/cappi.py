"""The six CAPPI levels turned into named vertical quantities.

Measured facts this module is built on (MASTER_CONTROL_ROOM.md section 2, F1):

* ``CAP20..CAP70`` are **height levels at 2, 3, 4, 5, 6 and 7 km**, not dBZ
  thresholds.  The evidence: the fraction of pixels with signal is almost the
  same at all six levels (0.2056 / 0.2146 / 0.2069 / 0.2116 / 0.2138 /
  0.2018), whereas a threshold would shrink monotonically; and wherever the
  echo top exceeds 7 km, 96.1 % of the pixels carry a CAP70 signal, while the
  0.1 km reading of the echo top is empty everywhere.
* The column profile has an interior maximum in 53.6 % of the live columns,
  i.e. a melting band exists and is observable.

Why this matters: none of the 25 upstream models feeds the six levels to the
network as a vertical structure.  ``furecast`` uses a CAPPI-level
self-consistency *penalty*; ``physical_qpe`` concatenates the ten channels and
lets a generic CNN rediscover the relations.  Here the vertical structure is
turned into explicit, physically named quantities, so that the rain operator
can use them instead of guessing.

Units in the packed archive: CAPPI and CR are 0.1 dBZ, ET is 0.1 km, VIL is
0.1 kg/m2.  Everything returned here is in physical units.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import CAPPI_LEVELS_KM, RADAR_VARS

#: Index of each CAPPI level inside the radar channel axis.
CAPPI_INDICES: tuple[int, ...] = tuple(RADAR_VARS.index(f"CAP{d}") for d in (20, 30, 40, 50, 60, 70))
CR_INDEX: int = RADAR_VARS.index("CR")
ET_INDEX: int = RADAR_VARS.index("ET")
HBR_INDEX: int = RADAR_VARS.index("HBR")
VIL_INDEX: int = RADAR_VARS.index("VIL")

#: The names of the vertical quantities, in the order ``stack`` returns them.
VERTICAL_QUANTITY_NAMES: tuple[str, ...] = (
    "cap_top_km",        # height of the highest level that carries signal
    "cap_top_dbz",       # dBZ at that level
    "cap_max_dbz",       # column maximum
    "cap_argmax_km",     # height of the column maximum (the bright band sits here)
    "cap_bright_dbz",    # column maximum minus the lowest level: band strength
    "cap_nz_levels",     # how many of the six levels carry signal
    "cap_col_dbz_sum",   # column integral of dBZ over the levels
    "cap_slope_dbz_km",  # vertical gradient of dBZ, dBZ per km
    "cap_et_gap_km",     # echo top minus the highest sampled level
    "cap_low_dbz",       # dBZ at the lowest level
)

#: A fixed scale per vertical quantity, so that the ten of them enter the
#: trunk on a comparable footing.  The scales are the largest value each
#: quantity can take over the archive, rounded up, so the scaled features
#: live roughly in [-1, 1].  They are constants rather than measured
#: statistics on purpose: a per-run statistic would make the input
#: distribution depend on the split, and the checkpoint would then mean
#: something different on the test cases.
VERTICAL_SCALES: tuple[float, ...] = (
    8.0,     # cap_top_km        0 .. 7 km
    60.0,    # cap_top_dbz       0 .. 60 dBZ
    60.0,    # cap_max_dbz       0 .. 60 dBZ
    8.0,     # cap_argmax_km     2 .. 7 km
    40.0,    # cap_bright_dbz    0 .. 40 dBZ
    6.0,     # cap_nz_levels     0 .. 6
    300.0,   # cap_col_dbz_sum   0 .. 360
    10.0,    # cap_slope_dbz_km  dBZ per km
    6.0,     # cap_et_gap_km     km
    60.0,    # cap_low_dbz       0 .. 60 dBZ
)


@dataclass
class VerticalQuantities:
    """The named vertical quantities of one frame.

    Attributes:
        values: (B, Q, H, W) tensor with the quantities stacked in the
            order of :data:`VERTICAL_QUANTITY_NAMES`.
        levels_km: the six level heights.
    """

    values: torch.Tensor
    levels_km: tuple[float, ...] = CAPPI_LEVELS_KM

    def __getitem__(self, name: str) -> torch.Tensor:
        """One quantity as a (B, 1, H, W) block.

        The channel axis is kept so that the blocks can be concatenated
        along it; a plain (B, H, W) would concatenate along the rows.
        """
        i = VERTICAL_QUANTITY_NAMES.index(name)
        return self.values[:, i:i + 1]

    @property
    def n_quantities(self) -> int:
        return len(VERTICAL_QUANTITY_NAMES)


def vertical_quantities(radar: torch.Tensor) -> VerticalQuantities:
    """Compute the vertical quantities from a radar frame.

    Args:
        radar: (B, 10, H, W) tensor in the archive's raw units, with the
            channels in the order of :data:`radar_nwp_bench.data.index.RADAR_VARS`.

    Returns:
        :class:`VerticalQuantities` with physical units.
    """
    if radar.dim() != 4:
        raise ValueError(f"expected (B, 10, H, W), got {tuple(radar.shape)}")
    if radar.shape[1] < len(RADAR_VARS):
        raise ValueError(f"expected {len(RADAR_VARS)} radar channels, got {radar.shape[1]}")

    levels = torch.tensor(CAPPI_LEVELS_KM, dtype=radar.dtype, device=radar.device)
    cap = radar[:, CAPPI_INDICES] / 10.0            # (B, 6, H, W) in dBZ
    et_km = radar[:, ET_INDEX] / 10.0               # (B, H, W) in km

    present = cap > 0.0
    masked = torch.where(present, cap, torch.zeros_like(cap))

    n_levels = present.sum(dim=1).to(radar.dtype)                       # (B, H, W)
    col_max = masked.max(dim=1).values                                  # (B, H, W)
    col_sum = masked.sum(dim=1)                                         # (B, H, W)
    low_dbz = cap[:, 0]                                                 # (B, H, W)

    # index of the highest level that carries signal; -1 where there is none
    idx = torch.arange(cap.shape[1], dtype=radar.dtype, device=radar.device)
    idx = idx.view(1, -1, 1, 1).expand_as(cap)
    top_idx = torch.where(present, idx, torch.full_like(idx, -1.0)).max(dim=1).values
    has_echo = top_idx >= 0
    # level index 0 is the 2 km level, so the height is index + 2
    top_km = torch.where(has_echo, top_idx + 2.0, torch.zeros_like(top_idx))

    # the level of the column maximum: an interior maximum is a melting band
    argmax_idx = cap.argmax(dim=1).to(radar.dtype)
    argmax_km = argmax_idx + 2.0                                        # index 0 -> 2 km

    bright = col_max - low_dbz

    # vertical gradient of dBZ, least squares over the six levels
    centred = levels - levels.mean()
    denom = float((centred ** 2).sum())
    slope = (masked * centred.view(1, -1, 1, 1)).sum(dim=1) / denom

    # echo top minus the highest sampled level; a large positive value means
    # the echo top is an overshooting top above the 7 km level
    et_gap = et_km - top_km

    top_dbz = torch.gather(cap, 1, top_idx.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    top_dbz = torch.where(has_echo, top_dbz, torch.zeros_like(top_dbz))

    stacked = torch.cat(
        [
            top_km.unsqueeze(1),
            top_dbz.unsqueeze(1),
            col_max.unsqueeze(1),
            argmax_km.unsqueeze(1),
            bright.unsqueeze(1),
            n_levels.unsqueeze(1),
            col_sum.unsqueeze(1),
            slope.unsqueeze(1),
            et_gap.unsqueeze(1),
            low_dbz.unsqueeze(1),
        ],
        dim=1,
    )
    return VerticalQuantities(values=stacked)


def scaled_vertical_quantities(radar: torch.Tensor) -> torch.Tensor:
    """The vertical quantities, divided by their fixed scales.

    Args:
        radar: (B, T, 10, H, W) or (B, 10, H, W) in the archive's raw units.

    Returns:
        (B, T, 10, H, W) or (B, 10, H, W) -- the scaled quantities, in the
        order of :data:`VERTICAL_QUANTITY_NAMES`.
    """
    squeeze = radar.dim() == 4
    x = radar[None] if squeeze else radar
    values = vertical_quantities(x.reshape(-1, *x.shape[2:])).values
    values = values.reshape(*x.shape[:2], *values.shape[1:])
    scales = torch.tensor(VERTICAL_SCALES, dtype=values.dtype, device=values.device)
    scales = scales.view(*([1] * (values.dim() - 3)), -1, 1, 1)
    out = values / scales
    return out[0] if squeeze else out


def phase_inputs(radar: torch.Tensor) -> torch.Tensor:
    """The subset of the vertical quantities that drive the phase routing.

    The phase weights must come from the *observed* column, not from the NWP
    environment: the measured regime analysis shows that routing the rain
    relation by DVG850 or by the freezing level reduces the residual by
    0.58 % on average, so an environment-driven router has nothing to gain
    (MASTER_CONTROL_ROOM.md section 2, F3).
    """
    vq = vertical_quantities(radar)
    names = ("cap_argmax_km", "cap_bright_dbz", "cap_nz_levels",
             "cap_slope_dbz_km", "cap_low_dbz", "cap_et_gap_km")
    return torch.cat([vq[n] for n in names], dim=1)

"""Grid geometry: the single source of truth for registration.

Facts this module encodes (all measured, see MASTER_CONTROL_ROOM.md section 2):

* RADAR / LABEL live on 301 x 301 @ 0.01 deg (3.00 deg span).
* NWP lives on 167 x 167 @ 0.03 deg (5.01 deg span) and *contains* the radar window.
* Registration is a scale-up of the NWP field by 3 followed by a centre crop.
  Because 3 is an integer, the scale-up is exact in the grid index: no
  interpolation assumption is needed to place the halo on the radar lattice.
* The centre crop throws away 100 px on each side, i.e. 33.33 native cells
  (1.00 degree) and 63.9 % of the area.  That discarded ring is the *halo*.
* The upstream loader uses **bilinear** upsampling with reflect padding; this
  module reproduces that convention so that anything built here is on the
  same footing as the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

RADAR_SHAPE: tuple[int, int] = (301, 301)
NWP_SHAPE: tuple[int, int] = (167, 167)
NWP_TO_RADAR_SCALE: int = 3

#: Spatial resolution of the radar grid, in degrees.
RADAR_RES_DEG: float = 0.01
#: Spatial resolution of the NWP grid, in degrees.
NWP_RES_DEG: float = 0.03

#: The six CAPPI levels and their heights in km.  CAP20 is the 2 km level.
CAPPI_LEVELS_KM: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

#: Radar variable order used by the packed archive.
RADAR_VARS: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)

#: The measured halo geometry, for the self-checks.  Source:
#: MASTER_CONTROL_ROOM.md section 2, F5 -- the centre crop throws away
#: 100 px on each side, which is 33.33 native cells or 1.00 degree, and
#: 63.9 % of the NWP area.
HALO_EXPECTED: dict[str, float] = {
    "crop_px_per_side": 100.0,
    "halo_native_cells_per_side": 100.0 / 3.0,
    "halo_degrees_per_side": 1.0,
    "halo_area_fraction": 1.0 - (301 * 301) / (501 * 501),
}


@dataclass(frozen=True)
class HaloGeometry:
    """The geometry of the discarded ring and of the upstream narrow annulus.

    Attributes:
        native: the NWP native grid size (167).
        scale: the upsample factor (3).
        radar: the radar grid size (301).
        crop: the crop offset in the upsampled grid (100).
        upsampled: the upsampled grid size (501).
    """

    native: int = NWP_SHAPE[0]
    scale: int = NWP_TO_RADAR_SCALE
    radar: int = RADAR_SHAPE[0]

    @property
    def upsampled(self) -> int:
        return self.native * self.scale

    @property
    def crop(self) -> int:
        return (self.upsampled - self.radar) // 2

    @property
    def halo_native_cells_per_side(self) -> float:
        return self.crop / self.scale

    @property
    def halo_degrees_per_side(self) -> float:
        return self.halo_native_cells_per_side * NWP_RES_DEG

    @property
    def halo_area_fraction(self) -> float:
        total = self.upsampled * self.upsampled
        centre = self.radar * self.radar
        return 1.0 - centre / total

    def describe(self) -> dict:
        return {
            "native": self.native,
            "scale": self.scale,
            "radar": self.radar,
            "upsampled": self.upsampled,
            "crop_px_per_side": self.crop,
            "halo_native_cells_per_side": self.halo_native_cells_per_side,
            "halo_degrees_per_side": self.halo_degrees_per_side,
            "halo_area_fraction": self.halo_area_fraction,
        }


def bilinear_upsample(field, scale: int = NWP_TO_RADAR_SCALE):
    """Bilinear upsample a (..., 167, 167) field to (..., 501, 501).

    Mirrors ``radar_nwp_bench.data.coordinates._bilinear_upsample``: reflect
    padding at the border, pixel centres aligned through ``(dst + 0.5) / s``.
    """
    import torch

    if field.shape[-2:] != NWP_SHAPE:
        raise ValueError(f"expected NWP field of shape {NWP_SHAPE}, got {tuple(field.shape[-2:])}")
    squeeze = field.dim() == 2
    x = field[None] if squeeze else field
    h, w = NWP_SHAPE
    dev = x.device

    def coords(n: int):
        dst = torch.arange(n * scale, device=dev, dtype=torch.float64)
        src = (dst + 0.5) / scale - 0.5
        i0 = torch.floor(src)
        weight = (src - i0).to(torch.float32)
        period = 2 * (n - 1)
        i0r = torch.remainder(i0, period)
        i0r = torch.where(i0r >= n, period - i0r, i0r)
        i1r = torch.remainder(i0 + 1, period)
        i1r = torch.where(i1r >= n, period - i1r, i1r)
        return i0r.long(), i1r.long(), weight

    r0, r1, rw = coords(h)
    c0, c1, cw = coords(w)
    xf = x.float()
    # Column interpolation first, then row interpolation -- exactly the
    # arithmetic of ``coordinates._bilinear_upsample`` (the numpy path the
    # data pipeline actually runs).  The upstream torch helper
    # ``resample_nwp_to_radar_torch`` weights the rows the other way round;
    # this module follows the numpy path on purpose.
    row0 = xf[..., r0, :]
    row1 = xf[..., r1, :]
    top = row0[..., c0] * (1 - cw) + row0[..., c1] * cw
    bot = row1[..., c0] * (1 - cw) + row1[..., c1] * cw
    rw_col = rw.view(*([1] * (top.dim() - 2)), -1, 1)
    out = top * (1 - rw_col) + bot * rw_col
    return out[0] if squeeze else out


def center_crop(field, radar: int = RADAR_SHAPE[0]):
    """Crop the upsampled field to the radar window (centre 301 x 301)."""
    import torch

    up = field.shape[-1]
    if up < radar:
        raise ValueError(f"cannot crop {radar} out of {up}")
    r0 = (field.shape[-2] - radar) // 2
    c0 = (field.shape[-1] - radar) // 2
    return field[..., r0:r0 + radar, c0:c0 + radar]


def nwp_to_radar(field):
    """The benchmark's registration: upsample by 3, then centre crop."""
    return center_crop(bilinear_upsample(field))


def symmetric_patch_origins(grid_dim: int, patch_size: int, nominal_stride: int) -> tuple[int, ...]:
    """Symmetric, evenly spaced sliding-window origins.

    The same rule as ``radar_nwp_bench.data.coordinates.symmetric_patch_origins``:
    the origins span the full field evenly, so the last window does not
    overlap the previous ones more than the first one does.
    """
    import math

    if patch_size >= grid_dim:
        return (0,)
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    return tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))

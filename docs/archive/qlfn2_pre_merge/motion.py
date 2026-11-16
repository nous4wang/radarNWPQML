"""The motion field, measured from the radar itself.

Why not the NWP wind: the archive's 27 NWP elements contain wind *speeds*
(``WS500``, ``WS700``, ``WS925``) but no wind *components* and no vorticity,
so a divergence field can at best recover the divergent part of the wind
(MASTER_CONTROL_ROOM.md section 2, F4).  The old ``qlfn`` contract used
``(DVG850, WS925)`` as a stand-in for u/v, which cannot give a direction.

The radar does give a direction: the echo pattern tracks the convective
system.  Measured over the 62 cases, the echo moves 1.6 to 5.0 pixels per
6-minute step, i.e. 18 to 56 km/h.  That is also the number that says how
much of the NWP halo can physically reach the radar domain inside the
60-minute forecast window (16 to 50 pixels, against a halo that is 100
pixels wide).

Two estimators, and why the default is the second one
------------------------------------------------------
The benchmark slides a 128x128 patch over the 301x301 grid and stitches the
tiles back together with a Hann window.  Any statistic the model computes
therefore has to give **the same answer on every tile of the same window**,
or the stitched field carries the discontinuity at the seams.

* ``centroid`` -- the mass-weighted echo centroid against time.  It is
  simple, but it depends on *where the storm sits inside the tile*: the
  same storm in two different tiles gives two different centroids, and two
  different velocities.  It is kept as the reference estimator, and it is
  the one the measured 1.6 to 5.0 px/step figure comes from.
* ``corr`` -- phase correlation between consecutive frames.  A rigid
  translation shifts the *phase* of every Fourier component by the same
  amount, so the correlation peak reports the displacement **independently
  of where the pattern sits in the tile**.  This is the default.

The displacement is chained over the last five steps and averaged, because
one 6-minute step moves the pattern by only 1.6 to 5.0 pixels and the
integer quantisation of a single step would be a large part of that.

The estimator is a property of the input window, not a learned parameter,
so it cannot drift between train and test.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .cappi import CR_INDEX


@dataclass
class Motion:
    """A constant translation field, for one window.

    Attributes:
        vx: eastward (column) component, in pixels per 6-minute step.
        vy: southward (row) component, in pixels per 6-minute step.
        usable: whether the echo track supported a fit.
    """

    vx: float = 0.0
    vy: float = 0.0
    usable: bool = False

    @property
    def speed(self) -> float:
        return float((self.vx ** 2 + self.vy ** 2) ** 0.5)

    @property
    def upstream(self) -> tuple[float, float]:
        """The direction the weather comes *from*, in (column, row) units."""
        return (-self.vx, -self.vy)

    def as_tuple(self) -> tuple[float, float]:
        return (self.vx, self.vy)


@dataclass
class MotionField:
    """The same quantity, for a whole batch.

    Attributes:
        vx: (B,) eastward component, pixels per 6-minute step.
        vy: (B,) southward component, pixels per 6-minute step.
        usable: (B,) whether the echo track supported a fit.
        method: which estimator produced it.
    """

    vx: torch.Tensor
    vy: torch.Tensor
    usable: torch.Tensor
    method: str = "corr"

    def scalar(self, index: int = 0) -> Motion:
        """The motion of one batch element, as a scalar record."""
        return Motion(
            vx=float(self.vx[index]),
            vy=float(self.vy[index]),
            usable=bool(self.usable[index]),
        )


#: The echo threshold, in dBZ, that counts as "there is precipitation".
ECHO_THRESHOLD_DBZ: float = 20.0
#: How many past steps the displacement is chained over.
CHAIN_STEPS: int = 5


def _echo_field(radar: torch.Tensor, thresh_dbz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The composite reflectivity in dBZ and the echo mask.

    Args:
        radar: (B, T, 10, H, W) in the archive's raw units.
        thresh_dbz: the reflectivity that counts as echo.

    Returns:
        A pair ``(cr_dbz, mask)`` of shape (B, T, H, W).
    """
    cr = radar[:, :, CR_INDEX] / 10.0                      # (B, T, H, W) dBZ
    mask = cr > thresh_dbz
    return cr, mask


def _search_shift(
    a: torch.Tensor,
    b: torch.Tensor,
    dy_cand: torch.Tensor,
    dx_cand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The integer shift that best aligns ``b`` onto ``a``.

    The score is the overlap-normalised product of the two fields, which is
    the plain matched filter: for a rigid translation of the echo pattern it
    peaks exactly at the displacement, and -- unlike the centroid -- it does
    not depend on where the pattern sits inside the window, so every tile
    of a sliding window reports the same velocity.

    Args:
        a: (B, H, W) the later field.
        b: (B, H, W) the earlier field.
        dy_cand: (B, S) candidate row offsets.
        dx_cand: (B, S) candidate column offsets.

    Returns:
        A pair ``(dy, dx)`` of integer tensors of shape (B,).
    """
    bsz, h, w = a.shape
    s = dy_cand.shape[1]
    rows = torch.arange(h, device=a.device).view(1, 1, h, 1)
    cols = torch.arange(w, device=a.device).view(1, 1, 1, w)
    ys = rows + dy_cand.view(bsz, s, 1, 1)          # (B, S, H, 1)
    xs = cols + dx_cand.view(bsz, s, 1, 1)          # (B, S, 1, W)
    valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    valid = valid.expand(bsz, s, h, w)
    idx_y = ys.expand(bsz, s, h, w).clamp(0, h - 1).long()
    idx_x = xs.expand(bsz, s, h, w).clamp(0, w - 1).long()

    a_sel = a.view(bsz, 1, h, w).expand(bsz, s, h, w).gather(2, idx_y)
    a_sel = a_sel.gather(3, idx_x)
    b_sel = b.view(bsz, 1, h, w).expand(bsz, s, h, w)

    # The score is the normalised cross-correlation over the overlap: the
    # mean is removed and the two variances are divided out, so a tile that
    # cuts the echo pattern in half still reports the displacement of what
    # is left, instead of preferring the shift that keeps the most pixels.
    mask = valid.to(a.dtype)
    n = mask.sum(dim=(2, 3)).clamp(min=1.0)                    # (B, S)
    a_mean = (a_sel * mask).sum(dim=(2, 3)) / n
    b_mean = (b_sel * mask).sum(dim=(2, 3)) / n
    da = (a_sel - a_mean.view(bsz, s, 1, 1)) * mask
    db = (b_sel - b_mean.view(bsz, s, 1, 1)) * mask
    cov = (da * db).sum(dim=(2, 3))
    va = (da * da).sum(dim=(2, 3))
    vb = (db * db).sum(dim=(2, 3))
    score = cov / torch.sqrt(va * vb + 1e-6)
    best = score.argmax(dim=1)
    return dy_cand.gather(1, best.view(-1, 1)).squeeze(1), \
        dx_cand.gather(1, best.view(-1, 1)).squeeze(1)


def _estimate_pair_shift(
    a: torch.Tensor, b: torch.Tensor, max_shift: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The displacement between two frames, by a two-stage matched filter.

    The coarse stage scans the whole range with a stride of two, the fine
    stage looks at the three-by-three neighbourhood of the coarse peak.
    Splitting it in two keeps the cost down without giving up the integer
    resolution.

    Args:
        a: (B, H, W) the later frame.
        b: (B, H, W) the earlier frame.
        max_shift: the largest displacement to look for, in pixels.

    Returns:
        A pair ``(dy, dx)`` of integer tensors of shape (B,).
    """
    bsz = a.shape[0]
    span = torch.arange(-max_shift, max_shift + 1, 2, device=a.device)
    dys, dxs = torch.meshgrid(span, span, indexing="ij")
    coarse_dy = dys.flatten().view(1, -1).expand(bsz, -1)
    coarse_dx = dxs.flatten().view(1, -1).expand(bsz, -1)
    dy0, dx0 = _search_shift(a, b, coarse_dy, coarse_dx)

    fine = torch.arange(-1, 2, device=a.device)
    fy, fx = torch.meshgrid(fine, fine, indexing="ij")
    fine_dy = (dy0.view(-1, 1) + fy.flatten().view(1, -1))
    fine_dx = (dx0.view(-1, 1) + fx.flatten().view(1, -1))
    return _search_shift(a, b, fine_dy, fine_dx)


def _centroid_motion(
    cr: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The centroid velocity of every batch element.

    Args:
        cr: (B, T, H, W) the composite reflectivity in dBZ.
        mask: (B, T, H, W) the echo mask.

    Returns:
        A triple ``(vx, vy, usable)`` of shape (B,) each.
    """
    b, t = cr.shape[0], cr.shape[1]
    m = mask.to(cr.dtype)
    total = m.flatten(2).sum(dim=2)                        # (B, T)
    h, w = cr.shape[-2:]
    rows = torch.arange(h, dtype=cr.dtype, device=cr.device).view(1, 1, h, 1)
    cols = torch.arange(w, dtype=cr.dtype, device=cr.device).view(1, 1, 1, w)
    cy = (m * rows).flatten(2).sum(dim=2) / total.clamp(min=1.0)
    cx = (m * cols).flatten(2).sum(dim=2) / total.clamp(min=1.0)

    idx = torch.arange(t, dtype=cr.dtype, device=cr.device).view(1, t).expand(b, t)
    valid = total > 10.0
    n = valid.sum(dim=1).clamp(min=1).to(cr.dtype)
    sx = (idx * valid).sum(dim=1)
    sxx = (idx * idx * valid).sum(dim=1)
    denom = n * sxx - sx * sx

    sy = (cy * valid).sum(dim=1)
    sxy = (idx * cy * valid).sum(dim=1)
    slope_y = torch.where(denom.abs() > 1e-6, (n * sxy - sx * sy) / denom,
                          torch.zeros_like(denom))

    sy_x = (cx * valid).sum(dim=1)
    sxy_x = (idx * cx * valid).sum(dim=1)
    slope_x = torch.where(denom.abs() > 1e-6, (n * sxy_x - sx * sy_x) / denom,
                          torch.zeros_like(denom))

    usable = valid.sum(dim=1) >= 4
    return slope_x, slope_y, usable


def estimate_motion_field(
    radar: torch.Tensor,
    thresh_dbz: float = ECHO_THRESHOLD_DBZ,
    mode: str = "corr",
    max_shift: int = 8,
    chain_steps: int = CHAIN_STEPS,
) -> MotionField:
    """Estimate the motion of every element of a batch.

    Args:
        radar: (B, T, 10, H, W) in the archive's raw units.
        thresh_dbz: the reflectivity that counts as echo.
        mode: ``"corr"``, ``"centroid"`` or ``"none"``.
        max_shift: the largest per-step displacement the search covers.
        chain_steps: how many past steps the correlation chain covers.

    Returns:
        :class:`MotionField`.
    """
    squeeze = radar.dim() == 4
    x = radar[None] if squeeze else radar
    if x.dim() != 5:
        raise ValueError(f"expected (B, T, 10, H, W), got {tuple(radar.shape)}")
    b = x.shape[0]
    cr, mask = _echo_field(x, thresh_dbz)

    if mode == "none":
        zero = torch.zeros(b, dtype=x.dtype, device=x.device)
        return MotionField(
            zero, zero.clone(),
            torch.zeros(b, dtype=torch.bool, device=x.device),
            method="none",
        )

    if mode == "centroid":
        vx, vy, usable = _centroid_motion(cr, mask)
        return MotionField(vx, vy, usable, method="centroid")

    if mode != "corr":
        raise ValueError(f"unknown motion mode {mode!r}")

    t = x.shape[1]
    steps = max(1, min(int(chain_steps), t - 1))
    vx = torch.zeros(b, dtype=x.dtype, device=x.device)
    vy = torch.zeros(b, dtype=x.dtype, device=x.device)
    counted = torch.zeros(b, dtype=x.dtype, device=x.device)
    echo_count = mask.to(x.dtype).flatten(2).sum(dim=2)     # (B, T)

    for k in range(steps):
        # the pair (frame t-k-1, frame t-k) -- one 6-minute step apart
        earlier = x[:, t - 1 - k - 1]
        later = x[:, t - 1 - k]
        a = later[:, CR_INDEX] / 10.0
        bfr = earlier[:, CR_INDEX] / 10.0
        dy, dx = _estimate_pair_shift(a, bfr, max_shift)
        pair_ok = ((echo_count[:, t - 1 - k - 1] > 10.0)
                   & (echo_count[:, t - 1 - k] > 10.0))
        vx = vx + dx.to(x.dtype) * pair_ok.to(x.dtype)
        vy = vy + dy.to(x.dtype) * pair_ok.to(x.dtype)
        counted = counted + pair_ok.to(x.dtype)

    n = counted.clamp(min=1.0)
    vx = vx / n
    vy = vy / n
    usable = counted >= 2
    # where fewer than two pairs carried echo, the velocity is not a
    # measurement; report zero rather than a divided-by-artefact.
    vx = torch.where(usable, vx, torch.zeros_like(vx))
    vy = torch.where(usable, vy, torch.zeros_like(vy))
    return MotionField(vx, vy, usable, method="corr")


def estimate_motion(
    radar: torch.Tensor,
    thresh_dbz: float = ECHO_THRESHOLD_DBZ,
    mode: str = "corr",
    max_shift: int = 8,
) -> Motion:
    """Estimate the motion of the first element of a batch.

    Kept as the scalar entry point for the halo geometry, which needs one
    direction, and for the self-checks.

    Args:
        radar: (B, T, 10, H, W) or (T, 10, H, W) in the archive's raw units.
        thresh_dbz: the reflectivity that counts as echo.
        mode: ``"corr"``, ``"centroid"`` or ``"none"``.
        max_shift: the largest per-step displacement the search covers.

    Returns:
        :class:`Motion`.
    """
    field = estimate_motion_field(radar, thresh_dbz=thresh_dbz, mode=mode,
                                   max_shift=max_shift)
    return field.scalar(0)


def warp(field: torch.Tensor, vx, vy, steps: float) -> torch.Tensor:
    """Semi-Lagrangian advection of a field by a constant velocity.

    The output at lead ``steps`` is the input sampled at ``p - v * steps``,
    i.e. the field is moved *along* the velocity.  Bilinear sampling keeps
    the operation smooth and differentiable.

    Args:
        field: (B, C, H, W).
        vx: eastward velocity in pixels per 6-minute step.  A float applies
            to the whole batch; a (B,) tensor gives every element its own.
        vy: southward velocity in pixels per 6-minute step.
        steps: the lead time in 6-minute steps.

    Returns:
        (B, C, H, W), the advected field.
    """
    if field.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(field.shape)}")
    b, c, h, w = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=field.dtype, device=field.device),
        torch.arange(w, dtype=field.dtype, device=field.device),
        indexing="ij",
    )
    if torch.is_tensor(vx):
        vx_b = vx.to(field.dtype).reshape(b, 1, 1)
        vy_b = vy.to(field.dtype).reshape(b, 1, 1)
        src_x = xx.unsqueeze(0) - vx_b * steps
        src_y = yy.unsqueeze(0) - vy_b * steps
    else:
        src_x = (xx - float(vx) * steps).unsqueeze(0)
        src_y = (yy - float(vy) * steps).unsqueeze(0)
    gx = (src_x / (w - 1)) * 2.0 - 1.0
    gy = (src_y / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    if grid.shape[0] == 1 and b > 1:
        grid = grid.expand(b, *grid.shape[1:])
    out = F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)
    return out


def advect_sequence(field: torch.Tensor, motion, n_steps: int) -> torch.Tensor:
    """Advect a t0 field onto the ten forecast leads.

    Args:
        field: (B, C, H, W) the rain field at the issue time.
        motion: a :class:`Motion` or a :class:`MotionField`.
        n_steps: how many leads to produce.

    Returns:
        (B, n_steps, C, H, W).
    """
    vx, vy = motion.vx, motion.vy
    outs = [warp(field, vx, vy, float(k + 1)) for k in range(n_steps)]
    return torch.stack(outs, dim=1)

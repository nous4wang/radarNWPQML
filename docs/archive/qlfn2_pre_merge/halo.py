"""The NWP halo, reduced to what the forecast window can actually reach.

The idea the user brought in: enlarge the NWP domain so the model can see the
convective systems outside the radar window, project the larger grid into the
latent space, and project back onto the radar grid one to one.

What the measurements say (MASTER_CONTROL_ROOM.md section 2, F5 and F6):

* The halo is not empty.  Its signal amplitude is 0.94 to 1.28 times the
  centre's, and it carries 63.9 % of the NWP area.
* At the domain level it carries **no** usable information about the next
  hour's rainfall: leave-one-case-out R^2 = -0.012 on its own, and adding
  it to a centre-only predictor *lowers* R^2 by 0.099.  Upstream versus
  downstream half of the ring is a coin flip (34 wins against 28 losses).
* The reason is geometric.  The echo moves 16 to 50 pixels inside the
  60-minute window, and the halo is 100 pixels wide on each side, so most of
  the ring cannot arrive in time to matter.

So the halo is not thrown away, but it is **narrowed**: the only part of it
that can matter is the upstream half-plane within the advection distance.
That is what this module builds, and it also builds the *downstream* mirror
of the same annulus, which is the control that turns "it helps" into "it
helps *because* it is upstream".

The registration needs no interpolation assumption: 0.03 deg is exactly three
times 0.01 deg, so the native grid maps onto the radar lattice by an integer
factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import HaloGeometry, bilinear_upsample
from .motion import Motion


@dataclass
class HaloMasks:
    """Boolean masks on the upsampled 501 x 501 grid.

    Attributes:
        centre: the 301 x 301 radar window.
        upstream: the upstream half-plane of the ring, within reach.
        downstream: the mirror of ``upstream``, for the control.
        geometry: the geometry the masks were built from.
    """

    centre: torch.Tensor
    upstream: torch.Tensor
    downstream: torch.Tensor
    geometry: HaloGeometry


def _half_plane(geom: HaloGeometry, vx: float, vy: float) -> torch.Tensor:
    """The mask of the upstream half-plane of the ring."""
    n = geom.upsampled
    c = (n - 1) / 2.0
    idx = torch.arange(n, dtype=torch.float32)
    dy = (idx - c).view(-1, 1)
    dx = (idx - c).view(1, -1)
    proj = dx * (-vx) + dy * (-vy)
    return proj > 0


def build_masks(motion: Motion, max_distance_px: float,
                geometry: HaloGeometry | None = None) -> HaloMasks:
    """Build the centre, upstream and downstream masks.

    Args:
        motion: the motion estimate; its upstream direction selects the half.
        max_distance_px: how far outside the radar window the annulus
            reaches.  This should be the distance the echo travels inside the
            forecast window -- 16 to 50 pixels for this archive -- not the
            full 100-pixel ring.
        geometry: the grid geometry; the default is the benchmark's.

    Returns:
        :class:`HaloMasks`.
    """
    geom = geometry or HaloGeometry()
    n = geom.upsampled
    centre = torch.zeros((n, n), dtype=torch.bool)
    c0 = geom.crop
    centre[c0:c0 + geom.radar, c0:c0 + geom.radar] = True
    ring = ~centre

    up_half = _half_plane(geom, motion.vx, motion.vy)
    # the distance outside the radar window, per pixel
    idx = torch.arange(n, dtype=torch.float32)
    dy = (idx - (n - 1) / 2.0).abs()
    dx = (idx - (n - 1) / 2.0).abs()
    outside = torch.maximum(dy.view(-1, 1), dx.view(1, -1)) - geom.radar / 2.0
    near = outside <= float(max_distance_px)

    upstream = ring & up_half & near
    downstream = ring & (~up_half) & near
    return HaloMasks(centre=centre, upstream=upstream, downstream=downstream,
                     geometry=geom)


def reachable_distance_px(motion: Motion, lead_minutes: float = 60.0,
                          step_minutes: float = 6.0) -> float:
    """How far the echo travels inside the forecast window, in pixels."""
    return motion.speed * (lead_minutes / step_minutes)


def summarise(field_upsampled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Summary statistics of one field over one mask.

    Args:
        field_upsampled: (B, H, W) on the 501 x 501 grid.
        mask: (H, W) boolean.

    Returns:
        (B, 4): mean, 10th percentile, minimum and the fraction of pixels
        below the zero level.
    """
    if field_upsampled.dim() != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(field_upsampled.shape)}")
    sel = field_upsampled[:, mask]                       # (B, M)
    if sel.shape[1] == 0:
        return field_upsampled.new_zeros((field_upsampled.shape[0], 4))
    q10 = torch.quantile(sel, 0.10, dim=1)
    return torch.stack(
        [sel.mean(dim=1), q10, sel.min(dim=1).values, (sel < 0).to(sel.dtype).mean(dim=1)],
        dim=1,
    )


def build_halo_context(
    nwp_native: torch.Tensor,
    motion: Motion,
    max_distance_px: float | None = None,
    variables: tuple[str, ...] = ("DVG850", "DVG925", "CAPE", "PWAT", "TTdMean74"),
) -> dict[str, torch.Tensor]:
    """Build the halo context for one NWP slice.

    Args:
        nwp_native: (B, 27, 167, 167) the raw NWP slice in archive units.
        motion: the motion estimate, used to aim the annulus.
        max_distance_px: the reach of the annulus.  The default is the
            distance the echo travels inside the forecast window.
        variables: which NWP variables to summarise.

    Returns:
        A dict with, per variable, the upstream and downstream summaries
        (4 numbers each) plus the centre summary, and the masks themselves
        under the keys ``mask_upstream`` and ``mask_downstream``.
    """
    if nwp_native.dim() != 4:
        raise ValueError(f"expected (B, 27, 167, 167), got {tuple(nwp_native.shape)}")
    geom = HaloGeometry()
    reach = reachable_distance_px(motion) if max_distance_px is None else max_distance_px
    masks = build_masks(motion, reach, geom)

    up = bilinear_upsample(nwp_native)                    # (B, 27, 501, 501)
    context: dict[str, torch.Tensor] = {}
    for i, name in enumerate(variables):
        if i >= up.shape[1]:
            break
        f = up[:, i]
        context[f"{name}_upstream"] = summarise(f, masks.upstream)
        context[f"{name}_downstream"] = summarise(f, masks.downstream)
        context[f"{name}_centre"] = summarise(f, masks.centre)
    context["mask_upstream"] = masks.upstream
    context["mask_downstream"] = masks.downstream
    context["reach_px"] = torch.tensor(float(reach))
    return context

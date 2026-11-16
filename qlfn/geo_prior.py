"""Static geo / radar reference tensor for QLFN (torch layer).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): this is a data-side prior only, not one of the rejected mechanism
families. It builds no claim and is not wired into the authoritative pipeline.

The blueprint hard-coded 301x301. Upstream models receive training patches
(``dataset.py:74``), so this module derives every coordinate from the actual
input shape and caches nothing device-specific.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Spatial size of each learnable clutter parameter. Interpolated to the
#: actual H/W, so this is a parameter shape, not a patch-size assumption.
CLUTTER_GRID = 16


class StaticGeoRadarPrior(nn.Module):
    """Build static coordinate / beam-geometry channels for any H and W.

    Five analytic channels are (normalised lon, normalised lat, normalised
    radial distance, sin(azimuth), cos(azimuth)). Remaining channels are
    low-resolution learnable clutter maps; they are interpolated to H x W
    because a fixed-size parameter cannot cover arbitrary patch sizes.

    Args:
        geo_channels: Total channels, 5..8.

    Raises:
        ValueError: If ``geo_channels`` is outside 5..8.
    """

    def __init__(self, geo_channels: int = 6) -> None:
        super().__init__()
        if not 5 <= geo_channels <= 8:
            raise ValueError("geo_channels must be between 5 and 8")
        self.geo_channels = geo_channels
        self.n_learned = geo_channels - 5
        if self.n_learned:
            self.clutter = nn.Parameter(
                torch.zeros(1, self.n_learned, CLUTTER_GRID, CLUTTER_GRID)
            )
        else:
            self.register_parameter("clutter", None)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """Return ``(B, geo_channels, H, W)`` on the reference's device.

        Args:
            reference: Any tensor with shape ``(B, C, H, W)``.

        Returns:
            Static prior broadcast to the batch size.

        Raises:
            ValueError: If ``reference`` is not four-dimensional.
        """
        if reference.dim() != 4:
            raise ValueError("reference must be (B, C, H, W)")
        batch, _, height, width = reference.shape
        device, dtype = reference.device, reference.dtype
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x ** 2 + grid_y ** 2) / math.sqrt(2.0)
        azimuth = torch.atan2(grid_y, grid_x)
        fixed = torch.stack(
            [grid_x, grid_y, radius, torch.sin(azimuth), torch.cos(azimuth)],
            dim=0,
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        if not self.n_learned:
            return fixed
        learned = F.interpolate(
            self.clutter.to(device=device, dtype=dtype),
            size=(height, width), mode="bilinear", align_corners=False,
        ).expand(batch, -1, -1, -1)
        return torch.cat([fixed, learned], dim=1)

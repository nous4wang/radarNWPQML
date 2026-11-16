"""The unit bridge: the benchmark's network scale <-> the archive's units.

Three conventions meet in this project, and confusing them is the easiest
way to produce a number that looks like a score and is not one.

1. **The archive's raw units.**  The packed ``RADAR`` series stores int16
   values: the six CAPPI levels, ``CR`` and ``HBR`` in 0.1 dBZ, ``ET`` in
   0.1 km, ``VIL`` in 0.1 kg m-2.  Every measured number in
   ``MASTER_CONTROL_ROOM.md`` section 2 was computed on this scale, and the
   qlfn2 library works on it too.

2. **Physical units.**  ``radar_nwp_bench.data.transforms.clean_radar_field``
   divides the raw values by ten and maps the sentinels to zero, giving dBZ,
   km and kg m-2.

3. **The network scale.**  ``radar_nwp_bench.data.transforms.radar_log_transform``
   compresses the physical values before the network sees them::

       dBZ variables (CAP20..CAP70, CR, HBR):  x = log1p(clip(v, 0)) / 6
       ET and VIL:                            x = log1p(clip(v, 0)) / 4

   The upstream harness hands the model *this*, never the physical value.

The transform is strictly monotone and exactly invertible, so the bridge is
lossless: ``net_to_archive(archive_to_net(x)) == x`` up to float32 rounding.
That is what lets the qlfn2 physics run on the archive's scale while the
network's input keeps the benchmark's conditioning -- and it is checked in
``qlfn2.checks``.

The channel order is the benchmark's:
``radar_nwp_bench.data.index.RADAR_VARS`` -- CAP20, CAP30, CAP40, CAP50,
CAP60, CAP70, CR, ET, HBR, VIL.
"""
from __future__ import annotations

import torch

from .geometry import RADAR_VARS

#: The archive stores the radar fields as int16 in tenths.
ARCHIVE_SCALE: float = 10.0

#: The divisor the benchmark uses for the dBZ variables.
NET_DIVISOR_DBZ: float = 6.0
#: The divisor it uses for ET and VIL.
NET_DIVISOR_OTHER: float = 4.0

#: The variables that are dBZ.
DBZ_VARS: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70", "CR", "HBR",
)
#: The variables that are not dBZ.
OTHER_VARS: tuple[str, ...] = ("ET", "VIL")


def _divisor_vector(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """One divisor per channel, shaped for broadcasting over the channel axis.

    Args:
        x: the tensor whose channel axis is being described.
        var_names: the variable name of each channel.

    Returns:
        A tensor of shape ``(1, ..., C, 1, 1)`` that broadcasts against
        ``x`` when ``x`` has the channel axis third from the end.

    Raises:
        ValueError: when the channel count does not match the names.
    """
    if x.shape[-3] != len(var_names):
        raise ValueError(
            f"expected {len(var_names)} channels, got {x.shape[-3]}"
        )
    values = [
        NET_DIVISOR_DBZ if name in DBZ_VARS else NET_DIVISOR_OTHER
        for name in var_names
    ]
    div = torch.tensor(values, dtype=x.dtype, device=x.device)
    return div.view(*([1] * (x.dim() - 3)), -1, 1, 1)


def net_to_physical(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """Network scale -> physical units (dBZ, km, kg m-2).

    Args:
        x: (..., C, H, W) the compressed field the harness produced.
        var_names: the variable name of each channel.

    Returns:
        The physical field, same shape as ``x``.
    """
    div = _divisor_vector(x, var_names)
    return torch.expm1(x.clamp(min=0.0) * div)


def physical_to_net(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """Physical units -> network scale.

    Args:
        x: (..., C, H, W) the physical field.
        var_names: the variable name of each channel.

    Returns:
        The compressed field, same shape as ``x``.
    """
    div = _divisor_vector(x, var_names)
    return torch.log1p(x.clamp(min=0.0)) / div


def net_to_archive(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """Network scale -> the archive's raw units (0.1 dBZ, 0.1 km, 0.1 kg m-2).

    Args:
        x: (..., C, H, W) the compressed field the harness produced.
        var_names: the variable name of each channel.

    Returns:
        The field on the archive's scale, same shape as ``x``.
    """
    return net_to_physical(x, var_names) * ARCHIVE_SCALE


def archive_to_net(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """The archive's raw units -> network scale.

    Args:
        x: (..., C, H, W) the field on the archive's scale.
        var_names: the variable name of each channel.

    Returns:
        The compressed field, same shape as ``x``.
    """
    return physical_to_net(x / ARCHIVE_SCALE, var_names)


def archive_to_physical(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """The archive's raw units -> physical units.

    Args:
        x: (..., C, H, W) the field on the archive's scale.
        var_names: the variable name of each channel.

    Returns:
        The physical field, same shape as ``x``.
    """
    return x / ARCHIVE_SCALE


def physical_to_archive(
    x: torch.Tensor, var_names: tuple[str, ...] = RADAR_VARS
) -> torch.Tensor:
    """Physical units -> the archive's raw units.

    Args:
        x: (..., C, H, W) the physical field.
        var_names: the variable name of each channel.

    Returns:
        The field on the archive's scale, same shape as ``x``.
    """
    return x * ARCHIVE_SCALE


__all__ = [
    "ARCHIVE_SCALE",
    "DBZ_VARS",
    "NET_DIVISOR_DBZ",
    "NET_DIVISOR_OTHER",
    "OTHER_VARS",
    "archive_to_net",
    "archive_to_physical",
    "net_to_archive",
    "net_to_physical",
    "physical_to_archive",
    "physical_to_net",
]

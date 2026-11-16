"""Data contract and shape algebra for the QLFN reference implementation.

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): this module fixes the *data* contract only. It does not implement or
endorse the rejected mechanism families, and it makes no research claim.

Standard library only. Every load-bearing constant carries its source:

* ``MASTER.md:51-54,74-76`` - IO seam and grid description.
* ``radar_nwp_bench/models/base.py:10-19`` - forward signature and shapes.
* ``radar_nwp_bench/data/index.py:14-24`` - radar / NWP channel order.
* ``radar_nwp_bench/data/dataset.py:74,84`` and ``scripts/train.py:59,117``
  - training patch defaults (128 / None), i.e. 301x301 is NOT the model input.
* ``MASTER.md:224`` - NWP edge is discarded, never extrapolated.
* ``MASTER.md:1186`` - no U/V components, no wind direction, no 850 hPa wind.
* ``MASTER.md:1556`` - the project's radar grid spacing is 1.11195 km/px;
  an external proposal's ``dx_km=1.0`` is explicitly listed as an interface
  error. Both the blueprint and QLFN v1.0 wrongly assumed 1 km/px.
* ``artifacts/decidability.md`` and ``MASTER.md §11.9`` - floor / oracle /
  headroom / seed range.
* Blueprint ``20261108`` §5.4 - 12 km cap on the learned drift displacement
  (an engineering bound, not a physical claim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

T_IN = 11
T_NWP = 2
T_OUT = 10
C_RADAR = 10
C_NWP = 27

RADAR_VARIABLES: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)
NWP_VARIABLES: tuple[str, ...] = (
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000",
    "Q700", "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74",
    "TTdMean74", "WS500", "WS700", "WS925",
)

CR_INDEX = RADAR_VARIABLES.index("CR")       # 6
HBR_INDEX = RADAR_VARIABLES.index("HBR")     # 8
VIL_INDEX = RADAR_VARIABLES.index("VIL")     # 9

THRESHOLDS_MM_PER_6MIN: tuple[float, ...] = (
    0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
)

# CSI@2.0 reference values. Units: mm per 6-min accumulation, NOT mm/h.
FLOOR_CSI2 = 0.0924
ORACLE_CSI2 = 0.1653
HEADROOM = 0.0729
SEED_RANGE = 0.1671

#: Explicitly rejected mechanism families, with upstream line anchors.
SEALED_DIRECTIONS: tuple[str, ...] = (
    "lagrangian_advection:MASTER.md:1048,1085,1088",
    "quantum_phase_tensor_network:MASTER.md:1083,805,382",
    "generative_flow_matching:MASTER.md:1085,1053,763",
    "rule:MASTER.md:1456",
)

#: No U/V wind vector exists in the 27 NWP channels. These scalar indices are
#: declared *surrogates*, not wind components; see ``MASTER.md:1186``.
WIND_SURROGATE_INDICES: tuple[int, int] = (
    NWP_VARIABLES.index("DVG850"),
    NWP_VARIABLES.index("WS925"),
)

FOLD_FACTOR = 3

#: Radar grid spacing in km/px. Source: ``MASTER.md:1556``. The commonly used
#: 1 km/px approximation is explicitly rejected upstream.
KM_PER_PIXEL = 1.11195

#: Maximum learned drift displacement in km, from blueprint §5.4. Used as an
#: engineering bound on ``grid_sample`` coordinates; not a physical claim.
MAX_DRIFT_KM = 12.0


def drift_scale(width: int) -> float:
    """Normalised-grid displacement for the maximum drift at a given width.

    ``grid_sample`` coordinates span ``[-1, 1]`` across the full width, so a
    ``MAX_DRIFT_KM`` displacement corresponds to
    ``MAX_DRIFT_KM / (width * KM_PER_PIXEL / 2)``. Using the real grid spacing
    and the actual patch width fixes blueprint defect S6 (which hard-coded
    ``12.0 / 150.0`` from a 301 grid at an assumed 1 km/px).

    Args:
        width: Spatial width of the model input in pixels.

    Returns:
        Displacement in normalised grid units.

    Raises:
        ValueError: If ``width`` is not positive.
    """
    if width < 1:
        raise ValueError("width must be positive")
    return MAX_DRIFT_KM / (width * KM_PER_PIXEL / 2.0)


@dataclass(frozen=True)
class QLFNConfig:
    """Immutable, hashable configuration for QLFN.

    Args:
        t_in: Radar history frames.
        t_out: Forecast lead steps.
        c_radar: Radar channels.
        c_nwp: NWP channels.
        geo_channels: Static geo/radar prior channels.
        latent_dim: Width of the folded latent space.
        rank: Number of angle-lift interaction modes (diagnostic only).
        n_observables: Number of read-out modes (diagnostic only).
        fold: Spatial-to-channel folding factor; H and W must be divisible by
            this after symmetric padding.
        flow_steps: ODE integration steps used at inference.
        wind_surrogate_indices: Scalar NWP channels used as a *declared
            surrogate* for drift. These are NOT U/V wind components.
    """

    t_in: int = T_IN
    t_out: int = T_OUT
    c_radar: int = C_RADAR
    c_nwp: int = C_NWP
    geo_channels: int = 6
    latent_dim: int = 96
    rank: int = 16
    n_observables: int = 32
    fold: int = FOLD_FACTOR
    flow_steps: int = 4
    wind_surrogate_indices: tuple[int, int] = WIND_SURROGATE_INDICES

    @property
    def folded_radar_channels(self) -> int:
        """Channels produced by folding radar + geo prior."""
        return (self.c_radar + self.geo_channels) * self.fold ** 2

    @property
    def subpixel_channels(self) -> int:
        """Channels the decode head must emit for every lead step."""
        return self.t_out * self.fold ** 2


@dataclass(frozen=True)
class FoldPlan:
    """Padding and crop algebra for one spatial shape.

    Args:
        height: Original height.
        width: Original width.
        fold: Spatial-to-channel folding factor.
        pad_top: Rows added above.
        pad_bottom: Rows added below.
        pad_left: Columns added left.
        pad_right: Columns added right.
    """

    height: int
    width: int
    fold: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @property
    def padded_height(self) -> int:
        """Padded height, divisible by ``fold``."""
        return self.height + self.pad_top + self.pad_bottom

    @property
    def padded_width(self) -> int:
        """Padded width, divisible by ``fold``."""
        return self.width + self.pad_left + self.pad_right

    @property
    def folded_height(self) -> int:
        """Latent height after folding."""
        return self.padded_height // self.fold

    @property
    def folded_width(self) -> int:
        """Latent width after folding."""
        return self.padded_width // self.fold

    @property
    def crop(self) -> tuple[int, int, int, int]:
        """Crop ``(top, bottom, left, right)`` restoring the original shape."""
        return (
            self.pad_top,
            self.pad_top + self.height,
            self.pad_left,
            self.pad_left + self.width,
        )


def fold_plan(height: int, width: int, fold: int = FOLD_FACTOR) -> FoldPlan:
    """Compute symmetric-needed padding so H and W are divisible by ``fold``.

    Args:
        height: Spatial height of the model input (a training patch, not
            necessarily the 301x301 scoring grid).
        width: Spatial width.
        fold: Folding factor, must be positive.

    Returns:
        A :class:`FoldPlan` with padding, folded shape and the crop slice.

    Raises:
        ValueError: If ``fold`` is not positive or a dimension is non-positive.
    """
    if fold < 1:
        raise ValueError("fold must be positive")
    if height < 1 or width < 1:
        raise ValueError("spatial dimensions must be positive")
    pad_h = (-height) % fold
    pad_w = (-width) % fold
    return FoldPlan(
        height=height,
        width=width,
        fold=fold,
        pad_top=pad_h // 2,
        pad_bottom=pad_h - pad_h // 2,
        pad_left=pad_w // 2,
        pad_right=pad_w - pad_w // 2,
    )


def validate_batch_shapes(
    x_radar: Sequence[int],
    x_nwp: Sequence[int],
    y_rain: Sequence[int] | None = None,
    config: QLFNConfig = QLFNConfig(),
    require_target: bool = False,
) -> tuple[str, ...]:
    """Validate shapes against the upstream IO seam without importing torch.

    Args:
        x_radar: Shape ``(B, T_in, C_radar, H, W)``.
        x_nwp: Shape ``(B, T_nwp, C_nwp, H, W)`` on the SAME grid.
        y_rain: Optional shape ``(B, T_out, 1, H, W)``. Required when
            ``require_target`` is True (a training batch).
        config: Immutable configuration.
        require_target: Treat a missing ``y_rain`` as missing evidence = FAIL.
            Inference callers leave this False.

    Returns:
        Problem descriptions; an empty tuple means the shapes are valid.
        A missing or malformed shape is always reported as a problem
        (missing evidence = FAIL).
    """
    problems: list[str] = []
    for name, shape, rank in (
        ("x_radar", x_radar, 5), ("x_nwp", x_nwp, 5),
    ):
        if len(tuple(shape)) != rank:
            problems.append(f"{name}: expected {rank} dims, got {tuple(shape)}")
    if problems:
        return tuple(problems)

    b_r, t_r, c_r, h_r, w_r = (int(v) for v in x_radar)
    b_n, t_n, c_n, h_n, w_n = (int(v) for v in x_nwp)
    if t_r != config.t_in:
        problems.append(f"x_radar: T_in must be {config.t_in}, got {t_r}")
    if c_r != config.c_radar:
        problems.append(f"x_radar: channels must be {config.c_radar}, got {c_r}")
    if t_n != T_NWP:
        problems.append(f"x_nwp: T_nwp must be {T_NWP}, got {t_n}")
    if c_n != config.c_nwp:
        problems.append(f"x_nwp: channels must be {config.c_nwp}, got {c_n}")
    if b_r != b_n:
        problems.append(f"batch mismatch: {b_r} vs {b_n}")
    if (h_r, w_r) != (h_n, w_n):
        problems.append(
            f"x_nwp must already be on the radar grid: {(h_n, w_n)} vs {(h_r, w_r)}"
        )
    if y_rain is None:
        if require_target:
            problems.append(
                "y_rain: required for a training batch; missing evidence = FAIL"
            )
    else:
        y_shape = tuple(int(v) for v in y_rain)
        expected = (b_r, config.t_out, 1, h_r, w_r)
        if y_shape != expected:
            problems.append(f"y_rain: expected {expected}, got {y_shape}")
    return tuple(problems)


def contract_report() -> dict[str, object]:
    """Return the contract constants as a plain dict for static self-checks.

    Returns:
        A JSON-serialisable snapshot of the load-bearing constants.
    """
    return {
        "t_in": T_IN,
        "t_nwp": T_NWP,
        "t_out": T_OUT,
        "c_radar": C_RADAR,
        "c_nwp": C_NWP,
        "radar_variables": list(RADAR_VARIABLES),
        "nwp_variables": list(NWP_VARIABLES),
        "cr_index": CR_INDEX,
        "hbr_index": HBR_INDEX,
        "vil_index": VIL_INDEX,
        "wind_surrogate_indices": list(WIND_SURROGATE_INDICES),
        "wind_surrogate_caveat": "scalar surrogates, NOT U/V wind vectors",
        "thresholds_mm_per_6min": list(THRESHOLDS_MM_PER_6MIN),
        "floor_csi2": FLOOR_CSI2,
        "oracle_csi2": ORACLE_CSI2,
        "headroom": HEADROOM,
        "seed_range": SEED_RANGE,
        "km_per_pixel": KM_PER_PIXEL,
        "max_drift_km": MAX_DRIFT_KM,
        "sealed_directions": list(SEALED_DIRECTIONS),
    }

"""QLag-FlowNet (QLFN) auditable reference implementation.

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the Lagrangian-alignment, quantum-inspired phase/tensor-network and
generative flow-matching families were rejected upstream. This package exists
only as an engineering reference: it does not train, evaluate, modify the
upstream repository, or claim any score. See ``docs/QLFN_SPEC.md`` and
``docs/QLFN_BLUEPRINT_REVIEW.md``.

The package root is standard-library only so the contract layer can be
verified on a workstation without PyTorch. Torch modules are loaded lazily.
"""
from __future__ import annotations

from typing import Any

from .contract import (
    CR_INDEX,
    C_NWP,
    C_RADAR,
    FLOOR_CSI2,
    HBR_INDEX,
    HEADROOM,
    NWP_VARIABLES,
    ORACLE_CSI2,
    RADAR_VARIABLES,
    SEED_RANGE,
    SEALED_DIRECTIONS,
    T_IN,
    T_NWP,
    T_OUT,
    THRESHOLDS_MM_PER_6MIN,
    VIL_INDEX,
    FoldPlan,
    QLFNConfig,
    fold_plan,
    validate_batch_shapes,
)

#: Display name used in papers, figures and documentation.
DISPLAY_NAME = "QLag-FlowNet"

__all__ = [
    "CR_INDEX", "C_NWP", "C_RADAR", "DISPLAY_NAME", "FLOOR_CSI2",
    "HBR_INDEX", "HEADROOM", "NWP_VARIABLES", "ORACLE_CSI2", "QLFN",
    "QLFNConfig", "RADAR_VARIABLES", "SEED_RANGE", "SEALED_DIRECTIONS",
    "T_IN", "T_NWP", "T_OUT", "THRESHOLDS_MM_PER_6MIN", "VIL_INDEX",
    "FoldPlan", "QLagrangeFlowNet", "fold_plan", "validate_batch_shapes",
]

_TORCH_EXPORTS = {"QLagrangeFlowNet", "QLFN"}


def __getattr__(name: str) -> Any:
    """Load torch-backed exports only when they are actually requested.

    Args:
        name: Attribute name looked up on the package.

    Returns:
        The requested attribute from ``qlfn.model``.

    Raises:
        AttributeError: If the name is not a known lazy export.
        ImportError: Propagated when PyTorch is unavailable (expected on the
            workstation; run model checks on the server instead).
    """
    if name in _TORCH_EXPORTS:
        from . import model as _model
        return _model.QLagrangeFlowNet if name == "QLFN" else getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

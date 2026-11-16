"""qmlbench: evidence-first integration layer for the radar-NWP baseline pool.

Public surface::

    from qmlbench import BenchSource, Catalog
    from qmlbench.results import standard_table, score_aware_table, decidability
    from qmlbench.gate import Claim, check, verdict

Zero third-party dependencies. This package does not train, evaluate or run a
model; it wires the upstream ``radar-nwp-bench`` baseline pool into a single
auditable catalogue and enforces the upstream evidence protocol.
"""
from .catalog import BASELINE_METADATA, BLUEPRINT_POOL, BaselineSpec, Catalog, GROUPS
from .sources import BenchNotFound, BenchSource
from .results import (
    BEST_TRAINED_CSI2,
    FLOOR_CSI2,
    HEADROOM,
    ORACLE_CSI2,
    PYSTEPS_CSI2,
    SEED_DISPERSION,
    SEED_RANGE,
    THRESHOLDS_MM_PER_6MIN,
    decidability,
    ranking,
    robustness_table,
    score_aware_table,
    standard_table,
)

__version__ = "0.1.0"

__all__ = [
    "BenchSource", "BenchNotFound", "Catalog", "BaselineSpec",
    "BASELINE_METADATA", "BLUEPRINT_POOL", "GROUPS",
    "standard_table", "robustness_table", "score_aware_table", "ranking",
    "decidability",
    "FLOOR_CSI2", "ORACLE_CSI2", "HEADROOM", "PYSTEPS_CSI2",
    "BEST_TRAINED_CSI2", "SEED_RANGE", "SEED_DISPERSION",
    "THRESHOLDS_MM_PER_6MIN",
]

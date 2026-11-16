"""Authoritative scores, kept separate from the catalogue.

Every number here is either read from an upstream JSON report / markdown table
at call time, or is a *load-bearing constant* with an explicit source in its
docstring. No score is transcribed into code by hand.

Important: the standard-loss table (``*.eval.json``) and the score-aware
matrix (``RESULTS_FINAL.md`` §10.1) are two different tables. They must never
be mixed in one ranking. ``MASTER.md`` §13 gives the score-aware matrix the
higher authority.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .sources import BenchSource

#: Zero-velocity persistence floor, ``persistence.eval.json`` thresholds.2.0.csi
#: (0.092419) and ``E08``. No learned parameters.
FLOOR_CSI2 = 0.0924

#: Perfect-QPE + no-motion oracle, ``E15-oracle-advection-ceiling.md`` §4.15.2.
#: Requires ground-truth rainfall as the initial field: privileged information,
#: not deployable.
ORACLE_CSI2 = 0.1653

#: The entire available headroom between the floor and the oracle.
HEADROOM = round(ORACLE_CSI2 - FLOOR_CSI2, 4)  # 0.0729

#: Peak-end spread across ten independent seeds, ``E13`` / ``MASTER.md`` §11.9 P1.
SEED_RANGE = 0.1671
SEED_DISPERSION = 9.59

#: Analytical PySTEPS advection, ``pysteps_adv.eval.json``. Explicitly NOT a
#: target to chase (``MASTER.md`` §10.2).
PYSTEPS_CSI2 = 0.1014

#: Best trained deep model under the authoritative score-aware matrix,
#: ``RESULTS_FINAL.md`` §10.1 (0.0896 -> 0.090 as tabulated).
BEST_TRAINED_CSI2 = 0.0896

#: Thresholds in mm per 6-minute accumulation. NOT mm/h.
THRESHOLDS_MM_PER_6MIN = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class ScoreRow:
    """A single model's verification scores from one report."""

    model: str
    table: str
    n_samples: int | None = None
    csi2: float | None = None
    csi05: float | None = None
    csi01: float | None = None
    csi4: float | None = None
    bias05: float | None = None
    far05: float | None = None
    fss2: float | None = None
    ablation: str | None = None

    @property
    def floor_delta(self) -> float | None:
        """Difference against the clean-split persistence floor.

        Only meaningful for rows from the **clean split**. The rendering layer
        must not compute this for robustness rows: the two splits contain
        disjoint cases with different rainfall intensity
        (``TGRS_EXPERIMENTS.md:468-469``).
        """
        if self.table == "robustness" or self.csi2 is None:
            return None
        return self.csi2 - FLOOR_CSI2

    def as_dict(self) -> dict[str, Any]:
        """Return the row as a plain mapping.

        Returns:
            A dict of every declared field (the derived property is excluded).

        Raises:
            None.
        """
        return asdict(self)


def _threshold_field(thresholds: dict[str, Any], key: str, field_name: str) -> float | None:
    """Read one numeric field out of a threshold bucket.

    Args:
        thresholds: the report's ``thresholds`` mapping.
        key: threshold key, e.g. ``"2.0"``.
        field_name: field within the bucket, e.g. ``"csi"``.

    Returns:
        The value as float, or ``None`` when absent or non-numeric.

    Raises:
        None.
    """
    entry = thresholds.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get(field_name)
    return float(value) if isinstance(value, (int, float)) else None


def _row_from_report(model: str, table: str, report: dict[str, Any]) -> ScoreRow:
    """Extract the fields the paper actually compares, with a stable key path."""
    thresholds = report.get("thresholds") or {}
    abl = report.get("ablation") or {}
    return ScoreRow(
        model=model,
        table=table,
        n_samples=report.get("n_samples"),
        csi2=_threshold_field(thresholds, "2.0", "csi"),
        csi05=_threshold_field(thresholds, "0.5", "csi"),
        csi01=_threshold_field(thresholds, "0.1", "csi"),
        csi4=_threshold_field(thresholds, "4.0", "csi"),
        bias05=_threshold_field(thresholds, "0.5", "bias"),
        far05=_threshold_field(thresholds, "0.5", "far"),
        fss2=_threshold_field(thresholds, "2.0", "fss"),
        ablation=abl.get("mode") if isinstance(abl, dict) else None,
    )


def _load_table(source: BenchSource, suffix: str, label: str) -> dict[str, ScoreRow]:
    """Build a score table from one family of upstream JSON reports.

    Args:
        source: read-only handle on the upstream repository.
        suffix: report suffix, ``.eval.json`` or ``.robustness.json``.
        label: value stored in ``ScoreRow.table``; ``"robustness"`` disables
            the floor-delta property.

    Returns:
        A mapping of model name to :class:`ScoreRow`.

    Raises:
        None. Unreadable reports are recorded as empty rows upstream.
    """
    return {
        name: _row_from_report(name, label, report)
        for name, report in source.scan_eval_reports(suffix).items()
    }


def standard_table(source: BenchSource) -> dict[str, ScoreRow]:
    """The clean-split standard-loss table, one row per ``*.eval.json`` (n=538)."""
    return _load_table(source, ".eval.json", "standard")


def robustness_table(source: BenchSource) -> dict[str, ScoreRow]:
    """The missing-frame robustness split (n=962), disjoint from the clean split.

    ``TGRS_EXPERIMENTS.md`` §5: these two splits contain completely different
    cases with different rainfall intensity. Their scores must not be
    subtracted from one another, and neither may be compared against the
    clean-split persistence floor.
    """
    return _load_table(source, ".robustness.json", "robustness")


def score_aware_table(source: BenchSource) -> dict[str, float]:
    """Authoritative ``CSI@2.0`` from ``RESULTS_FINAL.md`` §10.1."""
    return {
        name: float(row["CSI@2.0"])
        for name, row in source.score_aware_table().items()
        if "CSI@2.0" in row
    }


def ranking(rows: dict[str, ScoreRow]) -> list[ScoreRow]:
    """Descending by CSI@2.0, with unscored rows last and deterministically named."""
    return sorted(
        rows.values(),
        key=lambda r: (-(r.csi2 if r.csi2 is not None else float("-inf")), r.model),
    )


def decidability() -> dict[str, Any]:
    """Quantify whether a single-run improvement is distinguishable from noise.

    ``MASTER.md`` §11.9 P1: the single-seed range is 2.29x the entire
    available headroom. A single peak above the floor is therefore not
    evidence of an improvement.
    """
    ratio = round(SEED_RANGE / HEADROOM, 4)
    return {
        "floor": FLOOR_CSI2,
        "oracle": ORACLE_CSI2,
        "headroom": HEADROOM,
        "single_seed_range": SEED_RANGE,
        "range_over_headroom": ratio,
        "seed_dispersion": SEED_DISPERSION,
        "best_trained": BEST_TRAINED_CSI2,
        "best_trained_beats_floor": BEST_TRAINED_CSI2 > FLOOR_CSI2,
        "verdict": (
            "UNDECIDABLE: single-run peaks are indistinguishable from seed noise"
            if ratio > 1.0
            else "decidable"
        ),
    }

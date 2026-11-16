"""Case-level audit and valid-step masking protocol (standard library only).

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): this module is a data-side audit tool, not an implementation of the
rejected mechanism families.

Scope discipline: upstream ``evaluate.py`` / ``metrics/precipitation.py`` have
no ``valid_step_mask`` (verified: ``precipitation.py:262 update()`` takes no
mask). Anything produced here is therefore **NOT COMPARABLE** with the
authoritative 26-baseline table. ``MASTER.md:204`` forbids subtracting clean
and robustness split scores. This tool never patches upstream code.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NOT_COMPARABLE = (
    "NOT_COMPARABLE: this protocol is not wired into upstream evaluate.py; "
    "do not compare its counts with configs/baselines.json or master_table.md. "
    "MASTER.md:204 forbids clean-vs-robustness score subtraction."
)


@dataclass(frozen=True)
class CaseStat:
    """Per-case statistics used for a stratified split.

    Args:
        case_id: Stable case identifier.
        n_windows: Number of scored windows.
        missing_steps: Indices of lead steps with no ground-truth observation.
        bias_05: Aggregated BIAS@0.5 of the analytic Z-R baseline.
        p99: 99th percentile rain rate (mm per 6min).
        wet_20_ratio: Fraction of pixels at or above 2.0 mm per 6min.
    """

    case_id: str
    n_windows: int
    missing_steps: tuple[int, ...]
    bias_05: float
    p99: float
    wet_20_ratio: float


@dataclass(frozen=True)
class SplitReport:
    """Result of a stratified split, including an *actual* balance check.

    Args:
        train: Case ids in the training subset.
        val: Case ids in the validation subset.
        test: Case ids in the test subset.
        bias_means: Mean BIAS@0.5 per subset.
        max_relative_difference: Largest pairwise relative difference of the
            three means; ``None`` when a subset is empty or a mean is zero.
        balanced: Whether the requested tolerance is actually met.
        tolerance: Requested maximum relative difference.
        caveat: Scope statement that must accompany any use of this split.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    bias_means: tuple[float, float, float]
    max_relative_difference: float | None
    balanced: bool
    tolerance: float
    caveat: str = NOT_COMPARABLE


def stratified_case_split(
    stats: Sequence[CaseStat],
    val_n: int,
    test_n: int,
    seed: int = 42,
    tolerance: float = 0.10,
) -> SplitReport:
    """Split cases by sorted rain abundance, then *measure* BIAS balance.

    Unlike the blueprint's ``stratified_62case_split``, this function does not
    assume the split is balanced: it computes the achieved relative difference
    and reports ``balanced=False`` when the tolerance is not met.

    Args:
        stats: Per-case statistics.
        val_n: Number of validation cases.
        test_n: Number of test cases.
        seed: RNG seed for reproducibility.
        tolerance: Maximum allowed relative BIAS@0.5 difference.

    Returns:
        A :class:`SplitReport` with the realised balance.

    Raises:
        ValueError: If the requested subset sizes exceed the case count.
    """
    if val_n + test_n > len(stats):
        raise ValueError("val_n + test_n exceeds the number of cases")
    rng = random.Random(seed)
    ordered = sorted(
        stats, key=lambda c: (c.wet_20_ratio, c.bias_05), reverse=True
    )
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    val = tuple(ordered[i].case_id for i in indices[:val_n])
    test = tuple(ordered[i].case_id for i in indices[val_n:val_n + test_n])
    train = tuple(ordered[i].case_id for i in indices[val_n + test_n:])

    by_id = {c.case_id: c for c in stats}
    means = tuple(
        sum(by_id[cid].bias_05 for cid in group) / len(group) if group else 0.0
        for group in (train, val, test)
    )
    nonzero = [m for m in means if m > 0.0]
    if len(nonzero) == len(means) == 3:
        max_diff = (max(means) - min(means)) / min(means)
    else:
        max_diff = None
    return SplitReport(
        train=train,
        val=val,
        test=test,
        bias_means=means,
        max_relative_difference=max_diff,
        balanced=max_diff is not None and max_diff <= tolerance,
        tolerance=tolerance,
    )


def valid_step_mask(steps_present: Sequence[int], t_out: int) -> tuple[int, ...]:
    """Build a binary lead-step mask; a missing step is 0 and is NEVER filled.

    Args:
        steps_present: Lead-step indices with a real observation.
        t_out: Total number of forecast lead steps.

    Returns:
        A length-``t_out`` tuple of 0/1 values.

    Raises:
        ValueError: If an index is outside ``[0, t_out)``.
    """
    if any(step < 0 or step >= t_out for step in steps_present):
        raise ValueError("lead-step index out of range")
    present = set(steps_present)
    return tuple(1 if step in present else 0 for step in range(t_out))


def masked_contingency(
    pred_hit: Sequence[float],
    truth_hit: Sequence[float],
    mask: Sequence[int],
) -> tuple[float, float, float]:
    """Canonical pure-Python soft contingency counts (binary mask only).

    RETRACTED claim: v1.0 said this "fixes blueprint defect T2". It does not.
    The blueprint multiplies ``p_hit``/``t_hit`` by a binary ``mask`` first,
    and because ``m**2 == m`` its ``fp = p*m*(1-t*m)`` equals this function's
    ``fp = p_m*(m-t_m)``. The two are algebraically identical, so no defect
    was fixed. What this function genuinely adds is the binary-mask
    precondition and a reference implementation for cross-checking the torch
    version (see ``losses.masked_contingency_torch``).

    Args:
        pred_hit: Soft predicted hits in ``[0, 1]`` (sigmoid relaxed CSI).
        truth_hit: Hard observed hits, 0 or 1.
        mask: 1 when the lead step has a real observation, else 0. Must be
            binary; soft masks break the equivalence.

    Returns:
        ``(tp, fp, fn)`` as floats. A fully masked batch returns all zeros.

    Raises:
        ValueError: If the three sequences have different lengths, or if any
            mask value is not 0 or 1.
    """
    if not (len(pred_hit) == len(truth_hit) == len(mask)):
        raise ValueError("pred_hit, truth_hit and mask must have equal length")
    if any(int(m) not in (0, 1) or m != int(m) for m in mask):
        raise ValueError("mask must be binary 0/1")
    tp = fp = fn = 0.0
    for p, t, m in zip(pred_hit, truth_hit, mask):
        if not m:
            continue
        p_m = float(p) * m
        t_m = float(t) * m
        tp += p_m * t_m
        fp += p_m * (m - t_m)
        fn += (m - p_m) * t_m
    return tp, fp, fn


def audit_cases(stats: Sequence[CaseStat]) -> dict[str, object]:
    """Summarise case coverage without inventing missing measurements.

    Args:
        stats: Per-case statistics read from an explicit input file.

    Returns:
        A JSON-serialisable report. Cases with no BIAS value must be supplied
        as ``None``-free float by the caller; absent data is never imputed here.

    Raises:
        ValueError: If the input is empty (missing evidence = FAIL).
    """
    if not stats:
        raise ValueError("no case statistics supplied; missing evidence = FAIL")
    total_windows = sum(c.n_windows for c in stats)
    gapped = sum(1 for c in stats if c.missing_steps)
    return {
        "n_cases": len(stats),
        "n_windows": total_windows,
        "n_cases_with_missing_steps": gapped,
        "n_missing_lead_steps": sum(len(c.missing_steps) for c in stats),
        "bias_05_min": min(c.bias_05 for c in stats),
        "bias_05_max": max(c.bias_05 for c in stats),
        "p99_max": max(c.p99 for c in stats),
        "caveat": NOT_COMPARABLE,
    }


def load_case_stats(path: Path) -> tuple[CaseStat, ...]:
    """Read case statistics from a JSON list produced by an external audit.

    Args:
        path: Path to a JSON file with keys ``case_id``, ``n_windows``,
            ``missing_steps``, ``bias_05``, ``p99``, ``wet_20_ratio``.

    Returns:
        A tuple of :class:`CaseStat` records.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required field is missing (never silently defaulted).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        CaseStat(
            case_id=row["case_id"],
            n_windows=int(row["n_windows"]),
            missing_steps=tuple(int(s) for s in row["missing_steps"]),
            bias_05=float(row["bias_05"]),
            p99=float(row["p99"]),
            wet_20_ratio=float(row["wet_20_ratio"]),
        )
        for row in payload
    )


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    """Write an audit artifact with the non-comparability caveat attached.

    R12 requires the *filename* to carry the caveat, not just the JSON body,
    so a generated artifact can never be mistaken for an authoritative result.

    Args:
        path: Destination JSON path; parent directories are created. The
            filename must contain ``NOT_COMPARABLE``.
        payload: Report body.

    Returns:
        None.

    Raises:
        ValueError: If the filename lacks the ``NOT_COMPARABLE`` marker.
        OSError: Propagated when the file cannot be written.
    """
    if "NOT_COMPARABLE" not in path.name:
        raise ValueError(
            "audit artifact filename must contain NOT_COMPARABLE (R12)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["caveat"] = NOT_COMPARABLE
    path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

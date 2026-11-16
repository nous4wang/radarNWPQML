"""Shared row helpers for the two QLFN self-check scripts.

SEALED (upstream ``MASTER.md`` §9 / §11.8.2, lines 1048, 1083, 1085, 1088,
1456): the mechanism families are rejected upstream. This module contains no
model code; it only formats the results of an engineering self-test. It is
standard-library only so the workstation contract check can import it.

Both ``scripts/verify_qlfn_contract.py`` (zero dependency) and
``scripts/qlfn_smoke_test.py`` (torch, remote only) previously defined the
same ``check()`` helper and the same rendering loop. Code review flagged the
duplication; this module is the single definition.
"""
from __future__ import annotations

CheckRow = tuple[str, bool, str]


def check(name: str, condition: bool, detail: str = "") -> CheckRow:
    """Build one self-check result row.

    Args:
        name: Short human-readable label for the requirement being checked.
        condition: Truthy value; coerced to ``bool`` so tensors and sequences
            cannot leak into the report.
        detail: Optional context printed alongside the verdict.

    Returns:
        A ``(name, passed, detail)`` tuple.

    Raises:
        TypeError: If ``name`` is not a string (a malformed check is a bug,
            not a failed check).
    """
    if not isinstance(name, str):
        raise TypeError("check name must be a string")
    return name, bool(condition), detail


def render(rows: list[CheckRow], ok_banner: str) -> int:
    """Print a self-check report and return a process exit code.

    Args:
        rows: Result rows produced by :func:`check`.
        ok_banner: Banner printed only when every row passed.

    Returns:
        ``0`` when all rows passed, ``1`` otherwise. Skipped checks must be
        supplied as a failing row by the caller (missing evidence = FAIL).

    Raises:
        ValueError: If ``rows`` is empty (an empty report must never PASS).
    """
    if not rows:
        raise ValueError("empty check report; missing evidence = FAIL")
    width = max(len(name) for name, _, _ in rows)
    for name, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in rows if not ok]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    print(ok_banner)
    return 0


__all__ = ["CheckRow", "check", "render"]

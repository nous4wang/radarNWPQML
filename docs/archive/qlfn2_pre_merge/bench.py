"""Locating the benchmark, so that qlfn2 can sit on the upstream seam.

The benchmark is a **separate tree**: ``/home/wf/radarNWP`` in the remote
image, a sibling checkout locally.  qlfn2 is not allowed to vendor it and is
not allowed to modify it -- the comparison is only meaningful if the harness
that scores the models is the harness that ships with the benchmark.

This module is the single place that knows where the benchmark might be.
"""
from __future__ import annotations

import os
import sys

#: Where the benchmark may live, in order of preference.
BENCH_CANDIDATES: tuple[str, ...] = (
    os.environ.get("RADAR_NWP_BENCH") or "",
    "/home/wf/radarNWP",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "1radarNWPprediction", "radar-nwp-bench",
    ),
    "R:/1radarNWPprediction/radar-nwp-bench",
)


def add_bench_to_path() -> str | None:
    """Put the benchmark on the import path.

    Returns:
        The benchmark root that was added, or ``None`` when the benchmark is
        not importable here.  The caller decides whether that is fatal.
    """
    for root in BENCH_CANDIDATES:
        if root and os.path.isdir(os.path.join(root, "radar_nwp_bench")):
            root = os.path.abspath(root)
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


def require_bench() -> str:
    """Put the benchmark on the import path or fail with a usable message.

    Returns:
        The benchmark root.

    Raises:
        RuntimeError: when the benchmark cannot be found.
    """
    root = add_bench_to_path()
    if root is None:
        raise RuntimeError(
            "the benchmark (radar_nwp_bench) is not importable here. "
            "Set RADAR_NWP_BENCH to its root, or run from a checkout that "
            "has it as a sibling."
        )
    return root


__all__ = ["add_bench_to_path", "require_bench", "BENCH_CANDIDATES"]

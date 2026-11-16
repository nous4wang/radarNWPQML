#!/usr/bin/env python3
"""Score qlfn2 with the benchmark's own evaluation loop.

Like ``scripts/train_qlfn2.py``, this wrapper changes nothing in the
benchmark: it registers the model and hands the command line to the
benchmark's ``scripts/evaluate.py``.  The stitching, the verifier, the
thresholds, the FSS radii and the case-level split are therefore the
benchmark's, and the score is directly comparable with the ones the
benchmark reports for its own baselines.

Usage:
    python scripts/eval_qlfn2.py --model qlfn2 \\
        --checkpoint runs/qlfn2/last.pt \\
        --data-root /home/wf/data/radarNWP/BJ_mmap --split test \\
        --thresholds 0.5 1 2 4 --fss-radii 1 2 4
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlfn2.qlfn2_model import (                             # noqa: E402
    QLFN2Adapter, require_bench, register_qlfn2,
)

BENCH_ROOT = Path(require_bench())

MODEL_NAME = "qlfn2"

register_qlfn2(MODEL_NAME)


def load_bench_script(name: str):
    """Import one of the benchmark's own scripts as a module.

    Args:
        name: the script's stem, e.g. ``"evaluate"``.

    Returns:
        The imported module, with its ``main`` intact.
    """
    path = BENCH_ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"the benchmark has no script at {path}")
    spec = importlib.util.spec_from_file_location(f"_rnb_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Run the benchmark's evaluation loop."""
    module = load_bench_script("evaluate")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

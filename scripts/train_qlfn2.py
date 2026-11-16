#!/usr/bin/env python3
"""Train qlfn2 with the benchmark's own training loop.

The point of this wrapper is that it changes **nothing** in the benchmark.
It registers qlfn2 under the name ``qlfn2`` and then hands the command
line to the benchmark's ``scripts/train.py``:

    * the dataset, the split and the interpolation policy are the
      benchmark's;
    * the loss is one of the benchmark's own objectives;
    * the checkpoint layout is the benchmark's.

So a number produced by this script and a number produced by
``scripts/train.py --model physical_qpe`` were produced by the same
harness, and the difference between them is the model.

Usage:
    python scripts/train_qlfn2.py --model qlfn2 \\
        --data-root /home/wf/data/radarNWP/BJ_mmap --epochs 3 \\
        --loss balanced --qpe-weight 0.5 --out-dir runs/qlfn2
"""
from __future__ import annotations

import os

# cuBLAS needs a fixed workspace before the CUDA context exists, otherwise
# ``torch.use_deterministic_algorithms(True)`` raises on the first matmul.
# It has to be set before torch is imported, so it happens here, before
# anything else touches the interpreter.
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

#: The name qlfn2 is reached by.  It is registered here, not inside the
#: qlfn2 package, so that the library itself stays importable without the
#: benchmark -- the library is the science, the registration is the seam.
MODEL_NAME = "qlfn2"

register_qlfn2(MODEL_NAME)


def load_bench_script(name: str):
    """Import one of the benchmark's own scripts as a module.

    Args:
        name: the script's stem, e.g. ``"train"``.

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
    """Run the benchmark's training loop."""
    module = load_bench_script("train")
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

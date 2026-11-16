#!/usr/bin/env python3
"""Train qlfn2 in-process and report the peak video memory it used.

``scripts/train_qlfn2.py`` hands the command line to the benchmark's
``scripts/train.py``.  That is the right thing for a training run, but it
means the process that trains is not the process that can report how much
card it used: a separate ``python -c`` afterwards would start with an empty
allocator and report a peak of zero.

This wrapper keeps the training in one process.  It registers qlfn2 exactly
the way ``scripts/train_qlfn2.py`` does, runs the benchmark's own training
loop with the same command line, and then prints the allocator's peak
allocation and peak reservation from the process that did the work.

It exists for one purpose: to find out whether a batch size fits on the
card *before* a long job discovers it half way through.

Usage:
    python scripts/train_qlfn2_mem.py --model qlfn2 \\
        --data-root /home/wf/data/radarNWP/BJ_mmap --epochs 1 \\
        --max-steps 5 --batch-size 16 --patch-size 128 --slide-stride 64 \\
        --device cuda --out-dir runs/qlfn2_smoke/train
"""
from __future__ import annotations

import os

# cuBLAS needs a fixed workspace before the CUDA context exists, otherwise
# ``torch.use_deterministic_algorithms(True)`` raises on the first matmul.
# It has to be set before torch is imported, so it happens here.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys                                      # noqa: E402
from pathlib import Path                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch                                   # noqa: E402

from qlfn2.qlfn2_model import QLFN2Adapter, register_qlfn2  # noqa: E402

import train_qlfn2                              # noqa: E402

MODEL_NAME = "qlfn2"

register_qlfn2(MODEL_NAME)


def main() -> int:
    """Run the benchmark's training loop, then report the peak memory.

    Returns:
        The benchmark's own exit code.
    """
    module = train_qlfn2.load_bench_script("train")
    code = int(module.main())

    print()
    print("--- peak device memory -------------------------------------------")
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"peak allocated  : {allocated:.2f} GiB")
        print(f"peak reserved   : {reserved:.2f} GiB")
        print(f"card total      : {total:.2f} GiB")
        print(f"headroom        : {total - reserved:.2f} GiB")
    else:
        print("no device was used, so there is no peak to report")

    return code


if __name__ == "__main__":
    raise SystemExit(main())

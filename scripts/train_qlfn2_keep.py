#!/usr/bin/env python3
"""Train qlfn2 while keeping a copy of every epoch.

This is the qlfn2 counterpart of the benchmark's
``scripts/train_keep_epochs.py``.  The benchmark's own comment explains why
the wrapper exists:

    ``scripts/train.py`` writes one file, ``last.pt``, and rewrites it at the
    end of every epoch.  A run that is still training therefore has no stable
    checkpoint to point at: the file under a fixed name changes underneath
    the reader.  This wrapper keeps a copy per epoch so the score can be read
    off a known epoch instead of off a moment.

That matters because the benchmark's own epoch curve (E08/E19) shows the
score of these models peaking early and then decaying: ``physical_qpe``
scores 0.1226 at epoch 0, is already below the persistence floor at epoch 1,
and is at 51 % of the floor by epoch 2.  A run that scores ``last.pt`` and
calls the result "the model" is therefore reporting the collapsed end of the
run.

The patch is on ``torch.save`` rather than on the training loop, so the
training itself is bit-for-bit the benchmark's code -- the only difference
is that the checkpoints survive.
"""
from __future__ import annotations

import os

# cuBLAS needs a fixed workspace before the CUDA context exists.  It has to
# be set before torch is imported, so it happens here, before anything else
# touches the interpreter.
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
    """Import one of the benchmark's own scripts as a module."""
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


def install_epoch_keeper():
    """Keep a copy of every ``last.pt`` the training loop writes.

    The benchmark's ``train.py`` saves the state dict at the end of every
    epoch under the fixed name ``last.pt``.  Every such save is mirrored to
    ``epoch_NN.pt`` in the same directory, numbered from zero in the order
    the epochs happened, so the score of a run can be read off a known epoch.

    Returns the counter dictionary, so a caller can report how many epochs
    were kept.
    """
    import torch

    original_save = torch.save
    counter = {"n": 0}

    def patched_save(obj, path, *args, **kwargs):
        original_save(obj, path, *args, **kwargs)
        text = str(path)
        if text.endswith("last.pt"):
            directory = os.path.dirname(text)
            keep = os.path.join(directory, f"epoch_{counter['n']:02d}.pt")
            original_save(obj, keep)
            counter["n"] += 1
            print(f"kept {keep}", flush=True)

    torch.save = patched_save
    return counter


def main() -> int:
    """Run the benchmark's training loop, keeping every epoch."""
    counter = install_epoch_keeper()
    module = load_bench_script("train")
    rc = int(module.main())
    print(f"kept {counter['n']} epoch checkpoints", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

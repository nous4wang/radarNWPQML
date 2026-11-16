"""Check that qlfn2 sits on the upstream registry, next to the baselines.

Run this **before** submitting a job.  It answers three questions that a
failed training run would otherwise answer late:

* does the registry accept qlfn2, and does it survive the benchmark's own
  discovery pass (which imports every ``<name>/model.py`` it can find)?
* does the upstream trainer's ``--model`` list contain it, so that the
  command line accepts it?
* do the two arms have comparable capacity, so that a score difference is
  not a parameter-count difference?

Usage:
    python tools/_registry_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qlfn2.qlfn2_model import (                             # noqa: E402
    QLFN2Adapter, require_bench, register_qlfn2,
)

BENCH_ROOT = Path(require_bench())
sys.path.insert(0, str(BENCH_ROOT))

from radar_nwp_bench.models import (                         # noqa: E402
    build_model,
    list_models,
    registry_snapshot,
)


def main() -> int:
    """Register qlfn2 and report what the registry now holds."""
    register_qlfn2("qlfn2")

    names = list_models()
    snapshot = registry_snapshot()
    print(f"bench root       : {BENCH_ROOT}")
    print(f"registered models: {len(names)}")
    print(f"qlfn2 registered : {'qlfn2' in names}")
    print(f"qlfn2 -> {snapshot.get('qlfn2')}")
    print(f"physical_qpe -> {snapshot.get('physical_qpe')}")

    shapes = dict(in_radar_c=10, in_nwp_c=27, in_len=11, out_len=10)
    ours = build_model("qlfn2", **shapes)
    theirs = build_model("physical_qpe", **shapes)
    n_ours = ours.num_parameters()
    n_theirs = theirs.num_parameters()
    print(f"qlfn2 params      : {n_ours:,}")
    print(f"physical_qpe params: {n_theirs:,}")
    print(f"ratio qlfn2/physical_qpe: {n_ours / n_theirs:.2f}")

    # the shapes the trainer will hand the model
    import torch

    radar = torch.zeros(2, 11, 10, 32, 32)
    nwp = torch.zeros(2, 2, 27, 32, 32)
    with torch.no_grad():
        out = ours(radar, nwp)
    print(f"forward shape     : {tuple(out.shape)}")
    ok = tuple(out.shape) == (2, 10, 1, 32, 32)
    print(f"seam honoured     : {ok}")
    return 0 if ok and "qlfn2" in names else 1


if __name__ == "__main__":
    raise SystemExit(main())

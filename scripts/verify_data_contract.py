"""Check the declared input/output contract against the *real* data pipeline.

    # benchmark server
    RADAR_NWP_LIVE_ROOT=/home/wf/radarNWP \
        /home/wf/.conda/envs/radar_nwp/bin/python \
        scripts/verify_data_contract.py

    # workstation without torch: SKIP, exit 0
    python scripts/verify_data_contract.py

``smoke_forward.py`` feeds zero tensors to the adapters. That proves the model
side of the seam but not the data side: the catalogue claims 11 radar frames x
10 variables, 2 NWP time slices x 27 variables, and 10 forecast frames of
1 channel. This script instantiates the upstream ``RadarNWPDataset`` on the
server's real 9.2G dataset and reads one sample, so the claim is checked
against actual files rather than against a copy of the documentation.

No training, no checkpoints, no metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Unbatched sample shapes; the DataLoader adds B.
EXPECTED = {
    "x_radar": [11, 10, 64, 64],
    "x_nwp": [2, 27, 64, 64],
    "y_rain": [10, 1, 64, 64],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default=os.environ.get("RADAR_NWP_LIVE_ROOT"))
    ap.add_argument("--data-root",
                    default=os.environ.get("RADAR_NWP_DATA_ROOT",
                                          "/home/wf/data/radarNWP/BJ"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "data_contract.json"))
    args = ap.parse_args()

    if not args.live_root:
        print("SKIP: RADAR_NWP_LIVE_ROOT is not set (no live upstream package).")
        return 0

    live_root = Path(args.live_root).resolve()
    sys.path.insert(0, str(live_root))
    try:
        import torch
        from radar_nwp_bench.data.dataset import RadarNWPDataset
    except Exception as exc:  # noqa: BLE001 - missing torch is a SKIP
        print(f"SKIP: cannot import the runtime stack: {type(exc).__name__}: {exc}")
        return 0

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"SKIP: data root does not exist: {data_root}")
        return 0

    dataset = RadarNWPDataset(
        str(data_root),
        in_len=11,
        out_len=10,
        patch_size=64,
        random_crop=False,
        max_cases=2,
        max_samples=8,
    )
    sample = dataset[0]
    observed = {
        key: list(sample[key].shape) for key in EXPECTED if key in sample
    }
    dtypes = {
        key: str(sample[key].dtype) for key in EXPECTED if key in sample
    }
    shape_ok = all(observed.get(k) == v for k, v in EXPECTED.items())
    widths = {
        "radar_vars": len(dataset.radar_vars),
        "nwp_vars": len(dataset.nwp_vars),
        "nwp_frames": dataset.nwp_frames,
        "grid_shape": list(dataset.grid_shape),
        "n_samples": len(dataset),
    }
    widths_ok = (widths["radar_vars"] == 10 and widths["nwp_vars"] == 27
                 and widths["nwp_frames"] == 2)

    print(f"data root : {data_root}")
    print(f"observed  : {observed}")
    print(f"expected  : {EXPECTED}")
    print(f"dtypes    : {dtypes}")
    print(f"widths    : {widths}")
    print("VERDICT:", "PASS" if shape_ok and widths_ok else "FAIL")

    payload = {
        "meta": {"live_root": str(live_root), "data_root": str(data_root),
                 "torch": torch.__version__},
        "expected": EXPECTED,
        "observed": observed,
        "dtypes": dtypes,
        "widths": widths,
        "verdict": "PASS" if shape_ok and widths_ok else "FAIL",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0 if shape_ok and widths_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

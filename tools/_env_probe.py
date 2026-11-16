#!/usr/bin/env python3
"""Read-only remote environment probe for the R:/1radarNWPQML mirror.

Zero training, zero model inference, zero data reads and zero writes inside
either repository: this only stats paths and collects interpreter/GPU
metadata, then writes one JSON file to /tmp. It exists so that a
``--sync ... -- 'cmd'`` round trip can be verified without spending GPU time.

    /home/wf/.conda/envs/radar_nwp/bin/python \
        /home/wf/radarNWPQML/tools/_env_probe.py --out /tmp/qml_env_probe.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_CANDIDATES = (
    "/home/wf/data/radarNWP/BJ",
    "/home/wf/data/radarNWP/BJ_mmap",
    "/home/wf/data/radarNWP/BJ_mmp",
    "/home/wf/data/radarNWP/BJ_packed",
)
CODE_ROOTS = (
    "/home/wf/radarNWPQML",
    "/home/wf/radarNWP",
)
UPSTREAM_ENTRIES = (
    "scripts/train.py",
    "scripts/train_keep_epochs.py",
    "scripts/evaluate.py",
    "radar_nwp_bench/models/registry.py",
)


def probe() -> dict[str, object]:
    def stat(path: str) -> dict[str, object]:
        p = Path(path)
        return {
            "path": path,
            "exists": p.exists(),
            "kind": "dir" if p.is_dir() else "file" if p.is_file() else "none",
        }

    data = {p: stat(p) for p in DATA_CANDIDATES}
    mmap = Path("/home/wf/data/radarNWP/BJ_mmap")
    n_samples = (
        sum(1 for entry in mmap.iterdir() if entry.is_dir())
        if mmap.is_dir() else None
    )

    upstream = {
        name: stat(f"/home/wf/radarNWP/{name}") for name in UPSTREAM_ENTRIES
    }
    gpu: object
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=20,
        )
        gpu = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError) as exc:
        gpu = f"unavailable: {exc}"

    return {
        "probe": "tools/_env_probe.py",
        "executed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "code_roots": {p: stat(p) for p in CODE_ROOTS},
        "data_paths": data,
        "n_mmap_sample_dirs": n_samples,
        "upstream_entries": upstream,
        "gpus": gpu,
        "scope": "read-only; zero training / inference / data read / repo write",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"probe written: {args.out}")
    print(f"  mirror root : {payload['code_roots']['/home/wf/radarNWPQML']}")
    print(f"  BJ_mmap     : {payload['data_paths']['/home/wf/data/radarNWP/BJ_mmap']}")
    print(f"  BJ_mmp      : {payload['data_paths']['/home/wf/data/radarNWP/BJ_mmp']}")
    print(f"  sample dirs : {payload['n_mmap_sample_dirs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

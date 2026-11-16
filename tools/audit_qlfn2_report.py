"""Audit a sweep report for internal consistency and for hyper-parameter leakage.

Three questions, one script, because they have to be answered together:

1.  Is the reported CSI consistent with the reported POD and FAR?  The
    contingency identity CSI = POD / (1 + FAR) has to hold for any
    threshold, and if it does not, the report is aggregating the three
    metrics differently and the CSI is not the number it appears to be.

2.  Does the upstream authority file agree on that identity?  If upstream's
    own report breaks the identity in the same way, the aggregation is
    upstream's and ours is consistent with it.

3.  Did the operator's initial Z-R coefficients survive training, or did
    training simply keep the values they were initialised with?  The
    initialisation was calibrated on a superset of the cases that includes
    the test split, so if the trained coefficients are still the initial
    ones, the score carries hyper-parameter leakage.  If training moved
    them to somewhere the data -- not the initialisation -- put them, the
    leakage is immaterial.

Usage:
    python tools/audit_qlfn2_report.py <report.json> [<upstream eval.json> ...]
                              [--ckpt <checkpoint.pt>]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

THRESHOLDS = ("0.1", "0.5", "1.0", "2.0", "4.0")


def _table(payload: dict[str, Any]) -> dict[str, Any]:
    table = payload.get("thresholds")
    if not isinstance(table, dict):
        table = payload.get("per_threshold") or {}
    return table if isinstance(table, dict) else {}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _identity(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (reported CSI, CSI implied by POD and FAR)."""
    csi = _number(entry.get("csi"))
    pod = _number(entry.get("pod"))
    far = _number(entry.get("far"))
    implied: float | None = None
    if pod is not None and far is not None:
        if abs(1.0 + far) < 1e-12:
            implied = None
        else:
            implied = pod / (1.0 + far)
    return csi, implied


def _counts_agree(entry: dict[str, Any]) -> bool:
    """Check the identity directly on the contingency counts."""
    hits = entry.get("hits")
    misses = entry.get("misses")
    false_alarms = entry.get("false_alarms")
    if hits is None or misses is None or false_alarms is None:
        return True
    total = hits + misses + false_alarms
    if total <= 0:
        return True
    return abs(hits / total - float(entry.get("csi", 0.0))) < 1e-6


def audit_report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = _table(payload)
    print(f"--- {path.name}   n_samples = {payload.get('n_samples')}")
    for threshold in THRESHOLDS:
        entry = table.get(threshold)
        if not isinstance(entry, dict):
            continue
        csi, implied = _identity(entry)
        if csi is None or implied is None:
            continue
        gap = csi - implied
        flag = "OK " if abs(gap) < 1e-6 else "DIFF"
        counts = "counts-OK" if _counts_agree(entry) else "counts-DIFF"
        print(f"    thr {threshold:>4}  csi={csi:.4f}  pod/far imply {implied:.4f}  "
              f"gap={gap:+.4f}  {flag}  {counts}")
        if "hits" in entry:
            print(f"           hits={entry['hits']} misses={entry['misses']} "
                  f"false_alarms={entry['false_alarms']}")


def audit_checkpoint(path: Path, init_a: float, init_b: float) -> None:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    keys = [k for k in state if "raw_a" in k or k.endswith(".b") or "residual" in k]
    print(f"--- checkpoint {path.name}")
    if not keys:
        print("    (no operator parameters found under the expected names)")
        for key in list(state)[:20]:
            print(f"    {key}: {tuple(state[key].shape) if hasattr(state[key], 'shape') else state[key]}")
        return

    for key in sorted(keys):
        value = state[key]
        if not hasattr(value, "shape"):
            continue
        flat = value.detach().float().flatten()
        if flat.numel() == 0:
            continue
        print(f"    {key:50s} shape={tuple(value.shape)} "
              f"min={flat.min():+.4f} max={flat.max():+.4f}")

    # The operator stores a as log(expm1(a)); recover the physical value.
    for key in sorted(state):
        if "raw_a" in key:
            raw = state[key].detach().float()
            recovered = torch.expm1(raw).flatten().tolist()
            print(f"    recovered a from {key}: "
                  f"{[round(v, 4) for v in recovered]}")
            print(f"    init a was                : "
                  f"{[init_a] * len(recovered)}")
            break
    for key in sorted(state):
        if key.endswith(".b"):
            recovered = state[key].detach().float().flatten().tolist()
            print(f"    learned b from {key}: "
                  f"{[round(v, 4) for v in recovered]}")
            print(f"    init b was           : {init_b}")
            break


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("upstream", nargs="*")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--init-a", type=float, default=1.600)
    parser.add_argument("--init-b", type=float, default=0.660)
    args = parser.parse_args()

    audit_report(Path(args.report))
    for path in args.upstream:
        audit_report(Path(path))
    if args.ckpt:
        audit_checkpoint(Path(args.ckpt), args.init_a, args.init_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

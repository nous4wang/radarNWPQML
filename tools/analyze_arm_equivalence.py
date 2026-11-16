"""Is the ``--no-nwp`` arm an ablation, or a replicate?

The sweep runs four arms.  Arm 1 is ``qlfn2 --no-nwp``: the harness zeroes
the NWP tensor, so a model that reads the NWP would have to change.  The
bit-level probe ``scripts/check_nwp_path.py`` shows that qlfn2 does *not*
read the NWP at all -- its output is bit-identical for random, scaled,
zeroed and absent NWP.

If the probe is right, then arm 0 and arm 1 are the *same computation run
twice*, and their curves should differ only by GPU nondeterminism: small
differences, no consistent sign, and above all the same shape -- the same
epochs collapsing and the same epochs peaking.

This script makes that falsifiable against the numbers that are already on
disk (``artifacts/qlfn2_curve/curve.json``).  It reports, per seed:

* the per-epoch difference of CSI@2.0 between arm 0 and arm 1;
* the same for arm 0 vs arm 2 (the advection arm), which is the positive
  control -- a mechanism that is really switched off must show a signal
  much larger than the nondeterminism floor;
* the Pearson correlation between the two curves' epoch series.

Run::

    python tools/analyze_arm_equivalence.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "artifacts" / "qlfn2_curve" / "curve.json"
OUT = ROOT / "artifacts" / "qlfn2_curve" / "arm_equivalence.json"


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return float("nan")
    return num / (da * db)


def series(arm: dict, key: str = "csi_2.0") -> list[float]:
    """The epoch checkpoints only -- ``last.pt`` is not an epoch."""
    return [float(p[key]) for p in arm["curve"] if not p.get("is_last")]


def epochs(arm: dict) -> list[int]:
    out = []
    for k, p in enumerate(arm["curve"]):
        if p.get("is_last"):
            continue
        out.append(int(p["epoch"]) if p.get("epoch") is not None else k)
    return out


def main() -> int:
    data = json.loads(CURVE.read_text(encoding="utf-8"))
    arms = data["arms"]
    result = {
        "source": str(CURVE),
        "geometry": data.get("geometry"),
        "n_samples": data.get("n_samples"),
        "seeds": {},
    }

    print(f"geometry  : {data.get('geometry')}")
    print(f"n_samples : {data.get('n_samples')}")
    print()
    header = ("seed", "contrast", "mean|d|", "max|d|", "sign", "r(arm0,armX)")
    print(f"{header[0]:<6}{header[1]:<22}{header[2]:>10}{header[3]:>10}"
          f"{header[4]:>10}{header[5]:>14}")
    print("-" * 72)

    for seed, label in (("10", "seed10"), ("11", "seed11")):
        a0 = arms["qlfn2"][seed]
        s0 = series(a0)
        entry = {"epochs": epochs(a0)}

        contrasts = {
            "arm0 - arm1 (no-nwp)": arms["qlfn2_no_nwp"][seed],
            "arm0 - arm2 (no-motion)": arms["qlfn2_no_motion"][seed],
            "arm0 - arm3 (upstream)": arms["physical_qpe"][seed],
        }
        for name, arm in contrasts.items():
            sX = series(arm)
            diffs = [x - y for x, y in zip(s0, sX)]
            mad = sum(abs(d) for d in diffs) / len(diffs)
            mx = max(abs(d) for d in diffs)
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
            sign = f"{pos}+/{neg}-"
            r = pearson(s0, sX)
            print(f"{label:<6}{name:<22}{mad:>10.4f}{mx:>10.4f}{sign:>10}{r:>14.5f}")
            key = name.replace(" ", "_").replace("(", "").replace(")", "")
            entry[key] = {
                "diffs_csi2": diffs,
                "mean_abs": mad,
                "max_abs": mx,
                "sign": sign,
                "pearson_r_with_arm0": r,
            }
        result["seeds"][label] = entry
        print()

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written   : {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Report the epoch curve under three statistics, not one.

Why: the curve is not smooth.  In the first curve experiment (F29) the
qlfn2 seed-11 arm went 0.1286 (ep00) -> 0.0290 (ep01) -> 0.1783 (ep02) ->
0.1569 (ep03), and the control arm bounced between 0.0121 and 0.0930 from
one epoch to the next.  A peak read off a curve like that is a maximum over
ten noisy draws: it is biased upward, and how far upward depends on the
noise.  Reporting only the peak would therefore overstate the model.

The three statistics answer three different questions:

    peak     the best epoch -- this is what the upstream protocol reports,
             and it is aligned with the upstream's own numbers.  It is the
             most favourable number for every arm at the same time, which
             is what makes it comparable.
    median   the middle epoch -- a robust location estimate of the curve.
             It answers "how good is this arm typically" and it is the
             number to quote when the peak is being doubted.
    last     the final checkpoint -- the collapsed end.  Reported because
             it is what the (wrong) first sweep measured, so the gap
             between peak and last is the evidence that the epoch has to be
             swept at all.

All three are printed side by side, with the spread (max - min over the
epochs) so the noise level is visible next to the numbers it produced.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = {
    "qlfn2": "qlfn2 (full)",
    "qlfn2_no_nwp": "qlfn2 --no-nwp",
    "qlfn2_no_motion": "qlfn2 motion=none",
    "physical_qpe": "physical_qpe (control)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    return ap.parse_args(argv)


def load(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("*/*/curve/*_epoch_curve.json")):
        arm = path.parents[2].name
        seed = int(path.parents[1].name.replace("seed", ""))
        runs.setdefault(arm, {})[seed] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def stats_of(data: dict) -> dict:
    values = [r["csi_2.0"] for r in data["rows"] if not r.get("is_last")]
    last = next((r["csi_2.0"] for r in data["rows"] if r.get("is_last")), None)
    return {
        "peak": data["peak_csi2"],
        "peak_epoch": data["peak_epoch"],
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
        "last": last,
        "n_epochs": len(values),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    runs = load(root)
    if not runs:
        print(f"no curve summaries under {root}")
        return 1

    out: list[str] = []
    out.append("# 逐轮曲线的三种统计量（CSI@2.0，64/64 口径）")
    out.append("")
    out.append("曲线不光滑 ⇒ **取峰值是有偏的统计量**（在十个噪声样本上取最大）。"
               "所以峰值、中位数、末轮三个数一起报，不只报一个。")
    out.append("")

    widths = [24, 8, 10, 10, 10, 10, 10, 10]
    out.append("| " + " | ".join(p.ljust(w) for p, w in zip(
        ["臂", "seed", "峰值", "中位数", "均值", "末轮", "极差", "峰值轮"], widths)) + " |")
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    payload: dict[str, dict] = {}
    for arm, label in ARMS.items():
        if arm not in runs:
            continue
        payload[arm] = {}
        for seed in sorted(runs[arm]):
            st = stats_of(runs[arm][seed])
            payload[arm][str(seed)] = st
            out.append("| " + " | ".join(p.ljust(w) for p, w in zip(
                [label, str(seed), f"{st['peak']:.4f}", f"{st['median']:.4f}",
                 f"{st['mean']:.4f}",
                 f"{st['last']:.4f}" if st["last"] is not None else "-",
                 f"{st['spread']:.4f}", f"ep{st['peak_epoch']:02d}"], widths)) + " |")
    out.append("")

    out.append("**怎么读**")
    out.append("")
    out.append("- **峰值**对齐上游协议（上游也报 epoch 曲线的最好一轮），"
               "但它在噪声下偏乐观；")
    out.append("- **中位数**是稳健的位置估计，回答\"这个臂典型能到多少\"；")
    out.append("- **极差**= 轮次间最大波动，直接说明这条曲线有多可信；")
    out.append("- **末轮**是塌陷端，也是第一轮 sweep 误测的那个数。")
    out.append("")

    text = "\n".join(out) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
        print(f"markdown written to {args.markdown}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"json written to {args.json}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

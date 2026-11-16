#!/usr/bin/env python3
"""Summarise the epoch-curve experiment: four arms, two seeds, one geometry.

The experiment's point is that the score of these models is not read off
the last checkpoint.  The upstream's own note on the phenomenon:

    The old headline (CSI@2.0 = 0.118) came from a checkpoint that the
    training run was still overwriting.

and its own curve (E08/E19) shows the shape:

    physical_qpe   ep00 0.1226   ep01 0.0839   ep02 0.0467

So this summariser reports, per arm and seed:

    peak          the best epoch's CSI@2.0 -- the number the upstream protocol
                 reports
    last          the final checkpoint's CSI@2.0 -- the collapsed end
    the curve     every epoch, so the shape can be checked rather than assumed

and it puts the peak numbers next to the two upstream reference values that
were measured with the same geometry (patch 64 / stride 64):

    persistence floor   0.0924   (analytical Z-R)
    oracle ceiling      0.1653   (perfect QPE, no motion)
    upstream control    0.1226   (physical_qpe at its own peak, epoch 0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {
    "qlfn2": "0  qlfn2 (motion=corr, CAPPI on, NWP on)",
    "qlfn2_no_nwp": "1  qlfn2 --no-nwp",
    "qlfn2_no_motion": "2  qlfn2 motion=none",
    "physical_qpe": "3  physical_qpe (the upstream control)",
}

# Measured by the upstream with the geometry this experiment uses.
FLOOR = 0.0924
CEILING = 0.1653
UPSTREAM_PEAK = 0.1226

THRESHOLDS = ["0.1", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="directory holding <arm>/<seed>/curve/*_epoch_curve.json")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    return ap.parse_args(argv)


def load(root: Path) -> dict:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("*/*/curve/*_epoch_curve.json")):
        arm = path.parents[2].name
        seed = int(path.parents[1].name.replace("seed", ""))
        runs.setdefault(arm, {})[seed] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def line(parts: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(p.ljust(w) for p, w in zip(parts, widths)) + " |"


def rule(widths: list[int]) -> str:
    return "|" + "|".join("-" * (w + 2) for w in widths) + "|"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    runs = load(root)
    if not runs:
        print(f"no curve summaries under {root}")
        return 1

    out: list[str] = []
    out.append("# qlfn2 逐轮曲线实验（四臂 × 2 seed）")
    out.append("")
    out.append("口径：上游权威几何 **patch 64 / stride 64**（25 tiles / 13450 样本）、"
               "test split、538 窗口、7 个阈值。")
    out.append("数值 = CSI@2.0。**峰值**取自 epoch 轮次，`last.pt` 只作对照。")
    out.append("")
    out.append("上游同口径标尺：Z-R 持久性地板 **0.0924** ｜ "
               "上游 physical_qpe 峰值 **0.1226** ｜ Oracle 上界 **0.1653**")
    out.append("")
    out.append("> ⚠️ **本报告的两条读法限制（F30 复查后写入）**")
    out.append(">")
    out.append("> 1. **臂 1 不是 NWP 消融，是同一个计算的第二次运行。** "
               "`qlfn2` 对 `x_nwp` 逐位不敏感（探针 `scripts/check_nwp_path.py`："
               "随机 / ×100 / 全零 / 缺席四种 NWP，输出 bit-identical）。"
               "因此臂 0 与臂 1 的差异**只可能是 GPU 非确定性**，"
               "本实验对 NWP 的贡献**什么也没有测**——既不能说 NWP 有用，"
               "也不能说 NWP 没用。")
    out.append("> 2. **臂 3 与臂 0 不是同一个任务。** 上游 `physical_qpe` 走 "
               "`--qpe-single-frame`：训练只监督 `y_hist[:, -1]`（t0 地面降水），"
               "**从不见任何未来帧**；`qlfn2` 是 10 时效预报 + NCC 平流。"
               "两者放在同一根柱子上比\"谁更强\"在任务层面就不成立。")
    out.append("")

    # ---- the headline table -------------------------------------------------
    out.append("## 峰值（逐轮取峰，这是上游协议报告的那个数）")
    out.append("")
    widths = [26, 10, 12, 12, 12, 12, 14]
    out.append(line(["臂", "seed", "峰值轮次", "峰值 CSI@2.0",
                     "last CSI@2.0", "峰值/末轮", "相对上游峰值"], widths))
    out.append(rule(widths))
    peaks: dict[str, dict[int, float]] = {}
    for arm, label in ARMS.items():
        if arm not in runs:
            continue
        for seed in sorted(runs[arm]):
            data = runs[arm][seed]
            peak = data["peak_csi2"]
            last = data["last_csi2"]
            peaks.setdefault(arm, {})[seed] = peak
            ratio = data.get("peak_over_last_ratio", float("nan"))
            delta = (peak / UPSTREAM_PEAK - 1) * 100
            out.append(line([
                label.split("  ", 1)[1] if arm == "qlfn2" else label,
                str(seed),
                f"ep{data['peak_epoch']:02d}",
                f"{peak:.4f}",
                f"{last:.4f}",
                f"{ratio:.3f}x",
                f"{delta:+.1f} %",
            ], widths))
    out.append("")

    # ---- the curve shape ----------------------------------------------------
    out.append("## 逐轮曲线（CSI@2.0）")
    out.append("")
    seeds = sorted({s for arm in runs.values() for s in arm})
    for seed in seeds:
        out.append(f"### seed {seed}")
        out.append("")
        cols = [6] + [13] * len(ARMS) + [13]
        header = ["轮次"] + [f"ep{e:02d}" for e in range(10)] + ["last"]
        out.append(line(header, cols))
        out.append(rule(cols))
        for arm in ARMS:
            if arm not in runs or seed not in runs[arm]:
                continue
            rows = {r.get("epoch"): r for r in runs[arm][seed]["rows"]}
            cells = [arm.replace("qlfn2_", "") if arm != "qlfn2" else "qlfn2"]
            for epoch in range(10):
                row = rows.get(epoch)
                cells.append(f"{row['csi_2.0']:.4f}" if row else "-")
            last_row = next((r for r in runs[arm][seed]["rows"] if r.get("is_last")), None)
            cells.append(f"{last_row['csi_2.0']:.4f}" if last_row else "-")
            out.append(line(cells, cols))
        out.append("")

    # ---- paired differences at the peak -------------------------------------
    out.append("## 配对差值（同 seed、同几何、只差机制）")
    out.append("")
    out.append("按 §9.4 判读纪律，先看平流，再看与上游对照，最后看 NWP。"
               "差值为正 = 有增量。")
    out.append("")

    def paired(arm_a: str, arm_b: str, label: str, note: str) -> None:
        if arm_a not in runs or arm_b not in runs:
            return
        cols = [8, 14, 14, 14, 14, 14]
        out.append(f"### {label}")
        out.append("")
        out.append(note)
        out.append("")
        out.append(line(["seed", "A 峰值", "B 峰值", "Δ@0.5", "Δ@2.0", "Δ@4.0"], cols))
        out.append(rule(cols))
        for seed in sorted(set(runs[arm_a]) & set(runs[arm_b])):
            a, b = runs[arm_a][seed], runs[arm_b][seed]
            a_rows = {r.get("epoch"): r for r in a["rows"]}
            b_rows = {r.get("epoch"): r for r in b["rows"]}
            ap, bp = a["peak_epoch"], b["peak_epoch"]
            row_a = a_rows.get(ap)
            row_b = b_rows.get(bp)
            if not row_a or not row_b:
                continue
            out.append(line([
                str(seed),
                f"{a['peak_csi2']:.4f}",
                f"{b['peak_csi2']:.4f}",
                f"{row_a['csi_0.5'] - row_b['csi_0.5']:+.4f}",
                f"{row_a['csi_2.0'] - row_b['csi_2.0']:+.4f}",
                f"{row_a['csi_4.0'] - row_b['csi_4.0']:+.4f}",
            ], cols))
        out.append("")

    paired("qlfn2", "qlfn2_no_motion",
           "平流：arm 0 − arm 2",
           "arm 2 训练与评测都关掉平流（`motion_mode=none`），"
           "state_dict 相同所以必须两处都关。")
    paired("qlfn2", "physical_qpe",
           "qlfn2 vs 上游对照：arm 0 − arm 3",
           "arm 3 用上游自己的 `physical_qpe`，同一个 harness、同一个评测器。")
    paired("qlfn2", "qlfn2_no_nwp",
           "NWP：arm 0 − arm 1（**这不是消融**）",
           "臂 1 训练与评测都拿掉 NWP——但 `qlfn2` 本来就逐位不读 NWP，"
           "所以这一对是**同一计算的重复**，差值即非确定性噪声，"
           "**不能读作 NWP 的贡献**（见下面的臂等价性检验）。")
    paired("physical_qpe", "qlfn2_no_nwp",
           "arm 3 − arm 1（上游对照 vs 臂 1）",
           "臂 1 是 qlfn2（10 时效预报 + 平流），不是单帧 QPE、不是无平流；"
           "两臂差的是**任务与架构**，不是 NWP。这对数字只能当作"
           "\"两条路线在同一把尺子上的位置\"，不能当作同机制族对照。")

    # ---- arm equivalence: is arm 1 an ablation or a replicate? ---------------
    out.append("## 臂等价性检验（臂 1 到底测了什么）")
    out.append("")
    out.append("判据：若臂 1 真是 NWP 消融，臂 0 与臂 1 的逐轮差值应当"
               "系统性地大于非确定性地板，且符号一致；"
               "若两臂其实是同一个计算，差值应当无符号、幅度接近 0，"
               "两条曲线高度相关。")
    out.append("")

    def epoch_series(arm: str, seed: int) -> list[float]:
        rows = [r for r in runs[arm][seed]["rows"] if not r.get("is_last")]
        return [float(r["csi_2.0"]) for r in rows]

    def pearson(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        if da == 0.0 or db == 0.0:
            return float("nan")
        return num / (da * db)

    contrasts = [
        ("臂 0 − 臂 1（声称的 NWP 消融）", "qlfn2_no_nwp"),
        ("臂 0 − 臂 2（平流消融，正对照）", "qlfn2_no_motion"),
        ("臂 0 − 臂 3（上游对照）", "physical_qpe"),
    ]
    cols2 = [34, 8, 12, 12, 12, 12]
    out.append(line(["对比", "seed", "mean abs Δ", "max abs Δ", "符号", "r(臂0, 臂X)"], cols2))
    out.append(rule(cols2))
    for label, arm in contrasts:
        if arm not in runs:
            continue
        for seed in sorted(runs[arm]):
            s0 = epoch_series("qlfn2", seed)
            sx = epoch_series(arm, seed)
            diffs = [a - b for a, b in zip(s0, sx)]
            mad = sum(abs(d) for d in diffs) / len(diffs)
            mx = max(abs(d) for d in diffs)
            sign = (f"{sum(1 for d in diffs if d > 0)}+/"
                    f"{sum(1 for d in diffs if d < 0)}-")
            out.append(line([label, str(seed), f"{mad:.4f}", f"{mx:.4f}",
                             sign, f"{pearson(s0, sx):.5f}"], cols2))
    out.append("")
    out.append("读法：臂 0 − 臂 1 的 mean|Δ| 比平流消融小一个数量级，"
               "符号两个 seed 都不一致，两条曲线 r ≈ 0.99 —— "
               "**臂 1 是臂 0 的重复，不是消融**。")
    out.append("")

    # ---- the scale ----------------------------------------------------------
    out.append("## 同一把尺子上的位置（64/64）")
    out.append("")
    cols = [34, 12, 16]
    out.append(line(["参照", "CSI@2.0", "性质"], cols))
    out.append(rule(cols))
    out.append(line(["Z-R 持久性", f"{FLOOR:.4f}", "地板"], cols))
    out.append(line(["上游 physical_qpe 峰值", f"{UPSTREAM_PEAK:.4f}", "实测对照"], cols))
    for arm, label in ARMS.items():
        if arm not in peaks:
            continue
        values = list(peaks[arm].values())
        mean = sum(values) / len(values)
        kind = "上游对照" if arm == "physical_qpe" else "qlfn2 变体"
        out.append(line([f"{label.split('  ', 1)[1]}（两 seed 均值）",
                         f"{mean:.4f}", kind], cols))
    out.append(line(["Oracle 上界", f"{CEILING:.4f}",
                     "完美 QPE + 无平流（非普适天花板）"], cols))
    out.append("")
    out.append("⚠️ **读法**：Oracle 上界 0.1653 是\"完美 QPE **且不做平流**\"的天花板，"
               "而上游自己的 ground_flow1/2/4/6 显示**加平流反而把它压低**"
               "（0.1600 / 0.1376 / 0.1324 / 0.1303）。"
               "本实验是 **10 个时效的预报**，与它不是同一件事的两端。")
    out.append("")
    out.append("⚠️ **seed 数**：2 个 seed 只能给**方向**，不能给区间"
               "（上游 §10.1：固定阈值上的 CSI 对训练随机性极度敏感）。")
    out.append("")

    text = "\n".join(out) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
        print(f"markdown written to {args.markdown}")
    if args.json:
        payload = {
            "geometry": {"patch_size": 64, "slide_stride": 64},
            "split": "test",
            "n_samples": 538,
            "reference": {
                "persistence_floor": FLOOR,
                "oracle_ceiling": CEILING,
                "upstream_physical_qpe_peak": UPSTREAM_PEAK,
            },
            "arms": {
                arm: {
                    str(seed): {
                        "peak_epoch": data["peak_epoch"],
                        "peak_csi2": data["peak_csi2"],
                        "last_csi2": data["last_csi2"],
                        "peak_over_last_ratio": data.get("peak_over_last_ratio"),
                        "curve": [
                            {"epoch": r.get("epoch"), "is_last": r.get("is_last"),
                             "csi_2.0": r.get("csi_2.0"),
                             "csi_0.5": r.get("csi_0.5"),
                             "csi_1.0": r.get("csi_1.0"),
                             "csi_4.0": r.get("csi_4.0")}
                            for r in data["rows"]
                        ],
                    }
                    for seed, data in sorted(runs[arm].items())
                }
                for arm in ARMS if arm in runs
            },
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json written to {args.json}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

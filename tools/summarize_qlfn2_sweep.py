#!/usr/bin/env python3
"""Summarise the qlfn2 four-arm sweep into the table the control room needs.

The sweep runs four arms under one protocol, one job per arm and seed:

    0  qlfn2               motion=corr, CAPPI on, NWP on
    1  qlfn2_no_nwp        the NWP withheld
    2  qlfn2_no_motion     the advection switched off
    3  physical_qpe        the upstream control

The number that decides the mechanism is not any single arm's score, it is
the **paired** difference between two arms that were trained on the same
seed with the same budget and differ only in the mechanism.  So the summary
is built around paired contrasts:

    arm 2 - arm 0   the advection
    arm 0 - arm 3   qlfn2 against the upstream control
    arm 1 - arm 0   the NWP

The upstream's own warning applies to every one of them: the CSI at a fixed
threshold is extremely sensitive to the training draw, and with two seeds a
contrast can show a direction but not an interval.  The summary therefore
prints every seed's difference and says so, rather than reporting a mean
as if it were a measurement.

Usage:
    python tools/summarize_qlfn2_sweep.py --root runs/qlfn2_sweep
    python tools/summarize_qlfn2_sweep.py --root runs/qlfn2_sweep \\
        --markdown artifacts/qlfn2_sweep.md --json artifacts/qlfn2_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: The arm directory names, in the order the control room lists them.
ARMS: tuple[tuple[int, str, str], ...] = (
    (0, "qlfn2", "qlfn2 (motion=corr, CAPPI on, NWP on)"),
    (1, "qlfn2_no_nwp", "qlfn2 without the NWP"),
    (2, "qlfn2_no_motion", "qlfn2 without the advection"),
    (3, "physical_qpe", "physical_qpe (the upstream control)"),
)

#: The contrasts, in the order the control room says to read them.
CONTRASTS: tuple[tuple[str, int, int, str], ...] = (
    ("advection", 2, 0, "arm 2 - arm 0: does the advection carry anything"),
    ("qlfn2 vs the upstream control", 0, 3, "arm 0 - arm 3: does qlfn2 improve on physical_qpe"),
    ("the NWP", 1, 0, "arm 1 - arm 0: does the NWP carry anything"),
)

#: The thresholds the benchmark's tables are built around.  2.0 mm per
#: 6-minute accumulation is 20 mm/h, the short-duration heavy-rain line.
THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


def _read_report(path: Path) -> dict[str, Any] | None:
    """Load one evaluation report, or return None when it is missing."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _thresholds(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The threshold table of a report, under either of its key names."""
    table = report.get("thresholds") or report.get("per_threshold") or {}
    return {str(k): v for k, v in table.items() if isinstance(v, dict)}


def _value(report: dict[str, Any], threshold: float, key: str) -> float | None:
    """One number of one threshold, or None when it is absent or NaN."""
    table = _thresholds(report)
    entry = table.get(f"{threshold}")
    if entry is None:
        entry = table.get(str(threshold))
    if not isinstance(entry, dict):
        return None
    raw = entry.get(key)
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _ablation_mode(report: dict[str, Any]) -> str | None:
    """The configuration the scorer recorded for one report."""
    ablation = report.get("ablation")
    if not isinstance(ablation, dict):
        return None
    mode = ablation.get("mode")
    return str(mode) if mode is not None else None


def _lead_csi(report: dict[str, Any], threshold: float) -> dict[str, float | None]:
    """The CSI of one report at one threshold, for every lead."""
    table = report.get("lead_csi") or {}
    out: dict[str, float | None] = {}
    for lead, per_threshold in table.items():
        if not isinstance(per_threshold, dict):
            continue
        raw = per_threshold.get(f"{threshold}")
        if raw is None:
            raw = per_threshold.get(str(threshold))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isnan(value):
            value = None
        out[str(lead)] = value
    return out


def _fmt(value: float | None, width: int = 7, places: int = 4) -> str:
    """Format one number for the tables, keeping a missing value visible."""
    if value is None:
        return " " * (width - 2) + "na"
    return f"{value:{width}.{places}f}"


def _seeds(root: Path) -> list[int]:
    """The seeds that at least one arm has a report for."""
    found: set[int] = set()
    for _, name, _ in ARMS:
        arm_dir = root / name
        if not arm_dir.is_dir():
            continue
        for child in arm_dir.iterdir():
            if child.is_dir() and child.name.startswith("seed"):
                try:
                    found.add(int(child.name[4:]))
                except ValueError:
                    continue
    return sorted(found)


def collect(root: Path) -> dict[str, Any]:
    """Read every report under the root and assemble the tables."""
    seeds = _seeds(root)
    reports: dict[int, dict[int, dict[str, Any]]] = {}
    missing: list[str] = []

    for seed in seeds:
        per_arm: dict[int, dict[str, Any]] = {}
        for index, name, _ in ARMS:
            path = root / name / f"seed{seed}" / "eval_test.json"
            report = _read_report(path)
            if report is None:
                missing.append(f"{name}/seed{seed}")
            else:
                per_arm[index] = report
        reports[seed] = per_arm

    return {"seeds": seeds, "reports": reports, "missing": missing}


def build_tables(data: dict[str, Any]) -> dict[str, Any]:
    """Build the per-arm table and the paired contrasts."""
    seeds: list[int] = data["seeds"]
    reports: dict[int, dict[int, dict[str, Any]]] = data["reports"]

    per_arm: dict[str, Any] = {}
    for index, name, _ in ARMS:
        rows = []
        for seed in seeds:
            report = reports[seed].get(index)
            if report is None:
                rows.append({"seed": seed, "missing": True})
                continue
            rows.append({
                "seed": seed,
                "missing": False,
                "n_samples": report.get("n_samples"),
                "ablation_mode": _ablation_mode(report),
                "csi": {str(t): _value(report, t, "csi") for t in THRESHOLDS},
                "pod": {str(t): _value(report, t, "pod") for t in THRESHOLDS},
                "far": {str(t): _value(report, t, "far") for t in THRESHOLDS},
                "bias": {str(t): _value(report, t, "bias") for t in THRESHOLDS},
                "fss": {str(t): _value(report, t, "fss") for t in THRESHOLDS},
                "lead_csi": {str(t): _lead_csi(report, t) for t in THRESHOLDS},
            })
        per_arm[name] = {"arm": index, "rows": rows}

    contrasts: dict[str, Any] = {}
    for label, low, high, description in CONTRASTS:
        low_name = ARMS[low][1]
        high_name = ARMS[high][1]
        entries = []
        for seed in seeds:
            low_report = reports[seed].get(low)
            high_report = reports[seed].get(high)
            if low_report is None or high_report is None:
                entries.append({"seed": seed, "missing": True})
                continue
            entry: dict[str, Any] = {"seed": seed, "missing": False}
            for threshold in THRESHOLDS:
                a = _value(low_report, threshold, "csi")
                b = _value(high_report, threshold, "csi")
                entry[f"csi@{threshold}"] = {
                    low_name: a,
                    high_name: b,
                    "difference": (a - b) if (a is not None and b is not None) else None,
                }
            entries.append(entry)
        contrasts[label] = {
            "description": description,
            "low_arm": low,
            "high_arm": high,
            "low_name": low_name,
            "high_name": high_name,
            "entries": entries,
        }

    return {"per_arm": per_arm, "contrasts": contrasts}


def build_lead_profile(tables: dict[str, Any], threshold: float) -> dict[str, Any]:
    """The lead-time profile of the CSI, per arm and as a paired contrast.

    The lead profile is the direct test of whether a mechanism is doing
    nowcasting.  A field that is copied forward decays with the lead, and
    a field that is advected does not decay as fast, **if the advection is
    right**.  So the difference between the arm with the mechanism and the
    arm without it is not one number but a curve, and a curve that grows
    with the lead says something a single number cannot: it says the
    mechanism is adding lead-dependent skill rather than a constant offset.
    """
    per_arm = tables["per_arm"]
    threshold_key = str(threshold)

    leads: set[str] = set()
    for _, name, _ in ARMS:
        for row in per_arm[name]["rows"]:
            if row["missing"]:
                continue
            leads.update(row["lead_csi"][threshold_key].keys())
    ordered = sorted(leads, key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)

    profile: dict[str, Any] = {"threshold": threshold, "leads": ordered,
                                "arms": {}, "paired": {}}
    for _, name, _ in ARMS:
        columns: dict[str, dict[str, Any]] = {}
        for lead in ordered:
            values = []
            for row in per_arm[name]["rows"]:
                if row["missing"]:
                    continue
                value = row["lead_csi"][threshold_key].get(lead)
                if value is not None:
                    values.append(value)
            columns[lead] = {
                "values": values,
                "mean": (sum(values) / len(values)) if values else None,
            }
        profile["arms"][name] = columns

    # the paired difference, seed by seed, so that the mean of the
    # differences is the mean of the paired differences and not the
    # difference of the two means
    for label, low, high, _ in CONTRASTS:
        low_name = ARMS[low][1]
        high_name = ARMS[high][1]
        columns = {}
        for lead in ordered:
            values = []
            for left, right in zip(per_arm[low_name]["rows"],
                                   per_arm[high_name]["rows"]):
                if left["missing"] or right["missing"]:
                    continue
                a = left["lead_csi"][threshold_key].get(lead)
                b = right["lead_csi"][threshold_key].get(lead)
                if a is not None and b is not None:
                    values.append(a - b)
            columns[lead] = {
                "values": values,
                "mean": (sum(values) / len(values)) if values else None,
            }
        profile["paired"][label] = columns

    return profile


def render_markdown(data: dict[str, Any], tables: dict[str, Any],
                    profile: dict[str, Any]) -> str:
    """Render the summary as markdown."""
    seeds: list[int] = data["seeds"]
    lines: list[str] = []
    lines.append("# qlfn2 四臂扫描结果")
    lines.append("")
    lines.append(f"种子：{', '.join(str(s) for s in seeds) if seeds else '（无）'}")
    lines.append("")
    lines.append("> 上游 §10.1 警示：固定阈值上的 CSI 对训练随机性极度敏感，"
                 "两个种子只能给出**方向**，不能给出区间。")
    lines.append("")

    for index, name, description in ARMS:
        lines.append(f"## 臂 {index} · {name}")
        lines.append("")
        lines.append(description)
        lines.append("")
        header = "| seed | n_samples | ablation | " + " | ".join(
            f"CSI@{t}" for t in THRESHOLDS) + " | " + " | ".join(
            f"BIAS@{t}" for t in THRESHOLDS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (3 + 2 * len(THRESHOLDS)))
        for row in tables["per_arm"][name]["rows"]:
            if row["missing"]:
                lines.append(f"| {row['seed']} | *(缺失)* | |" + " |" * (2 * len(THRESHOLDS)))
                continue
            cells = [str(row["seed"]), str(row["n_samples"]),
                     str(row.get("ablation_mode") or "-")]
            cells += [_fmt(row["csi"][str(t)]) for t in THRESHOLDS]
            cells += [_fmt(row["bias"][str(t)]) for t in THRESHOLDS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 配对差值（同 seed、同预算、只差机制）")
    lines.append("")
    for label, _, _, description in CONTRASTS:
        contrast = tables["contrasts"][label]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(description)
        lines.append("")
        lines.append(f"| seed | " + " | ".join(
            f"CSI@{t} ({contrast['low_name']} − {contrast['high_name']})"
            for t in THRESHOLDS) + " |")
        lines.append("|" + "---|" * (1 + len(THRESHOLDS)))
        for entry in contrast["entries"]:
            if entry["missing"]:
                lines.append(f"| {entry['seed']} |" + " *(缺失)* |" * len(THRESHOLDS))
                continue
            cells = [str(entry["seed"])]
            for threshold in THRESHOLDS:
                diff = entry[f"csi@{threshold}"]["difference"]
                cells.append(_fmt(diff))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if data["missing"]:
        lines.append("## 缺失的报告")
        lines.append("")
        for item in data["missing"]:
            lines.append(f"- {item}")
        lines.append("")

    threshold = profile["threshold"]
    lines.append(f"## 逐时效 CSI@{threshold:g} 剖面（lead 0 = 第一个 6 min，lead 9 = 最后一个）")
    lines.append("")
    lines.append("拷贝前推的场随时效**衰减**，平流正确的场**衰减得慢**。"
                 "所以两个臂的差值是一条曲线而不是一个数："
                 "**差值随时效增长**才说明机制带来的是时效相关的技巧，"
                 "而不是一个常数偏移。")
    lines.append("")
    arm_names = [name for _, name, _ in ARMS]
    lines.append("| lead | " + " | ".join(arm_names) + " | " +
                 " | ".join(f"Δ ({label})" for label, _, _, _ in CONTRASTS) + " |")
    lines.append("|" + "---|" * (1 + len(arm_names) + len(CONTRASTS)))
    for lead in profile["leads"]:
        cells = [lead]
        for name in arm_names:
            cells.append(_fmt(profile["arms"][name][lead]["mean"]))
        for label, _, _, _ in CONTRASTS:
            cells.append(_fmt(profile["paired"][label][lead]["mean"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Read the reports and print (or write) the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/qlfn2_sweep",
                        help="the directory that holds the arm directories")
    parser.add_argument("--markdown", default=None,
                        help="write the markdown summary here")
    parser.add_argument("--json", default=None,
                        help="write the machine-readable summary here")
    parser.add_argument("--lead-threshold", type=float, default=2.0,
                        help="the threshold of the lead-time profile "
                             "(default: %(default)s)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"FATAL: no sweep directory at {root}", file=sys.stderr)
        return 5

    data = collect(root)
    tables = build_tables(data)
    profile = build_lead_profile(tables, args.lead_threshold)

    markdown = render_markdown(data, tables, profile)
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"markdown written to {out}")
    else:
        print(markdown)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"seeds": data["seeds"], "missing": data["missing"],
                        "tables": tables, "lead_profile": profile},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"json written to {out}")

    if data["missing"]:
        print(f"WARNING: {len(data['missing'])} reports are missing: "
              f"{', '.join(data['missing'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

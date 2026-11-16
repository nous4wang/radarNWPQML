"""Render the catalogue, the two score tables, and the evidence gate as text.

Kept separate from the data classes so that no module in ``qmlbench`` both
computes a verdict and formats it. This module receives already-computed
scores, findings and dispositions; it never derives a new number or a new
rule outcome. Output is Markdown or CSV, both plain text, so the generated
tables can be pasted into a paper draft or a spec diff.
"""
from __future__ import annotations

import csv
import io
from typing import Iterable

from .catalog import Catalog, GROUPS
from .gate import Claim, Finding, SEALED_DIRECTIONS
from .results import (
    BEST_TRAINED_CSI2, FLOOR_CSI2, ORACLE_CSI2, PYSTEPS_CSI2,
    ScoreRow, decidability, ranking,
)

#: A review is (claim, findings, disposition). Findings and the disposition
#: are produced by ``qmlbench.gate``; this module only formats them.
Review = tuple[Claim, list[Finding], str]


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """Format one optional number for a Markdown cell.

    Args:
        value: the number, or ``None`` for "not reported".
        digits: decimal places.

    Returns:
        A formatted string, or an em dash for ``None``.

    Raises:
        None.
    """
    return "—" if value is None else f"{value:.{digits}f}"


def master_table(rows: dict[str, ScoreRow], score_aware: dict[str, float],
                 catalog: Catalog) -> str:
    """Render the clean-split table: standard-loss next to score-aware.

    The two CSI@2.0 columns are deliberately kept side by side and are never
    merged into one ranking. A model present only in the score-aware matrix
    (``simvp_early_concat``) still gets a row.

    Args:
        rows: clean-split standard-loss rows from ``results.standard_table``.
        score_aware: authoritative ``CSI@2.0`` from ``RESULTS_FINAL.md`` §10.1.
        catalog: the baseline catalogue, for group and coverage columns.

    Returns:
        A Markdown table with a provenance note.

    Raises:
        None.
    """
    lines = [
        "| model | group | standard CSI@2.0 | score-aware CSI@2.0 | floor Δ | has robustness |",
        "|---|---|---|---|---|---|",
    ]
    ordered = ranking(rows)
    extra = sorted(set(score_aware) - {r.model for r in ordered})
    for row in ordered:
        lines.append(_master_row(row, score_aware.get(row.model), catalog))
    for name in extra:
        # A score-aware-only model has no standard-loss row; render a sparse
        # record rather than silently dropping it from the comparison.
        lines.append(_master_row(
            ScoreRow(model=name, table="standard"), score_aware.get(name), catalog
        ))
    lines.append("")
    lines.append(
        f"参照：persistence floor = {FLOOR_CSI2}，PySTEPS = {PYSTEPS_CSI2}，"
        f"Oracle（特权信息，不可部署）= {ORACLE_CSI2}，"
        f"最佳已训练深度模型 = {BEST_TRAINED_CSI2}（仍低于地板）。"
    )
    lines.append("两张 CSI@2.0 表口径不同，禁止合并排名：standard 来自 `*.eval.json`（n=538），"
                 "score-aware 来自 `RESULTS_FINAL.md` §10.1（n=538）。")
    return "\n".join(lines)


def _master_row(row: ScoreRow, sa: float | None, catalog: Catalog) -> str:
    """Render one master-table row; all numbers arrive pre-computed."""
    spec = catalog.get(row.model)
    group = spec.group if spec else "(unclassified)"
    rob = "yes" if spec and spec.has_robustness else "no"
    return (
        f"| `{row.model}` | {group} | {_fmt(row.csi2, 6)} | "
        f"{_fmt(sa, 3)} | {_fmt(row.floor_delta, 6)} | {rob} |"
    )


def robustness_report(rows: dict[str, ScoreRow], catalog: Catalog) -> str:
    """Render the missing-frame robustness split on its own terms.

    This split (n=962) contains cases completely disjoint from the clean
    split (n=538) with different rainfall intensity
    (``TGRS_EXPERIMENTS.md:468-469``). It therefore has **no** floor-delta
    column and **no** score-aware column: neither would be a valid comparison.

    Args:
        rows: robustness rows from ``results.robustness_table``.
        catalog: the baseline catalogue, for the group column.

    Returns:
        A Markdown table with a disjoint-split warning.

    Raises:
        None.
    """
    lines = [
        "# 断帧鲁棒划分（robustness，n=962）",
        "",
        "| model | group | CSI@2.0 | n_samples |",
        "|---|---|---|---|",
    ]
    for row in ranking(rows):
        spec = catalog.get(row.model)
        group = spec.group if spec else "(unclassified)"
        lines.append(
            f"| `{row.model}` | {group} | {_fmt(row.csi2, 6)} | "
            f"{row.n_samples if row.n_samples is not None else '—'} |"
        )
    lines.append("")
    lines.append("数据来源：`*.robustness.json`。该划分与洁净划分（n=538）的 case 完全不相交、"
                 "降水强度不同，**两组分数禁止直接相减**；本表不提供 floor Δ，也不与 "
                 "`RESULTS_FINAL.md` §10.1 的 score-aware 表并列。")
    return "\n".join(lines)


def coverage_table(catalog: Catalog) -> str:
    """Show which baselines exist and which have recorded evidence.

    Args:
        catalog: the baseline catalogue.

    Returns:
        A Markdown coverage table plus the blueprint pool listing.

    Raises:
        None.
    """
    lines = ["| group | count | with eval | with robustness | with score-aware |",
             "|---|---|---|---|---|"]
    for group, label in GROUPS.items():
        specs = catalog.by_group(group)
        lines.append(
            f"| {group} ({label}) | {len(specs)} | "
            f"{sum(s.has_eval for s in specs)} | "
            f"{sum(s.has_robustness for s in specs)} | "
            f"{sum(s.has_score_aware for s in specs)} |"
        )
    lines.append("")
    lines.append(f"合计 {len(catalog.all())} 个基线；蓝图 §6.1 的 14 个已全部接入："
                 + ", ".join(f"`{s.name}`" for s in catalog.blueprint_pool()))
    return "\n".join(lines)


def baseline_csv(catalog: Catalog) -> str:
    """Serialise the full baseline catalogue as CSV.

    Args:
        catalog: the baseline catalogue.

    Returns:
        CSV text with a header row, one line per baseline.

    Raises:
        None.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([
        "name", "group", "paper", "venue", "license", "blueprint_alias",
        "adapter", "adapter_class", "config", "is_learnable",
        "has_eval", "has_robustness", "has_score_aware",
    ])
    for spec in catalog.all():
        writer.writerow([
            spec.name, spec.group, spec.paper, spec.venue, spec.license,
            spec.blueprint_alias or "", spec.adapter or "", spec.adapter_class or "",
            spec.config or "", int(spec.is_learnable), int(spec.has_eval),
            int(spec.has_robustness), int(spec.has_score_aware),
        ])
    return buf.getvalue()


def decidability_report() -> str:
    """Render the pre-computed undecidability diagnostic.

    Args:
        None.

    Returns:
        A short Markdown section quoting ``results.decidability()``.

    Raises:
        None.
    """
    d = decidability()
    return "\n".join([
        "## 可判定性（MASTER §11.9 P1）",
        "",
        f"- 地板 (persistence, 无参数)：`{d['floor']}`",
        f"- Oracle (完美 QPE + 不动, 特权信息)：`{d['oracle']}`",
        f"- 可用空间：`{d['headroom']}`",
        f"- 10 seed 单 seed 极差：`{d['single_seed_range']}` = 可用空间的 **{d['range_over_headroom']}×**",
        f"- 峰值端离散：`{d['seed_dispersion']}×`",
        f"- 最佳已训练深度模型：`{d['best_trained']}`，是否超过地板：**{d['best_trained_beats_floor']}**",
        f"- 判定：**{d['verdict']}**",
    ])


def gate_report(reviews: Iterable[Review]) -> str:
    """Render gate reviews side by side, without computing any of them.

    Args:
        reviews: ``(claim, findings, disposition)`` tuples produced by the
            caller via ``gate.check`` and ``gate.verdict``.

    Returns:
        A Markdown report with one section per claim and the sealed index.

    Raises:
        None.
    """
    lines = ["# 候选方法准入审查（§11.9 P1–P6）", ""]
    for claim, findings, disposition in reviews:
        lines.append(f"## `{claim.name}` → **{disposition}**")
        lines.append("")
        lines.append(f"机制：{claim.mechanism}；sealed_direction = `{claim.sealed_direction}`")
        lines.append("")
        lines.append("| rule | status | finding |")
        lines.append("|---|---|---|")
        for finding in findings:
            message = finding.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {finding.rule} | {finding.status} | {message} |")
        lines.append("")
    lines.append("## 已封口方向索引（MASTER §9 / §10.2）")
    lines.append("")
    for key, reason in SEALED_DIRECTIONS.items():
        lines.append(f"- `{key}`：{reason}")
    return "\n".join(lines)

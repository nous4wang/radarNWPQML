"""Executable form of the upstream evidence gate (``MASTER.md`` §11.9 P1-P6).

The benchmark concluded that the dominant failure mode in this project is
*not* a missing architecture but a missing standard of evidence: the spread
across ten training seeds is 2.29x the whole available headroom. Any claimed
improvement therefore has to pass an eligibility gate before it is allowed
into a comparison table.

This module turns that written protocol into a checkable object. It is
deliberately conservative: an absent field is a FAIL, never a pass. The gate
reports facts and rule outcomes; it never decides the science for the user.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .results import HEADROOM, SEED_DISPERSION, SEED_RANGE

#: Directions sealed by ``MASTER.md`` §9 / §10.2. Any candidate whose
#: mechanism belongs to one of these is rejected at the gate without compute.
#: Sources: MASTER.md:1067-1090 (discarded list), :1133-1191 (§10.2),
#: docs/radarNWP-QI-optimization.md:3-8 (E34 seal).
SEALED_DIRECTIONS: dict[str, str] = {
    "lagrangian_advection": "显式运动/半拉格朗日平流；E15 任何平流负增益、E16 门控 Lagrangian FAIL",
    "wind_drift_alignment": "风漂移/粒子下落时滞对齐；27 个 NWP 变量无 U/V 风矢量（条件缺失）",
    "hilbert_peps": "量子启发复数 Hilbert / PEPS 二维张量网络 / 相位对齐拓扑约束；E34 全方向封口",
    "complex_interference": "复数相消干涉残差块；表达力不超过实值卷积（MASTER §10.2）",
    "quantum_coupling": "Koopman/量子倾向耦合；E02 最确定的负面结果（0.0122 vs 0.0384）",
    "generative_flow": "生成式/扩散/流匹配新架构；E35 GitHub 新架构族整体舍弃",
    "frequency_intervention": "由 E33 的 61% 派生的频域/相位干预训练方案；信息缺口≠优化缺口",
    "amplitude_calibration": "幅度/分布/Z-R 标定族；E17/E18/E21/E22/E27 五连负",
    "cross_attention_fusion": "复杂 NWP 融合（xattn 掉 48%）；E03 越复杂越差",
    "ensemble_variants": "集成派生变体；E36 5+5 代理负期望",
    "ordinal_head": "序贯概率/分级测度头；E04 头本身未证实增益",
    "soft_fss": "多尺度软 FSS 损失；E16 CSI@0.5 显著为负",
}

#: §11.9 P2/P3/P5 anchor values a batch must reproduce or exceed.
#: Sources: ``MASTER.md:1484-1499`` (§11.9 P1-P6 text), ``E13-ten-seed-verdict.md:39-73``
#: (5/10 vs 4/10, Fisher one-sided p=0.5, 9.59x peak dispersion),
#: ``MASTER.md:1139-1150`` / ``E10`` (1.78x collapse-end spread),
#: ``MASTER.md:1489`` (single-seed range 0.1671, headroom 0.0729).
BATCH_ANCHORS: dict[str, Any] = {
    "qpe_0_pass_rate": 0.5,          # E13: 5/10 at ep00
    "base_concat_pass_rate": 0.4,    # E13: 4/10 at ep00
    "fisher_one_sided_p": 0.5,       # E13: Fisher one-sided p = 0.5
    "peak_dispersion": SEED_DISPERSION,   # E13: 9.59x
    "epoch": "ep00",                 # §11.9 P3: pass rates bind to ep00
    "collapse_end_ratio": 1.78,      # E10 / §11.9 P5
    "peak_end_ratio": 9.6,           # §11.9 P5
    "single_seed_range": SEED_RANGE, # MASTER.md:1489
    "headroom": HEADROOM,            # E15: 0.1653 - 0.0924
}

PASS, FAIL, REJECT, WARN = "PASS", "FAIL", "REJECT", "WARN"


@dataclass(frozen=True)
class Claim:
    """Everything a candidate method must report about one claimed improvement.

    ``None`` means "not reported". The gate treats missing evidence as FAIL,
    because §11.9 P4 requires all four items together.
    """

    name: str
    mechanism: str = "unspecified"
    #: A sealed id from SEALED_DIRECTIONS, or None.
    sealed_direction: str | None = None
    #: §11.2 Gate 1: does the module introduce a genuinely new observation
    #: channel or a verifiable causal prior?
    new_observation_source: bool | None = None
    #: §11.2 Gate 1: is the control single-variable with real-channel parity?
    single_variable_control: bool | None = None
    #: P1: a single peak value, and the number of independent training seeds.
    single_peak_csi2: float | None = None
    n_seeds: int = 0
    #: P2: did the same batch reproduce qpe_0 / base_concat anchors?
    #: §11.9 P2 also requires the four measured batch values, not a bare bool.
    batch_reproduced: bool | None = None
    batch_qpe0_pass_rate: float | None = None
    batch_base_concat_pass_rate: float | None = None
    batch_fisher_one_sided_p: float | None = None
    batch_peak_dispersion: float | None = None
    #: P3: the epoch the pass rate was measured at.
    epoch: str | None = None
    #: P4 four items.
    pass_rate: float | None = None
    baseline_pass_rate: float | None = None
    paired_seed_diffs: tuple[float, ...] | None = None
    two_level_bootstrap_ci: tuple[float, float] | None = None
    #: P5: observed gain and the two dispersion ratios.
    gain: float | None = None
    collapse_end_ratio: float | None = None
    peak_end_ratio: float | None = None

    def __post_init__(self) -> None:
        """Normalise JSON-decoded sequences so the record stays hashable.

        Args:
            None; called by the dataclass generated ``__init__``.

        Raises:
            TypeError: when a sequence field has an unexpected element type.
        """
        if isinstance(self.paired_seed_diffs, list):
            object.__setattr__(self, "paired_seed_diffs", tuple(self.paired_seed_diffs))
        if isinstance(self.two_level_bootstrap_ci, list):
            ci = tuple(self.two_level_bootstrap_ci)
            if len(ci) != 2:
                raise TypeError("two_level_bootstrap_ci must be a (low, high) pair")
            object.__setattr__(self, "two_level_bootstrap_ci", ci)

    def as_dict(self) -> dict[str, Any]:
        """Return the claim as a plain mapping.

        Returns:
            A JSON-serialisable dict of every declared field.
        """
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    rule: str
    status: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check(claim: Claim) -> list[Finding]:
    """Run Gate 1 and P1-P6 over one claim, in the protocol's order.

    Args:
        claim: the candidate's reported evidence. ``None`` means unreported.

    Returns:
        One :class:`Finding` per rule. Missing evidence is always a FAIL;
        no rule may pass merely because its inputs were not supplied.

    Raises:
        None. Malformed values are reported as findings, not exceptions.
    """
    out: list[Finding] = []

    # -- Gate 1 / §11.2: physical information source ------------------------
    if claim.sealed_direction is not None:
        reason = SEALED_DIRECTIONS.get(
            claim.sealed_direction, "unknown sealed direction id"
        )
        out.append(Finding(
            "§9/§10.2 sealed direction", REJECT,
            f"'{claim.name}' maps to a sealed direction ({claim.sealed_direction}): {reason}. "
            "§11.8.2: 被证否 = 停止派生，不是换个说法再试一次。",
        ))
    if claim.new_observation_source is not True:
        out.append(Finding(
            "§11.2 (1) new physical information source", WARN,
            "模块未声明引入新的观测量或可核验的因果先验；按 §11.2 它只能在条件期望 "
            "E[Y|X] 内打转，不得把 E33 量出的信息缺口当成可训练的优化缺口。",
        ))
    if claim.single_variable_control is not True:
        out.append(Finding(
            "§11.2 (2) single-variable control", WARN,
            "对照未声明为单变量受控 + 真实通道参数配平；参数量差异无法与增益分离。",
        ))

    # -- P1 ------------------------------------------------------------------
    if claim.n_seeds < 10 or claim.single_peak_csi2 is not None and claim.pass_rate is None:
        out.append(Finding(
            "P1 single peak", FAIL,
            f"独立 seed = {claim.n_seeds} (<10) 或仅给出单次峰值 "
            f"({claim.single_peak_csi2})。单 seed 极差 {SEED_RANGE} = 可用空间 "
            f"{HEADROOM} 的 {SEED_RANGE / HEADROOM:.2f} 倍：单次过地板与抽签不可区分。",
        ))
    else:
        out.append(Finding("P1 single peak", PASS, f"{claim.n_seeds} 个独立 seed，未以单次峰值作判据。"))

    # -- P2 ------------------------------------------------------------------
    # §11.9 P2 requires the four measured batch values, not just a bool.
    # A bare "reproduced" assertion is unfalsifiable, so it cannot pass alone.
    anchors = (
        ("qpe_0 通过率", claim.batch_qpe0_pass_rate, "qpe_0_pass_rate"),
        ("base_concat 通过率", claim.batch_base_concat_pass_rate, "base_concat_pass_rate"),
        ("Fisher 单侧 p", claim.batch_fisher_one_sided_p, "fisher_one_sided_p"),
        ("峰值端离散", claim.batch_peak_dispersion, "peak_dispersion"),
    )
    anchor_gaps = [
        f"{label} 未报告" if value is None
        else f"{label} = {value}，锚点 {BATCH_ANCHORS[key]}"
        for label, value, key in anchors
        if value is None or abs(float(value) - float(BATCH_ANCHORS[key])) > 1e-6
    ]
    if claim.batch_reproduced is not True or anchor_gaps:
        detail = "；".join(anchor_gaps) if anchor_gaps else "未声明同批次复现"
        out.append(Finding(
            "P2 batch anchors", FAIL,
            f"同批次基线锚点未复现：{detail}。按 §11.9 P2 该批次所有『提升』作废。",
        ))
    else:
        out.append(Finding("P2 batch anchors", PASS, "同批次四项锚点逐项复现。"))

    # -- P3 ------------------------------------------------------------------
    if claim.pass_rate is None:
        out.append(Finding(
            "P3 epoch binding", FAIL,
            "未提供 ≥10 seed 通过率，无法判定其是否绑定 ep00（缺失证据不作 PASS）。",
        ))
    elif claim.epoch != BATCH_ANCHORS["epoch"]:
        out.append(Finding(
            "P3 epoch binding", FAIL,
            f"通过率 {claim.pass_rate} 未绑定 ep00（当前 {claim.epoch!r}）。"
            "96 份复算：ep00 5/10 对 4/10、ep01 6/10 对 6/10、ep02 2/10 对 4/10。",
        ))
    else:
        out.append(Finding("P3 epoch binding", PASS, "通过率绑定 ep00 口径。"))

    # -- P4 ------------------------------------------------------------------
    missing = [
        label for label, value in (
            ("① ≥10 seed 过地板通过率", claim.pass_rate),
            ("② 同批次基线通过率", claim.baseline_pass_rate),
            ("③ 逐 seed 配对差值", claim.paired_seed_diffs),
            ("④ 两级配对 bootstrap", claim.two_level_bootstrap_ci),
        ) if value is None
    ]
    if missing:
        out.append(Finding(
            "P4 four required numbers", FAIL,
            "缺少报数四件套：" + "、".join(missing) + "。仅覆盖窗口抽样不覆盖种子方差的 CI 不算。",
        ))
    else:
        ci = claim.two_level_bootstrap_ci
        assert ci is not None
        ci_excludes_zero = ci[0] > 0 or ci[1] < 0
        status = PASS if ci_excludes_zero else FAIL
        out.append(Finding(
            "P4 four required numbers", status,
            f"四件套齐全；两级配对 bootstrap CI = [{ci[0]}, {ci[1]}]"
            + ("（不含零）" if ci_excludes_zero else "（含零 → 不可分离）"),
        ))

    # -- P5 ------------------------------------------------------------------
    p5: list[str] = []
    if claim.gain is None:
        p5.append("未报告增益，无法检验是否落在可用空间内")
    elif claim.gain > HEADROOM:
        p5.append(
            f"增益 {claim.gain} 超出 Oracle−地板可用空间 {HEADROOM} → 先怀疑窗口集/种子/信息泄漏"
        )
    if claim.collapse_end_ratio is None:
        p5.append("未报告崩塌端差值比")
    elif claim.collapse_end_ratio < BATCH_ANCHORS["collapse_end_ratio"]:
        p5.append(f"崩塌端差值 {claim.collapse_end_ratio} < {BATCH_ANCHORS['collapse_end_ratio']}×")
    if claim.peak_end_ratio is None:
        p5.append("未报告峰值端差值比")
    elif claim.peak_end_ratio < BATCH_ANCHORS["peak_end_ratio"]:
        p5.append(f"峰值端差值 {claim.peak_end_ratio} < {BATCH_ANCHORS['peak_end_ratio']}×")
    out.append(Finding(
        "P5 difference thresholds", FAIL if p5 else PASS,
        "；".join(p5) if p5 else f"增益落在 {HEADROOM} 内且两端差值均超过门槛。",
    ))

    # -- P6 ------------------------------------------------------------------
    # §11.9 P6: disposition and archival. This rule is about what happens to
    # a failing claim, so it is deliberately the last finding and reads the
    # outcomes already accumulated rather than the claim again.
    blocked = [f for f in out if f.status in (FAIL, REJECT)]
    if blocked:
        out.append(Finding(
            "P6 disposition & archival", FAIL,
            f"共 {len(blocked)} 条规则未通过：结果不得写入 §10.1 性能层，只可留在详情文件里作"
            "诊断；按 §11.8 被证否即停止派生，舍弃必须留档，删文件等于销毁证据。"
            "本条与本次 diff 一并归档为裁决证据。",
        ))
    else:
        out.append(Finding(
            "P6 disposition & archival", PASS,
            "无违规项，结果可进入 §10.1 性能层候选；仍需保留 seed 清单与两级 bootstrap 记录。",
        ))

    return out


def verdict(findings: list[Finding]) -> str:
    """Collapse rule outcomes into one disposition (never a scientific claim).

    Args:
        findings: the output of :func:`check` for a single claim.

    Returns:
        One of ``REJECTED_AT_GATE_1``, ``INELIGIBLE``,
        ``ELIGIBLE_WITH_WARNINGS``, ``ELIGIBLE``. REJECT outranks FAIL.

    Raises:
        None.
    """
    statuses = {f.status for f in findings}
    if REJECT in statuses:
        return "REJECTED_AT_GATE_1"
    if FAIL in statuses:
        return "INELIGIBLE"
    if WARN in statuses:
        return "ELIGIBLE_WITH_WARNINGS"
    return "ELIGIBLE"


def required_evidence() -> list[str]:
    """Return the pre-registration reporting checklist (§11.9 P2/P4/P5).

    Args:
        None.

    Returns:
        The required evidence items, in protocol order.

    Raises:
        None.
    """
    return [
        "≥10 独立训练 seed 的 CSI@2.0 过地板通过率（绑定 ep00）",
        "同批次 qpe_0 / base_concat 各 ≥10 seed 的通过率、Fisher 单侧 p、峰值端离散度（四项实测值，非布尔声明）",
        "逐 seed 配对差值及其方向一致性计数",
        "两级配对 bootstrap（同时重采样评测窗口与训练种子）的 95% CI",
        "崩塌端与峰值端的差值比（门槛 1.78× / 9.6×）与增益是否落在 0.0729 内",
    ]

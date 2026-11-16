"""The baseline catalogue: one typed record for every baseline in the pool.

Group assignments and paper attributions come from ``MASTER.md`` §2.1 / §2.2
and ``README.md`` §5 of the upstream repository. The fourteen names in
``BLUEPRINT_POOL`` are the ones listed in §6.1 of the Q-LagrangeFlow blueprint
(``from_gemini/radar_nwp_quantum_inspired_ml_blueprint_cvpr_icml.md``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs" / "baselines.json"

#: group id -> human-readable description.
GROUPS: dict[str, str] = {
    "paper_classic": "复现的论文方法 · 经典时空预测",
    "paper_multimodal": "复现的论文方法 · 多模态/生成式 SOTA",
    "physical": "无训练参数的物理/解析基线",
    "ablation": "本项目自建变体 · 已纳入主表的消融臂",
    "proposal_only": "已注册但未纳入权威主表的路线（多为已舍弃方向）",
}

#: name -> (group, paper, venue, license, blueprint alias or None).
#: ``None`` for the blueprint alias means the model is not in the blueprint's
#: fourteen-baseline pool. Sources: MASTER.md §2.1 / §2.2, README.md §5.
BASELINE_METADATA: dict[str, tuple[str, str, str, str, str | None]] = {
    "convlstm": (
        "paper_classic", "ConvLSTM", "NeurIPS 2015 / arXiv 1907.05242",
        "in-repo", "ConvLSTM",
    ),
    "predrnn": (
        "paper_classic", "PredRNN / ST-LSTM", "NeurIPS 2017",
        "in-repo", "PredRNN",
    ),
    "earthformer": (
        "paper_classic", "Earthformer", "NeurIPS 2022",
        "in-repo", "Earthformer",
    ),
    "simvp_v2": (
        "paper_classic", "SimVP-v2", "ICLR 2022",
        "in-repo", "SimVP-v2 (项目骨干)",
    ),
    "metnet3": (
        "paper_multimodal", "MetNet-3", "arXiv 2306.06079 (2023)",
        "MIT (3rd-party)", "MetNet-3",
    ),
    "diffcast": (
        "paper_multimodal", "DiffCast", "CVPR 2024",
        "GPL-3.0", "DiffCast",
    ),
    "cascast_nwp": (
        "paper_multimodal", "CasCast (+NWP conditioning)", "ICML 2024",
        "none asserted", "CasCast",
    ),
    "mocast": (
        "paper_multimodal", "MoCast", "AAAI 2026",
        "n/a (no code released)", "MoCast",
    ),
    "pw_foucast": (
        "paper_multimodal", "PW-FouCast", "IJCNN 2026",
        "none asserted", "PW-FouCast",
    ),
    "fusioncast": (
        "paper_multimodal", "FusionCast", "arXiv 2603.13298 (2026)",
        "none asserted", "FusionCast",
    ),
    "mmf_rnn": (
        "paper_multimodal", "MMF-RNN", "IEEE TGRS 2025",
        "n/a (no code released)", "MMF-RNN",
    ),
    "furecast": (
        "paper_multimodal", "FURECast", "GRL 2026",
        "CC-BY-4.0 (utils)", "FURECast",
    ),
    "pysteps_adv": (
        "physical", "PySTEPS advection", "JOSS 2023",
        "MIT", "PySTEPS-Advection",
    ),
    "persistence": (
        "physical", "Zero-velocity persistence (Z-R copy)", "analytical",
        "in-repo", "Persistence",
    ),
    "simvp_radar_only": (
        "ablation", "SimVP, radar-only", "in-repo ablation", "in-repo", None,
    ),
    "simvp_early_concat": (
        "ablation", "SimVP, early-concat", "in-repo ablation", "in-repo", None,
    ),
    "simvp_v2_nwp_concat": (
        "ablation", "M1: NWP input concat", "in-repo ablation", "in-repo", None,
    ),
    "simvp_v2_nwp_xattn": (
        "ablation", "M2: NWP spatial cross-attention", "in-repo ablation", "in-repo", None,
    ),
    "simvp_v2_nwp_delta": (
        "ablation", "M3: NWP tendency difference", "in-repo ablation", "in-repo", None,
    ),
    "simvp_v2_quantum": (
        "ablation", "M4: Koopman tendency coupler", "in-repo ablation", "in-repo", None,
    ),
    "simvp_v2_hqma": (
        "ablation", "HQMA: ordinal head + coupler", "in-repo ablation", "in-repo", None,
    ),
    "physical_qpe": (
        "proposal_only", "Physical QPE (Z-R analytic)", "in-repo proposal", "in-repo", None,
    ),
    "physical_qpe_v2": (
        "proposal_only", "Physical QPE v2", "in-repo proposal", "in-repo", None,
    ),
    "simvp_v2_assimilated": (
        "proposal_only", "Assimilated observation route", "in-repo proposal", "in-repo", None,
    ),
    "simvp_v2_lagrangian": (
        "proposal_only", "Lagrangian warp route", "in-repo proposal", "in-repo", None,
    ),
    "simvp_v2_qpe_guided": (
        "proposal_only", "QPE-guided two-stage route", "in-repo proposal", "in-repo", None,
    ),
}

#: The exact fourteen names in blueprint §6.1. Stated separately so the
#: verifier can detect a blueprint/catalogue drift rather than infer it.
BLUEPRINT_POOL: tuple[str, ...] = (
    "convlstm", "predrnn", "simvp_v2", "earthformer",
    "pysteps_adv", "persistence",
    "metnet3", "cascast_nwp", "diffcast", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
)


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline, as the integration layer knows it."""

    name: str
    group: str
    paper: str
    venue: str
    license: str
    blueprint_alias: str | None
    adapter: str | None = None
    adapter_class: str | None = None
    config: str | None = None
    is_learnable: bool = True
    has_eval: bool = False
    has_robustness: bool = False
    has_score_aware: bool = False
    upstream_notes: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # JSON round-trips a tuple as a list. Normalise back so the record
        # stays hashable and can be used in sets by callers.
        if isinstance(self.tags, list):
            object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def in_blueprint_pool(self) -> bool:
        return self.blueprint_alias is not None

    @property
    def is_paper_baseline(self) -> bool:
        return self.group.startswith("paper_")

    @property
    def is_analytical(self) -> bool:
        return not self.is_learnable

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Catalog:
    """An immutable, validated view of ``configs/baselines.json``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else CATALOG_PATH
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.schema_version: int = int(payload["schema_version"])
        self.source_repo: str = payload["source_repo"]
        self.source_revision: str = payload.get("source_revision", "unknown")
        self.master_version: str = payload.get("master_version", "unknown")
        self.contract: dict[str, Any] = payload["contract"]
        self._by_name: dict[str, BaselineSpec] = {
            row["name"]: BaselineSpec(**row) for row in payload["baselines"]
        }

    def all(self) -> tuple[BaselineSpec, ...]:
        return tuple(sorted(self._by_name.values(), key=lambda s: s.name))

    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.all())

    def get(self, name: str) -> BaselineSpec | None:
        return self._by_name.get(name)

    def by_group(self, group: str) -> tuple[BaselineSpec, ...]:
        return tuple(s for s in self.all() if s.group == group)

    def blueprint_pool(self) -> tuple[BaselineSpec, ...]:
        return tuple(s for s in self.all() if s.in_blueprint_pool)

    def with_evidence(self, kind: str) -> tuple[BaselineSpec, ...]:
        attr = {
            "eval": "has_eval",
            "robustness": "has_robustness",
            "score_aware": "has_score_aware",
        }.get(kind)
        if attr is None:
            raise ValueError(f"unknown evidence kind: {kind}")
        return tuple(s for s in self.all() if getattr(s, attr))

    def group_counts(self) -> dict[str, int]:
        counts = {g: 0 for g in GROUPS}
        for spec in self.all():
            counts[spec.group] = counts.get(spec.group, 0) + 1
        return counts

    def validate(self, bench_root: Path) -> list[str]:
        """Return a list of integration defects; empty means fully wired.

        Checks are structural: every declared adapter/config exists, the
        blueprint pool is complete, analytical baselines are flagged, and
        evidence flags agree with the upstream files on disk.
        """
        problems: list[str] = []
        root = Path(bench_root)

        wired = {s.name for s in self.blueprint_pool()}
        if wired != set(BLUEPRINT_POOL):
            problems.append(
                "blueprint pool mismatch; "
                f"missing: {sorted(set(BLUEPRINT_POOL) - wired)}, "
                f"unexpected: {sorted(wired - set(BLUEPRINT_POOL))}"
            )

        for spec in self.all():
            for label, rel in (("adapter", spec.adapter), ("config", spec.config)):
                if rel and not (root / rel).is_file():
                    problems.append(f"{spec.name}: declared {label} not found: {rel}")
            if spec.group == "physical" and spec.is_learnable:
                problems.append(f"{spec.name}: analytical baseline must set is_learnable=False")
            if spec.has_eval and not (root / ".workbuddy" / "runs" / f"{spec.name}.eval.json").is_file():
                problems.append(f"{spec.name}: has_eval=True but report missing")
            if spec.has_robustness and not (root / ".workbuddy" / "runs" / f"{spec.name}.robustness.json").is_file():
                problems.append(f"{spec.name}: has_robustness=True but report missing")
            if spec.has_score_aware and not spec.is_learnable:
                problems.append(f"{spec.name}: score-aware matrix only covers trained models")

        runs = root / ".workbuddy" / "runs"
        covered = {s.name for s in self.all() if s.has_eval}
        for report in sorted(runs.glob("*.eval.json")):
            name = report.name[: -len(".eval.json")]
            if name not in covered:
                problems.append(f"upstream eval report not wired into catalogue: {name}")

        return problems

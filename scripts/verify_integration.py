"""Verify that every upstream baseline is wired in and that the gate works.

    python scripts/verify_integration.py

Exits non-zero on any defect. Deliberately checks the *scientific* invariants
too: the floor must equal the recorded persistence score, the score-aware
table must be fully covered, and a blueprint-style claim must be rejected by
the evidence gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qmlbench import BLUEPRINT_POOL, Catalog, FLOOR_CSI2, BenchSource, decidability
from qmlbench.gate import Claim, check, verdict
from qmlbench.results import score_aware_table, standard_table
from qmlbench.tables import master_table

PASS, FAIL = "PASS", "FAIL"


def report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{PASS if ok else FAIL}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", default=None)
    args = ap.parse_args()

    source = BenchSource.locate(args.bench_root)
    catalog = Catalog()
    checks: list[bool] = []

    print(f"upstream: {source.root} @ {source.git_revision()} ({catalog.master_version})")
    print(f"catalogue: {len(catalog.all())} baselines, schema v{catalog.schema_version}\n")

    # -- structural wiring --------------------------------------------------
    problems = catalog.validate(source.root)
    checks.append(report("catalogue structural validation", not problems,
                         "; ".join(problems) if problems else "no defects"))

    blueprint_names = {s.name for s in catalog.blueprint_pool()}
    checks.append(report("blueprint §6.1 pool complete",
                         blueprint_names == set(BLUEPRINT_POOL),
                         f"{len(blueprint_names)}/{len(BLUEPRINT_POOL)}"))

    adapters = source.scan_adapters()
    catalog_adapters = {s.adapter for s in catalog.all() if s.adapter}
    checks.append(report("all registered adapters represented",
                         catalog_adapters == {a["adapter"] for a in adapters.values()},
                         f"{len(catalog_adapters)} adapter files"))

    configs = source.scan_configs()
    catalog_configs = {s.config for s in catalog.all() if s.config}
    checks.append(report("all model configs represented",
                         catalog_configs == {c["config"] for c in configs.values()},
                         f"{len(catalog_configs)} configs"))

    # -- evidence coverage --------------------------------------------------
    evals = source.scan_eval_reports(".eval.json")
    checks.append(report("all eval reports wired",
                         {s.name for s in catalog.with_evidence("eval")} == set(evals),
                         f"{len(evals)} reports"))

    sa = score_aware_table(source)
    checks.append(report("authoritative score-aware table parsed and covered",
                         len(sa) >= 18 and set(sa) <= {s.name for s in catalog.with_evidence("score_aware")},
                         f"{len(sa)} rows"))

    persistence_report = evals.get("persistence", {})
    at_two_mm = (persistence_report.get("thresholds") or {}).get("2.0") or {}
    observed_floor = at_two_mm.get("csi")
    checks.append(report("persistence floor constant matches report",
                         observed_floor is not None and round(observed_floor, 4) == FLOOR_CSI2,
                         f"{observed_floor} vs {FLOOR_CSI2}"))

    d = decidability()
    checks.append(report("single-run undecidability reproduced",
                         d["range_over_headroom"] > 2.0 and not d["best_trained_beats_floor"],
                         f"range/headroom = {d['range_over_headroom']}×, "
                         f"best trained {d['best_trained']} < floor {d['floor']}"))

    # -- the gate itself ----------------------------------------------------
    blueprint_claim = Claim(
        name="Q-LagrangeFlow (blueprint as written)",
        mechanism="Lagrangian drift + Hilbert/PEPS + flow matching",
        sealed_direction="hilbert_peps",
        n_seeds=0,
        single_peak_csi2=0.1250,
    )
    checks.append(report("gate rejects the sealed blueprint claim",
                         verdict(check(blueprint_claim)) == "REJECTED_AT_GATE_1"))

    single_run = Claim(name="single-peak improvement", mechanism="new loss",
                       n_seeds=1, single_peak_csi2=0.1300)
    checks.append(report("gate rejects a single-peak-only claim",
                         "P1 single peak" in {
                             f.rule for f in check(single_run) if f.status == FAIL
                         }))

    eligible = Claim(
        name="fully-evidenced candidate",
        mechanism="new observation channel, single-variable control",
        new_observation_source=True,
        single_variable_control=True,
        n_seeds=10,
        batch_reproduced=True,
        batch_qpe0_pass_rate=0.5,
        batch_base_concat_pass_rate=0.4,
        batch_fisher_one_sided_p=0.5,
        batch_peak_dispersion=9.59,
        epoch="ep00",
        pass_rate=0.7,
        baseline_pass_rate=0.5,
        paired_seed_diffs=[0.01] * 10,
        two_level_bootstrap_ci=(0.002, 0.030),
        gain=0.015,
        collapse_end_ratio=2.1,
        peak_end_ratio=11.0,
    )
    checks.append(report("gate admits a fully-evidenced candidate",
                         verdict(check(eligible)) == "ELIGIBLE"))

    # Regression for the two-axis review: unreported evidence must never be
    # scored as PASS, and P6 (disposition + archival) must be present.
    blueprint_findings = check(blueprint_claim)
    by_rule = {f.rule: f.status for f in blueprint_findings}
    checks.append(report("unreported evidence is FAIL, not PASS",
                         by_rule.get("P3 epoch binding") == "FAIL"
                         and by_rule.get("P5 difference thresholds") == "FAIL"
                         and by_rule.get("P6 disposition & archival") == "FAIL"))
    checks.append(report("Claim records are hashable",
                         isinstance(hash(eligible), int)))

    # The score-aware matrix has one row (simvp_early_concat) with no
    # standard-loss counterpart; the renderer must not drop it.
    rendered = master_table(standard_table(source), sa, catalog)
    checks.append(report("master table renders every score-aware row",
                         all(f"`{name}`" in rendered for name in sa),
                         f"{len(sa)} rows"))


    print()
    failed = len([c for c in checks if not c])
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

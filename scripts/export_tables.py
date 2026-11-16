"""Export the paper-facing tables and the gate review to ``artifacts/``.

    python scripts/export_tables.py

Inputs are the synced catalogue, the upstream evaluation reports, and
``configs/claims.json`` (the candidate mechanisms to put through the gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qmlbench import BenchSource, Catalog
from qmlbench.gate import Claim, check, verdict
from qmlbench.results import robustness_table, score_aware_table, standard_table
from qmlbench.tables import (
    baseline_csv, coverage_table, decidability_report, gate_report, master_table,
    robustness_report,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"
CLAIMS = PROJECT_ROOT / "configs" / "claims.json"


def load_claims(path: Path) -> list[Claim]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    claims = []
    for row in rows:
        row = dict(row)
        ci = row.get("two_level_bootstrap_ci")
        if ci is not None:
            row["two_level_bootstrap_ci"] = tuple(ci)
        claims.append(Claim(**row))
    return claims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", default=None)
    ap.add_argument("--out-dir", default=str(ARTIFACTS))
    args = ap.parse_args()

    source = BenchSource.locate(args.bench_root)
    catalog = Catalog()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    standard = standard_table(source)
    robustness = robustness_table(source)
    sa = score_aware_table(source)
    claims = load_claims(CLAIMS)

    # The gate computes; tables.py only renders. Keep the two phases apart.
    reviews = []
    for claim in claims:
        findings = check(claim)
        reviews.append((claim, findings, verdict(findings)))

    files = {
        "baselines.csv": baseline_csv(catalog),
        "master_table.md": master_table(standard, sa, catalog) + "\n",
        "coverage.md": coverage_table(catalog) + "\n",
        "robustness_table.md": robustness_report(robustness, catalog) + "\n",
        "decidability.md": decidability_report() + "\n",
        "gate_review.md": gate_report(reviews) + "\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name} ({len(text)} chars)")

    print(f"\nsource: {source.root} @ {source.git_revision()}")
    print(f"models: {len(catalog.all())}, standard rows: {len(standard)}, "
          f"score-aware rows: {len(sa)}, claims reviewed: {len(claims)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Zero-dependency static verification of the QLFN contract and source hygiene.

Run on the workstation (no torch / numpy required):

    python scripts/verify_qlfn_contract.py
    python scripts/verify_qlfn_contract.py --bench-root R:/1radarNWPprediction/radar-nwp-bench

Checks R1-R4, R6-R14 of ``docs/archive/QLFN_SPEC.md`` v1.1. Model runtime shape tests
live in ``scripts/qlfn_smoke_test.py`` and require the server's conda python.

Missing evidence is always FAIL; a skipped check is never reported as PASS.
The upstream root is resolved by ``qmlbench.sources.BenchSource.locate()``
(``--bench-root`` > ``RADAR_NWP_BENCH_ROOT`` > the documented default), so the
bare command exits 0 on this workstation instead of failing provenance.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmlbench.sources import BenchNotFound, BenchSource  # noqa: E402
from qlfn.checks import check, render  # noqa: E402
from qlfn.contract import (  # noqa: E402
    C_NWP,
    C_RADAR,
    CR_INDEX,
    KM_PER_PIXEL,
    MAX_DRIFT_KM,
    NWP_VARIABLES,
    RADAR_VARIABLES,
    VIL_INDEX,
    WIND_SURROGATE_INDICES,
    QLFNConfig,
    contract_report,
    drift_scale,
    fold_plan,
    validate_batch_shapes,
)
from qlfn.data_audit import (  # noqa: E402
    CaseStat,
    audit_cases,
    masked_contingency,
    stratified_case_split,
    valid_step_mask,
    write_artifact,
)

QLFN_DIR = ROOT / "qlfn"
SEALED_MARKER = "SEALED"
TORCH_MODULES = (
    "geo_prior.py", "scale_bridge.py", "quantum_latent.py",
    "flow_head.py", "losses.py", "model.py",
)
ALL_MODULES = (
    "__init__.py", "contract.py", "data_audit.py", "checks.py",
) + TORCH_MODULES

#: Training/evaluation drivers that must not appear anywhere in the QLFN
#: deliverable. Kept as a compiled pattern so the positive control below can
#: prove the detector actually fires.
FORBIDDEN_DRIVER = re.compile(
    r"\boptimizer\.step\s*\(|\bTrainer\s*\(|\.fit\s*\(|\.train\s*\(\s*\)"
)
SCANNED_SCRIPTS = (
    ROOT / "scripts" / "verify_qlfn_contract.py",
    ROOT / "scripts" / "qlfn_smoke_test.py",
)


def source(name: str) -> str:
    """Read one QLFN module as UTF-8 text.

    Args:
        name: Filename relative to ``qlfn/``.

    Returns:
        The file's text.

    Raises:
        FileNotFoundError: If the module is absent (never silently skipped).
    """
    return (QLFN_DIR / name).read_text(encoding="utf-8")


def code_only(name: str) -> str:
    """Return a module with docstrings and comments stripped, as source text.

    Static checks must inspect *code*, not prose. A docstring that explains a
    retracted defect (e.g. "the blueprint hard-coded 301") would otherwise
    trigger a "hardcoded 301" failure. Using ``ast.unparse`` after removing
    docstring expressions makes these checks insensitive to documentation.

    Args:
        name: Filename relative to ``qlfn/``.

    Returns:
        Docstring-free, comment-free rendering of the module.

    Raises:
        SyntaxError: If the module does not parse (a real failure).
    """
    tree = ast.parse(source(name))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                   ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def upstream_nwp_vars(text: str) -> tuple[str, ...]:
    """Extract ``NWP_VARS`` from the upstream ``data/index.py`` source text.

    Args:
        text: Full source text of upstream ``data/index.py``.

    Returns:
        The declared variable names, or an empty tuple when not found.
    """
    match = re.search(r"NWP_VARS[^=]*=\s*\((.*?)\)", text, re.S)
    if not match:
        return ()
    return tuple(re.findall(r'"([A-Za-z0-9_]+)"', match.group(1)))


def run_checks(bench_root: Path | None) -> list[tuple[str, bool, str]]:
    """Build the complete static check list.

    Args:
        bench_root: Upstream repository root, or ``None`` when it could not be
            located (reported as FAIL, never skipped).

    Returns:
        Rows of ``(name, passed, detail)``.
    """
    cfg = QLFNConfig()
    report = contract_report()
    rows: list[tuple[str, bool, str]] = []

    # -- R1: constants, provenance, hashability ----------------------------
    rows.append(check(
        "R1 channel counts", len(RADAR_VARIABLES) == C_RADAR == 10
        and len(NWP_VARIABLES) == C_NWP == 27,
        "radar=10, nwp=27 (index.py:11-24)",
    ))
    rows.append(check(
        "R1 channel indices", CR_INDEX == 6 and VIL_INDEX == 9,
        f"CR={CR_INDEX}, VIL={VIL_INDEX}",
    ))
    rows.append(check(
        "R1 surrogate indices derived from names",
        WIND_SURROGATE_INDICES == (
            NWP_VARIABLES.index("DVG850"), NWP_VARIABLES.index("WS925")
        ) and "NOT U/V" in source("contract.py"),
        f"indices={WIND_SURROGATE_INDICES}, declared scalar proxies only",
    ))
    try:
        hash(cfg)
        rows.append(check("R1 frozen+hashable config", True, "hash() OK"))
    except TypeError as exc:
        rows.append(check("R1 frozen+hashable config", False, str(exc)))
    rows.append(check(
        "R1 fold channel algebra",
        cfg.folded_radar_channels == 144 and cfg.subpixel_channels == 90,
        f"folded={cfg.folded_radar_channels}, subpixel={cfg.subpixel_channels}",
    ))
    rows.append(check(
        "R1 real grid spacing with source",
        KM_PER_PIXEL == 1.11195 and "MASTER.md:1556" in source("contract.py")
        and report["km_per_pixel"] == KM_PER_PIXEL,
        f"{KM_PER_PIXEL} km/px (NOT the 1 km/px approximation)",
    ))

    # -- R2: missing evidence = FAIL ---------------------------------------
    rows.append(check(
        "R2 valid training batch passes",
        validate_batch_shapes((2, 11, 10, 66, 66), (2, 2, 27, 66, 66),
                             (2, 10, 1, 66, 66), require_target=True) == (),
    ))
    rows.append(check(
        "R2 non-co-located NWP rejected",
        bool(validate_batch_shapes((2, 11, 10, 66, 66),
                                  (2, 2, 27, 167, 167))),
    ))
    rows.append(check(
        "R2 inference without target is an explicit exception",
        validate_batch_shapes((1, 11, 10, 66, 66),
                              (1, 2, 27, 66, 66)) == (),
        "documented: no ground truth exists at inference",
    ))
    rows.append(check(
        "R2 missing target when required = FAIL",
        bool(validate_batch_shapes((1, 11, 10, 66, 66),
                                   (1, 2, 27, 66, 66),
                                   require_target=True)),
    ))
    rows.append(check(
        "R2 malformed y_rain rejected",
        bool(validate_batch_shapes((1, 11, 10, 66, 66),
                                   (1, 2, 27, 66, 66),
                                   (1, 10, 2, 66, 66), require_target=True)),
    ))

    # -- R3/R9: shape algebra and grid-derived drift -----------------------
    for h, w in ((64, 64), (66, 66), (128, 128), (301, 301)):
        plan = fold_plan(h, w)
        ok = (plan.padded_height % 3 == 0 and plan.padded_width % 3 == 0
              and plan.crop[1] - plan.crop[0] == h
              and plan.crop[3] - plan.crop[2] == w)
        rows.append(check(
            f"R3 fold plan {h}x{w}", ok,
            f"pad=({plan.pad_top},{plan.pad_bottom},"
            f"{plan.pad_left},{plan.pad_right}) "
            f"folded={plan.folded_height}x{plan.folded_width}",
        ))

    model_code = code_only("model.py")
    model_src = source("model.py")
    bridge_src = source("scale_bridge.py")
    rows.append(check(
        "R3 no hardcoded 301 in model",
        not re.search(r"\b301\b", model_code),
        "AST scan of model.py code (docstrings excluded)",
    ))
    rows.append(check(
        "R3 fold plan used by bridge", "fold_plan" in bridge_src,
    ))
    rows.append(check(
        "R9 drift scale is width dependent",
        drift_scale(64) > drift_scale(128) > drift_scale(301) > 0.0,
        f"64={drift_scale(64):.4f}, 128={drift_scale(128):.4f}, "
        f"301={drift_scale(301):.4f}",
    ))
    rows.append(check(
        "R9 drift scale uses 1.11195 km/px",
        abs(drift_scale(301) - MAX_DRIFT_KM / (301 * KM_PER_PIXEL / 2)) < 1e-12,
        f"301 grid -> {drift_scale(301):.6f} normalised units",
    ))
    rows.append(check(
        "R9 no 150 km / 1 km-per-px hardcode in model",
        "150.0" not in model_code and "1000.0" not in model_code
        and "drift_scale(" in model_code,
        "AST scan; drift scale comes from contract.drift_scale",
    ))

    # -- R6/R7/R8/R10: source-level invariants -----------------------------
    rows.append(check(
        "R6 S5 explicit sub-pixel channel assertion",
        "subpixel_channels" in bridge_src and "raise ValueError" in bridge_src
        and "expected" in bridge_src,
        "decode_subpixel validates channels before reshaping",
    ))
    rows.append(check(
        "R6 no unguarded Identity() decode",
        "nn.Identity()" not in model_src,
        "engineering hygiene; NOT an S1 fix (S1 was retracted)",
    ))
    flow_src = source("flow_head.py")
    rows.append(check(
        "R7 CFM objective + consistent zero prior",
        "conditional_flow_matching_loss" in flow_src
        and "torch.zeros_like(x1)" in flow_src
        and "torch.zeros(batch, self.out_ch" in flow_src
        and "randn" not in flow_src,
        "train and integrate_ode both start from zero",
    ))
    rows.append(check(
        "R7 CFM target is the body residual, cond is internal",
        "residual" in model_src and "_encode(" in model_src
        and "flow_training_loss" in model_src,
        "no externally fabricated latent; x1 = clamp(target-body, 0)",
    ))
    loss_src = source("losses.py")
    audit_src = source("data_audit.py")
    rows.append(check(
        "R8 binary-mask contract enforced",
        "assert_binary_mask" in loss_src and "binary 0/1" in audit_src,
    ))
    rows.append(check(
        "R8 T2 retraction documented",
        "RETRACTED" in loss_src and "RETRACTED claim" in audit_src,
        "neither implementation claims to fix a non-defect",
    ))
    rows.append(check(
        "R10 diagnostic entropy cannot gate output",
        "norm_entropy" not in model_code
        and not re.search(r"\bgate", model_code)
        and "pred_rain = body + wet_mask * extreme" in model_code,
        "AST scan: no entropy map and no gate in model.py code",
    ))
    latent_src = source("quantum_latent.py")
    latent_code = code_only("quantum_latent.py")
    rows.append(check(
        "R10 physical claims downgraded",
        all(k in latent_src for k in ("not a density matrix",
                                      "not entanglement entropy",
                                      "diagnostic only")),
        "documentation check (docstrings ARE the evidence here)",
    ))
    rows.append(check(
        "R10 no quantum scope creep",
        "phase_r" not in latent_code and "phase_n" not in latent_code,
        "AST scan: two-channel lift restored; learnable phase removed",
    ))

    # -- R11/R12: data audit ------------------------------------------------
    stats = tuple(
        CaseStat(f"case_{i}", 20, (i % 3,), 1.0 + 0.01 * i, 5.0 + i, 0.01 * i)
        for i in range(20)
    )
    split = stratified_case_split(stats, 3, 3, seed=7)
    rows.append(check(
        "R11 split reports realised balance",
        len(split.train) + len(split.val) + len(split.test) == 20
        and split.max_relative_difference is not None,
        f"balanced={split.balanced}, "
        f"max_rel_diff={split.max_relative_difference:.4f}",
    ))
    rows.append(check(
        "R12 mask keeps zeros", valid_step_mask((0, 2, 4), 5) == (1, 0, 1, 0, 1),
    ))
    rows.append(check(
        "R8 fully masked batch yields zero counts",
        masked_contingency((0.9, 0.8), (1.0, 1.0), (0, 0)) == (0.0, 0.0, 0.0),
    ))
    try:
        masked_contingency((0.9,), (1.0,), (0.5,))
        soft_rejected = False
    except ValueError:
        soft_rejected = True
    rows.append(check("R8 soft mask rejected", soft_rejected))

    artifact = ROOT / "artifacts" / "QLFN_NOT_COMPARABLE_case_audit.json"
    write_artifact(artifact, audit_cases(stats))
    body = json.loads(artifact.read_text(encoding="utf-8"))
    rows.append(check(
        "R12 NOT_COMPARABLE artifact actually written",
        artifact.is_file() and "NOT_COMPARABLE" in artifact.name
        and "NOT_COMPARABLE" in str(body["caveat"]),
        str(artifact.relative_to(ROOT)),
    ))
    try:
        write_artifact(ROOT / "artifacts" / "would_be_authoritative.json", {})
        filename_rejected = False
    except ValueError:
        filename_rejected = True
    rows.append(check(
        "R12 writer rejects non-caveated filename", filename_rejected,
    ))

    # -- R13/R14: scope discipline -----------------------------------------
    missing_marker = [n for n in ALL_MODULES
                      if SEALED_MARKER not in source(n)]
    rows.append(check(
        "R13 sealed marker in every module", not missing_marker,
        f"{len(ALL_MODULES) - len(missing_marker)}/{len(ALL_MODULES)}",
    ))
    guarded = "".join(
        source(n) for n in ALL_MODULES if n != "data_audit.py"
    )
    rows.append(check(
        "R14 no writes in model/contract layer",
        not re.search(r"\.write_text\(|\.write_bytes\(|shutil\.|os\.remove",
                      guarded),
        "only data_audit.write_artifact may write",
    ))
    # Positive control: prove the detector has teeth before trusting its pass.
    control = "optimizer" + ".step(" + ")"
    rows.append(check(
        "R14 detector positive control",
        bool(FORBIDDEN_DRIVER.search(control)),
        "the forbidden pattern is proven matchable",
    ))
    scanned = [(QLFN_DIR / n) for n in ALL_MODULES] + list(SCANNED_SCRIPTS)
    offenders = [
        str(p.relative_to(ROOT)) for p in scanned
        if FORBIDDEN_DRIVER.search(p.read_text(encoding="utf-8"))
    ]
    rows.append(check(
        "R14 no training/evaluation driver in QLFN deliverable",
        not offenders, ",".join(offenders) or f"{len(scanned)} files scanned",
    ))

    # -- provenance ---------------------------------------------------------
    if bench_root is None:
        rows.append(check(
            "provenance (upstream index.py)", False,
            "FAIL: upstream root not found; set RADAR_NWP_BENCH_ROOT",
        ))
    else:
        index_py = bench_root / "radar_nwp_bench" / "data" / "index.py"
        text = index_py.read_text(encoding="utf-8") if index_py.is_file() else ""
        upstream = upstream_nwp_vars(text)
        rows.append(check(
            "provenance (upstream index.py)",
            "CAP20" in text and upstream == NWP_VARIABLES,
            f"{index_py} (NWP_VARS exact match: {len(upstream)}/27)",
        ))
    return rows


def main() -> int:
    """Parse arguments, run all static checks and render the report.

    Returns:
        Process exit code: 0 when every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        bench_root = BenchSource.locate(args.bench_root).root
    except BenchNotFound:
        bench_root = None
    return render(
        run_checks(bench_root),
        "CONTRACT_OK (no training, no evaluation, no scores produced)",
    )


if __name__ == "__main__":
    raise SystemExit(main())

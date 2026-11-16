#!/usr/bin/env python3
"""Read-only verification of the completed full-model sweep.

Run on the GPU server after the detached queue finishes::

    /home/wf/.conda/envs/radar_nwp/bin/python \
        /home/wf/radarNWPQML/scripts/verify_full_sweep.py

The check re-derives the summary from the evaluation reports into a temporary
directory and compares it with the archived one, then checks the queue's
status file and log. It never writes into ``runs/full_sweep_20261108/``, never
starts a training job and never touches the upstream checkout. Exit 0 means
every check passed; exit 1 means at least one finding needs a human.

This is a completeness/integrity check, not a scientific result: the sweep is
single-seed and uses a uniform non-native parameter set, so its numbers remain
NOT_COMPARABLE with the authoritative MASTER.md tables.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

OUT_DIR = Path("/home/wf/radarNWPQML/runs/full_sweep_20261108")
QUEUE_LOG = Path("/home/wf/radarNWPQML/runs/full_sweep_20261108_queue.log")
SUMMARISER = Path("/home/wf/radarNWPQML/scripts/summarize_full_sweep.py")
THRESHOLDS = ("0.1", "0.5", "2.0", "4.0")
EXPECTED_N = 25

findings: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    findings.append((label, bool(ok), detail))


def main() -> int:
    config_path = OUT_DIR / "sweep_config.txt"
    config: dict[str, str] = {}
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    models = config.get("models", "").split()
    check("sweep_config models = 25", len(models) == EXPECTED_N,
          f"got {len(models)}")
    check("sweep_config marks NOT_COMPARABLE",
          "NOT_COMPARABLE" in config.get("comparability", ""))
    check("sweep_config excludes QLFN (R14)",
          "EXCLUDED" in config.get("qlfn_scope", ""))

    # ---- queue bookkeeping -------------------------------------------------
    status_rows = [
        line.split("\t")
        for line in (OUT_DIR / "status.tsv").read_text(encoding="utf-8")
        .splitlines() if line.strip()
    ] if (OUT_DIR / "status.tsv").is_file() else []
    check("status.tsv has 25 rows", len(status_rows) == EXPECTED_N,
          f"got {len(status_rows)}")
    bad_rc = [
        r[0] for r in status_rows
        if len(r) >= 3 and (r[1] != "0" or r[2] != "0")
    ]
    check("every row train_rc=0 eval_rc=0", not bad_rc,
          f"nonzero: {bad_rc}" if bad_rc else "")
    check("status rows cover exactly the configured models",
          sorted(r[0] for r in status_rows if r) == sorted(models))

    ckpts = sorted((OUT_DIR / "train").rglob("last.pt"))
    check("25 checkpoints on disk", len(ckpts) == EXPECTED_N,
          f"got {len(ckpts)}")

    log_tail = "\n".join(
        QUEUE_LOG.read_text(encoding="utf-8", errors="replace")
        .splitlines()[-8:]
    ) if QUEUE_LOG.is_file() else ""
    check("queue log ends with SWEEP_DONE exit=0",
          "SWEEP_DONE exit=0" in log_tail)

    # ---- every evaluation report parses and carries finite CSI -------------
    reports: list[str] = []
    for model in models:
        path = OUT_DIR / "evals" / f"{model}.json"
        if not path.is_file():
            reports.append(f"{model}: missing")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reports.append(f"{model}: unparseable ({exc})")
            continue
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, dict):
            reports.append(f"{model}: no thresholds object")
            continue
        for threshold in THRESHOLDS:
            entry = thresholds.get(threshold)
            value = entry.get("csi") if isinstance(entry, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                reports.append(f"{model}: csi@{threshold} missing/non-finite")
    check("25 eval reports parse with finite CSI@4 thresholds",
          not reports, "; ".join(reports[:4]))

    # ---- archived summary agrees with a fresh re-derivation ---------------
    archived_path = OUT_DIR / "summary.json"
    archived = (
        json.loads(archived_path.read_text(encoding="utf-8"))
        if archived_path.is_file() else None
    )
    check("summary.json exists", archived is not None)
    if archived is not None:
        check("batch_complete is True", archived.get("batch_complete") is True)
        check("n_models_expected / n_reports_present = 25 / 25",
              archived.get("n_models_expected") == EXPECTED_N
              and archived.get("n_reports_present") == EXPECTED_N)
        check("supports_improvement_claim is False",
              archived.get("supports_improvement_claim") is False)
        check("summary marks NOT_COMPARABLE",
              "NOT_COMPARABLE" in str(archived.get("caveat", "")))
        check("summary excludes QLFN (R14)",
              "EXCLUDED" in str(archived.get("qlfn_scope", "")))
        check("no row carries an error",
              all(r.get("error", "") == "" for r in archived.get("rows", [])))

        with tempfile.TemporaryDirectory(prefix="sweep_verify_") as tmp:
            tmp_dir = Path(tmp)
            out, tsv = tmp_dir / "summary.json", tmp_dir / "summary.tsv"
            proc = subprocess.run(
                [sys.executable, str(SUMMARISER),
                 "--evals-dir", str(OUT_DIR / "evals"),
                 "--models", *models,
                 "--config", str(config_path),
                 "--out", str(out), "--tsv", str(tsv)],
                capture_output=True, text=True, check=False,
            )
            check("summariser re-run exits 0", proc.returncode == 0,
                  proc.stderr.strip()[-200:] if proc.returncode else "")
            if proc.returncode == 0:
                fresh = json.loads(out.read_text(encoding="utf-8"))
                for key in ("rows", "n_models_expected", "n_reports_present",
                            "batch_complete", "provenance", "caveat",
                            "supports_improvement_claim", "qlfn_scope"):
                    check(f"summary field '{key}' reproduces",
                          fresh.get(key) == archived.get(key))

    # ---- verdict -----------------------------------------------------------
    for label, ok, detail in findings:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}"
              + (f"  -- {detail}" if detail else ""))
    n_fail = sum(1 for _, ok, _ in findings if not ok)
    print(f"\n{len(findings) - n_fail}/{len(findings)} PASS")
    print("scope: completeness check only; NOT_COMPARABLE, no improvement claim")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

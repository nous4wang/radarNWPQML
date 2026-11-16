"""Locate and *statically* scan the upstream ``radar-nwp-bench`` repository.

The workstation this integration layer was written on has neither torch nor
numpy nor PyYAML. Importing the benchmark's own registry would pull in all
three, so the upstream repository is read as *data* instead: decorator names,
config files and JSON evaluation reports. Nothing here executes benchmark
code, and nothing here requires a third-party package.

This is deliberate. The integration layer answers "which baselines exist,
under what contract, and with what recorded evidence" -- questions about
metadata, not about tensors. Forward passes stay in the upstream harness.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_VAR = "RADAR_NWP_BENCH_ROOT"
DEFAULT_BENCH_ROOT = Path("R:/1radarNWPprediction/radar-nwp-bench")

#: A decorator at column zero. Anchoring on the start of the line excludes the
#: example inside ``models/__init__.py`` and the escaped mention inside the
#: ``physical_qpe_v2`` docstring, both of which are not real registrations.
_REGISTER = re.compile(r'^@register_model\(\s*["\']([a-z0-9_]+)["\']\s*\)', re.M)
_CLASS = re.compile(r"^class\s+(\w+)\s*\(", re.M)
_ANALYTICAL = re.compile(r"^\s*is_learnable\s*=\s*False\b", re.M)
_SCALAR = re.compile(r"^([a-z_]+):\s*(.*?)\s*$", re.M)
_TABLE_ROW = re.compile(r"^\|\s*([a-z0-9_]+)\s*\|")

_CONFIG_KEYS = ("paper", "repo", "license", "notes")


class BenchNotFound(RuntimeError):
    """The upstream benchmark repository could not be located."""


@dataclass(frozen=True)
class BenchSource:
    """A read-only handle on the upstream benchmark repository."""

    root: Path

    @classmethod
    def locate(cls, root: str | Path | None = None) -> "BenchSource":
        """Resolve the repository root from an argument, the environment, or default.

        Raises:
            BenchNotFound: when no candidate contains the expected layout.
        """
        candidates = [
            Path(root) if root is not None else None,
            Path(os.environ[ENV_VAR]) if ENV_VAR in os.environ else None,
            DEFAULT_BENCH_ROOT,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = candidate.expanduser()
            if (candidate / "models").is_dir() and (candidate / "docs" / "MASTER.md").is_file():
                return cls(candidate.resolve())
        raise BenchNotFound(
            f"radar-nwp-bench not found; set {ENV_VAR} or pass --bench-root. "
            f"Tried: {[str(c) for c in candidates if c is not None]}"
        )

    # -- layout -------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def config_dir(self) -> Path:
        return self.root / "configs" / "models"

    @property
    def runs_dir(self) -> Path:
        return self.root / ".workbuddy" / "runs"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    # -- scans --------------------------------------------------------------
    def scan_adapters(self) -> dict[str, dict[str, Any]]:
        """Map every registered name to its adapter file and class.

        Only ``models/<pkg>/model.py`` is scanned. Column-zero decorators are
        the registration site; the first class definition *after* one is the
        adapter the decorator binds to.
        """
        found: dict[str, dict[str, Any]] = {}
        for model_file in sorted(self.model_dir.glob("*/model.py")):
            text = model_file.read_text(encoding="utf-8")
            for match in _REGISTER.finditer(text):
                name = match.group(1)
                following = _CLASS.search(text, match.end())
                rel = model_file.relative_to(self.root).as_posix()
                record = {
                    "adapter": rel,
                    "class": following.group(1) if following else None,
                    "line": text.count("\n", 0, match.start()) + 1,
                    # Analytical baselines declare this at class scope. Read
                    # from the file rather than guessing from the group name.
                    "is_learnable": _ANALYTICAL.search(text) is None,
                }
                found.setdefault(name, record)
        return found

    def scan_configs(self) -> dict[str, dict[str, Any]]:
        """Read the descriptive scalar fields of each model config.

        The YAML files are not parsed as YAML: only four top-level scalar
        keys are needed, and a line parser keeps this module dependency-free.
        The ``model:`` section (constructor kwargs) is intentionally ignored.
        """
        found: dict[str, dict[str, Any]] = {}
        for config_file in sorted(self.config_dir.glob("*.yaml")):
            text = config_file.read_text(encoding="utf-8")
            record: dict[str, Any] = {
                "config": config_file.relative_to(self.root).as_posix()
            }
            for key, raw in _SCALAR.findall(text):
                if key not in _CONFIG_KEYS:
                    continue
                value = raw.strip().strip('"').strip("'")
                if value in ("null", "~", ""):
                    value = None
                record[key] = value
            found[config_file.stem] = record
        return found

    def scan_eval_reports(self, suffix: str = ".eval.json") -> dict[str, dict[str, Any]]:
        """Load every verification report with the given suffix."""
        found: dict[str, dict[str, Any]] = {}
        if not self.runs_dir.is_dir():
            return found
        for report in sorted(self.runs_dir.glob(f"*{suffix}")):
            name = report.name[: -len(suffix)]
            try:
                found[name] = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A truncated report is a data problem, not a reason to stop
                # the whole catalogue. The verifier reports the gap.
                found[name] = {}
        return found

    def score_aware_table(self) -> dict[str, dict[str, float]]:
        """Parse the authoritative score-aware matrix from RESULTS_FINAL.md §10.1.

        Reading the table rather than hard-coding eighteen numbers means an
        upstream correction is picked up the next time the catalogue is synced.
        """
        path = self.docs_dir / "RESULTS_FINAL.md"
        text = path.read_text(encoding="utf-8")
        start = text.find("## 10.1 全矩阵 score-aware 结果")
        if start < 0:
            return {}
        end = text.find("\n## ", start + 1)
        section = text[start : end if end > 0 else len(text)]
        header: list[str] | None = None
        rows: dict[str, dict[str, float]] = {}
        for line in section.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] == "model":
                header = cells
                continue
            if header is None or not _TABLE_ROW.match(line):
                continue
            values: dict[str, float] = {}
            for key, raw in zip(header[1:], cells[1:]):
                try:
                    values[key] = float(raw.replace(",", ""))
                except ValueError:
                    continue
            rows[cells[0]] = values
        return rows

    def git_revision(self) -> str:
        """Short HEAD hash of the upstream checkout, or ``"unknown"``."""
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def master_version(self) -> str:
        """Read the authoritative MASTER.md version stamp from its first lines."""
        head = (self.docs_dir / "MASTER.md").read_text(encoding="utf-8")[:600]
        match = re.search(r"v\d+\.\d+", head)
        return match.group(0) if match else "unknown"

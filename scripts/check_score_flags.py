#!/usr/bin/env python3
"""Check that the scoring flags survive the trip to the evaluator.

The flags that have to travel from the training stage to the scoring stage
are ``--no-nwp`` and ``--model-kwargs``.  They used to travel as a
free-form remainder, and ``argparse.REMAINDER`` collects *everything* after
the first flag -- so a second flag arrived at the evaluator as a stray
token and the evaluator rejected it.

This check calls the flag builder the same way the batch script does and
prints what the evaluator would receive.  It is a local check, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "tools" / "score_epoch_curve.py"


def load():
    spec = importlib.util.spec_from_file_location("sec", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load()
    failures = 0

    cases = [
        ([], []),
        (["--no-nwp"], ["--no-nwp"]),
        (["--model-kwargs", json.dumps({"motion_mode": "none"})],
         ["--model-kwargs", json.dumps({"motion_mode": "none"})]),
    ]

    for argv, expected in cases:
        args = module.parse_args([
            "--model", "qlfn2",
            "--checkpoint-dir", "/tmp",
            "--data-root", "/tmp",
            "--out-dir", "/tmp",
            *argv,
        ])
        flags = module.evaluator_flags(args)
        ok = flags == expected
        print(f"{'ok  ' if ok else 'FAIL'} argv={argv} -> flags={flags}")
        if not ok:
            print(f"     expected {expected}")
            failures += 1

    print(f"\n{len(cases) - failures}/{len(cases)} flag cases passed")
    return failures


if __name__ == "__main__":
    sys.exit(main())

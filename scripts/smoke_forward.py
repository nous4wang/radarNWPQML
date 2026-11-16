"""GPU forward-contract smoke test for every wired baseline.

This is the runtime companion to the metadata-level catalogue. It answers a
question that static scanning cannot: with the real package importable, does
each registered adapter construct, accept the declared input tensors, and
return the declared output shape?

    # benchmark server (needs torch + the live upstream checkout)
    RADAR_NWP_LIVE_ROOT=/home/wf/radarNWP \
        /home/wf/.conda/envs/radar_nwp/bin/python \
        scripts/smoke_forward.py

    # workstation without torch: reports SKIP, exits 0
    python scripts/smoke_forward.py

It does **not** train, load checkpoints or compute metrics. The input patch is
64x64, the same patch size the upstream training loop uses. One extra check
exercises the ablation seam: ``x_nwp=None`` must be resolved to zeros by
``BasePrecipitationModel.__call__`` and must not change the output shape.

The declared contract comes from ``configs/baselines.json`` (the CONTRACT
written by ``sync_baselines.py``); the expected shapes below are its tensor
form and are stated explicitly so a mismatch is attributable.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# (B, T_in, C_radar, H, W) -> (B, T_out, 1, H, W)
EXPECTED_X_RADAR = (1, 11, 10, 64, 64)
EXPECTED_X_NWP = (1, 2, 27, 64, 64)
EXPECTED_Y = (1, 10, 1, 64, 64)
ABLATION_MODEL = "simvp_v2"


class Timeout(Exception):
    """Raised when one model's forward pass exceeds its budget."""


def _alarm(signum, frame) -> None:  # noqa: ANN001 - signal handler signature
    raise Timeout("forward exceeded the per-model time budget")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default=os.environ.get("RADAR_NWP_LIVE_ROOT"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seconds", type=int, default=90,
                    help="per-model forward budget")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "smoke_forward.json"))
    args = ap.parse_args()

    if not args.live_root:
        print("SKIP: RADAR_NWP_LIVE_ROOT is not set (no live upstream package).")
        return 0

    live_root = Path(args.live_root).resolve()
    sys.path.insert(0, str(live_root))
    try:
        import torch
        import yaml
        from radar_nwp_bench.models.registry import build_model, list_models
    except Exception as exc:  # noqa: BLE001 - a missing torch is a SKIP
        print(f"SKIP: cannot import the runtime stack: {type(exc).__name__}: {exc}")
        return 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    catalog_path = PROJECT_ROOT / "configs" / "baselines.json"
    catalogue = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalogued = sorted(row["name"] for row in catalogue["baselines"] if row["adapter"])
    runtime = list_models()
    names = sorted(set(catalogued) | set(runtime))

    config_dir = live_root / "configs" / "models"
    records: list[dict] = []
    print(f"live root : {live_root}")
    print(f"device    : {device}  patch=64  budget={args.seconds}s/model")
    print(f"models    : {len(names)} ({len(runtime)} registered, {len(catalogued)} catalogued)")
    print("")

    old_handler = signal.signal(signal.SIGALRM, _alarm)
    try:
        for name in names:
            record: dict = {
                "name": name,
                "in_registry": name in runtime,
                "in_catalogue": name in catalogued,
                "built": False,
                "forward_ok": False,
                "shape_ok": False,
                "finite": False,
                "params": None,
                "seconds": None,
                "shape": None,
                "error": None,
            }
            started = time.monotonic()
            model = None
            try:
                yaml_path = config_dir / f"{name}.yaml"
                kwargs = {}
                if yaml_path.is_file():
                    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                    kwargs = cfg.get("model") or {}
                model = build_model(name, **kwargs)
                model = model.to(device).eval()
                record["built"] = True
                record["params"] = sum(p.numel() for p in model.parameters())

                x_radar = torch.zeros(*EXPECTED_X_RADAR, device=device, dtype=torch.float32)
                x_nwp = torch.zeros(*EXPECTED_X_NWP, device=device, dtype=torch.float32)
                signal.alarm(args.seconds)
                with torch.no_grad():
                    y = model(x_radar, x_nwp)
                signal.alarm(0)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

                record["shape"] = list(y.shape)
                record["shape_ok"] = tuple(y.shape) == EXPECTED_Y
                record["finite"] = bool(torch.isfinite(y).all().item())
                record["forward_ok"] = record["shape_ok"] and record["finite"]
            except Exception as exc:  # noqa: BLE001 - one model must not abort the sweep
                signal.alarm(0)
                record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                record["seconds"] = round(time.monotonic() - started, 2)
                del model
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            records.append(record)
            status = "PASS" if record["forward_ok"] else "FAIL"
            detail = "" if record["forward_ok"] else f"  {record['error'] or record['shape']}"
            print(f"[{status}] {name:<24} shape={record['shape']} "
                  f"params={record['params']} {record['seconds']}s{detail}")
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    # The ablation seam: x_nwp=None must become an all-zero tensor at the
    # base class, not leak into the adapter. Tested on one representative.
    ablation: dict = {"model": ABLATION_MODEL, "checked": False, "ok": False,
                      "shape": None, "error": None}
    try:
        yaml_path = config_dir / f"{ABLATION_MODEL}.yaml"
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        probe = build_model(ABLATION_MODEL, **(cfg.get("model") or {})).to(device).eval()
        x = torch.zeros(*EXPECTED_X_RADAR, device=device, dtype=torch.float32)
        with torch.no_grad():
            y = probe(x, None)
        ablation.update({"checked": True, "shape": list(y.shape),
                         "ok": tuple(y.shape) == EXPECTED_Y})
        del probe, x, y
    except Exception as exc:  # noqa: BLE001
        ablation["error"] = f"{type(exc).__name__}: {exc}"[:400]
    print(f"\nablation x_nwp=None on {ABLATION_MODEL}: "
          + ("PASS" if ablation["ok"] else f"FAIL {ablation['error'] or ablation['shape']}"))

    built = sum(r["built"] for r in records)
    passed = sum(r["forward_ok"] for r in records)
    failed = [r["name"] for r in records if not r["forward_ok"]]
    payload = {
        "meta": {
            "live_root": str(live_root),
            "device": device,
            "torch": torch.__version__,
            "patch": [EXPECTED_X_RADAR, EXPECTED_X_NWP, EXPECTED_Y],
            "per_model_budget_seconds": args.seconds,
            "catalogue_contract": catalogue.get("contract"),
        },
        "ablation": ablation,
        "summary": {
            "total": len(records),
            "built": built,
            "forward_ok": passed,
            "failed": failed,
        },
        "models": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n{built}/{len(records)} constructed, {passed}/{len(records)} forward OK")
    print(f"failed: {failed}")
    print(f"wrote {out}")
    # A catalogue/registry mismatch or a failed ablation is a real defect;
    # an adapter that cannot forward is reported but does not fail the run,
    # because several baselines are analytical and need real data semantics.
    registry_ok = set(catalogued) == set(runtime)
    return 0 if registry_ok and ablation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

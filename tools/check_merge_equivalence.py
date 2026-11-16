"""证明 2026-11-11 的合并是「融合」而不是「重写」。

把 ``docs/archive/qlfn2_pre_merge/`` 的十个旧文件挂成一个临时包
``qlfn2_legacy``，与新实现 ``qlfn2.qlfn2_model`` 在同一份权重下跑同一批
合成雷达，逐位比较输出。

判据（先写后跑）
----------------
1. 关掉新增的两条 NWP 通路（``use_halo=False, use_nwp=False``）后，
   新旧 state_dict 的**键集合必须完全相同**；
2. 把旧权重灌进新模型，``rain`` / ``rain_t0`` / ``rain_history`` 的
   ``max|diff|`` 必须为 **0.000e+00**（逐位相同，不是"接近"）；
3. 诊断量（平流向量、算子 a/b、相态权重）必须完全一致；
4. 打开 NWP 通路后，参数量增量必须等于
   ``halo_proj(20*40+40) + nwp_gate(27*16+16+16*1+1)``。

只要第 2 条成立，"合并改变了模型数学"这个担心就可以排除；剩下的差异
全部来自新接的 NWP，而那正是本次要检验的机制。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEGACY = ROOT / "docs" / "archive" / "qlfn2_pre_merge"


def load_legacy():
    pkg = types.ModuleType("qlfn2_legacy")
    pkg.__path__ = [str(LEGACY)]
    sys.modules["qlfn2_legacy"] = pkg
    import importlib
    return importlib.import_module("qlfn2_legacy.model")


def synthetic_radar(b=2, t=11, size=32):
    torch.manual_seed(7)
    x = torch.zeros(b, t, 10, size, size)
    yy = torch.linspace(0, 1, size).view(1, 1, 1, size)
    xx = torch.linspace(0, 1, size).view(1, 1, size, 1)
    blob = torch.exp(-(((xx - 0.4) ** 2 + (yy - 0.6) ** 2) / 0.08))
    for k in range(t):
        x[:, k, 6] = 300.0 * blob * (0.7 + 0.04 * k)
        x[:, k, 7] = 78.0 * blob
        x[:, k, 9] = 180.0 * blob
        for i in range(6):
            x[:, k, i] = (520.0 - 60.0 * (i + 2)) * blob * (0.7 + 0.04 * k)
        x[:, k, 8] = 300.0 * blob
    return x


def main() -> int:
    from qlfn2.qlfn2_model import QLFN2Model as NewModel

    old_mod = load_legacy()
    kw = dict(in_len=11, out_len=10, use_history_head=True,
              width=8, motion_mode="corr", max_shift=8)
    torch.manual_seed(123)
    old = old_mod.QLFN2Model(use_halo=False, **kw).eval()
    torch.manual_seed(999)
    new = NewModel(use_halo=False, use_nwp=False, **kw).eval()

    old_keys, new_keys = set(old.state_dict()), set(new.state_dict())
    failures = 0
    print(f"legacy parameters  : {sum(p.numel() for p in old.parameters())}")
    print(f"merged parameters  : {sum(p.numel() for p in new.parameters())} "
          f"(NWP paths off)")
    print(f"state_dict keys    : legacy {len(old_keys)}, merged {len(new_keys)}, "
          f"symmetric diff {len(old_keys ^ new_keys)}")
    if old_keys != new_keys:
        print("FAIL: the key sets differ ->", sorted(old_keys ^ new_keys)[:10])
        failures += 1

    new.load_state_dict(old.state_dict(), strict=True)
    x = synthetic_radar()
    with torch.no_grad():
        a, b = old(x), new(x)

    def cmp(name, u, v):
        nonlocal failures
        d = float((u - v).abs().max())
        print(f"{name:<34} max|diff| = {d:.3e}")
        if d != 0.0:
            failures += 1
        return d

    print()
    cmp("rain (B,10,1,H,W)", a.rain, b.rain)
    cmp("rain_t0", a.rain_t0, b.rain_t0)
    cmp("rain_history", a.rain_history, b.rain_history)

    diag_ok = True
    for key in ("motion_vx", "motion_vy", "operator_a", "operator_b",
                "phase_weights_mean"):
        if a.diagnostics.get(key) != b.diagnostics.get(key):
            print(f"diagnostic {key}: {a.diagnostics.get(key)} != "
                  f"{b.diagnostics.get(key)}")
            diag_ok = False
    print(f"diagnostics identical    : {diag_ok}")
    if not diag_ok:
        failures += 1

    # -- the new NWP paths: exact parameter delta -------------------------
    torch.manual_seed(999)
    wired = NewModel(use_halo=True, use_nwp=True, **kw)
    delta = (sum(p.numel() for p in wired.parameters())
             - sum(p.numel() for p in new.parameters()))
    expected = 20 * 40 + 40 + (27 * 16 + 16) + (16 * 1 + 1)
    print()
    print(f"NWP wiring parameters    : {delta} (expected {expected})")
    if delta != expected:
        failures += 1
    print(f"wired model parameters   : "
          f"{sum(p.numel() for p in wired.parameters())}")

    # zero-init => the wiring is the identity at the training start
    torch.manual_seed(999)
    baseline = NewModel(use_halo=False, use_nwp=False, **kw).eval()
    wired.load_state_dict(
        {k: v for k, v in baseline.state_dict().items()
         if k in wired.state_dict()}, strict=False)
    nwp = torch.rand(2, 2, 27, 32, 32)
    with torch.no_grad():
        r_base = baseline(x).rain
        r_wired = wired(x, nwp).rain
    d = float((r_base - r_wired).abs().max())
    print(f"zero-init identity       : max|diff| = {d:.3e}")
    if d != 0.0:
        failures += 1

    print()
    print(f"RESULT: {'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failure(s))")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())

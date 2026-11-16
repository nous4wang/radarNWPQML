"""qlfn2 · 结构化 CAPPI 垂直算子 + 平流对齐的 NWP 上游窄环带。

2026-11-11 起本包只有**两个实现文件**（`checks.py` 是自检）：

* ``qlfn2_model.py``  —— geometry / units / cappi / operator / motion /
  halo / model / adapter 全部内联的自包含模型，含 NWP 真连线；
* ``qlfn2_losses.py`` —— 阈值对数居中对齐 + 逐时效衰减的
  ``ThresholdAwareLoss``。

另有 **3 个纯别名 shim**（``adapter.py`` / ``model.py`` / ``losses.py``），
每个只做 ``from .qlfn2_model ...`` / ``from .qlfn2_losses ...`` 转发，
不含任何实现，用于兼容合并前外部脚本的 import 路径。
``check_compat_aliases`` 会验证它们指向同一个对象，防止退化成第二份实现。

合并前的十个模块已移到 ``docs/archive/qlfn2_pre_merge/``，只作证据，
不在 import 路径上。设计依据与全部实测事实只写在项目根
``MASTER_CONTROL_ROOM.md``。
"""
from __future__ import annotations

from .qlfn2_model import (
    ModelOutput, QLFN2Adapter, QLFN2Model, add_bench_to_path,
    register_qlfn2, require_bench,
)
from .qlfn2_losses import LossBreakdown, ThresholdAwareLoss, threshold_logits

__all__ = [
    "ModelOutput", "QLFN2Adapter", "QLFN2Model",
    "LossBreakdown", "ThresholdAwareLoss", "threshold_logits",
    "add_bench_to_path", "require_bench", "register_qlfn2",
    "qlfn2_model", "qlfn2_losses", "checks",
]

VERSION = "qlfn2-0.3-merged"

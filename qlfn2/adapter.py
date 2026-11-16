"""向后兼容别名（shim）—— 唯一实现在 ``qlfn2_model.py``。

合并前外部脚本可能写 ``from qlfn2.adapter import QLFN2Adapter``。
本文件只做转发，不含任何实现，也不定义任何新的类；两个自包含实现文件
仍是唯一真相源。``qlfn2.checks.check_compat_aliases`` 会验证别名与实现
是**同一个对象**，防止这里退化成第二份实现。
"""
from .qlfn2_model import (
    QLFN2Adapter, add_bench_to_path, register_qlfn2, require_bench,
)

__all__ = ["QLFN2Adapter", "register_qlfn2", "add_bench_to_path",
           "require_bench"]

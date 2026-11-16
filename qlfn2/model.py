"""向后兼容别名（shim）—— 唯一实现在 ``qlfn2_model.py``。

合并前外部脚本可能写 ``from qlfn2.model import QLFN2Model``。
本文件只做转发，不含任何实现。
"""
from .qlfn2_model import ModelOutput, QLFN2Model

__all__ = ["QLFN2Model", "ModelOutput"]

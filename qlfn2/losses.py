"""向后兼容别名（shim）—— 唯一实现在 ``qlfn2_losses.py``。

合并前外部脚本可能写 ``from qlfn2.losses import ThresholdAwareLoss``。
本文件只做转发，不含任何实现；``__all__`` 直接取自实现模块。
"""
from .qlfn2_losses import *  # noqa: F401,F403
from .qlfn2_losses import __all__ as _impl_all

__all__ = list(_impl_all)

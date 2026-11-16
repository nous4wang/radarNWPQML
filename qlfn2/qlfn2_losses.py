"""qlfn2_losses.py · 面向分数的判别式损失（阈值对齐 / 时效衰减均已修复）。

2026-11-11 第一轮修复（用户审查第 3 条）
----------------------------------------
    修复前：score = torch.log(p)
    修复后：score = torch.log(p) - math.log(thr)

Sigmoid 的判定分水岭是 ``score == 0``；``log(p)`` 在 ``p == 1.0 mm`` 处
恰好为 0，会把所有阈值档的分类边界钉死在 1.0 mm。减去 ``log(thr)`` 后
``p == thr`` 处 score 严格为 0，四档各自拥有自己的决策边界。

2026-11-11 第二轮修复（用户审查第 1、2 条，均经实测复现）
----------------------------------------------------------
**(1) 5D 广播陷阱 —— 静默的通道轴裂变。** 原实现把逐时效权重固定写成
``decay.view(1, t, 1, 1)``。上游模型输出契约是 ``(B, T, 1, H, W)``，
与 4 维的 ``decay`` 右对齐广播时，``T`` 落在**通道维**上，而通道维大小为
1 与 T 是"可广播"的 ⇒ 中间张量静默膨胀成 ``(B, T, T, H, W)``，通道轴
裂变、显存与计算量 ×T，且 loss 数值被改掉。实测（t=10, decay=0.2）：

    4D (2,10,8,8)      -> 0.114381   （正确值）
    5D (2,10,1,8,8)    -> 0.114000   （错误值，静默通过）
    5D (2,10,3,8,8)    -> RuntimeError（通道维 3 与 10 不兼容才报错）

修法：``decay_shape = [1, t] + [1] * (pred.dim() - 2)``，让时效轴永远
落在 ``dim=1``，无论输入是 4D 还是 5D；修复后两者逐位相同。

**(2) lead_decay 指数方向写反。** 原 ``decay ** (t-1-k)`` 使 k=0（第一个
时效）权重为 ``decay**(t-1)``、k=t-1 权重为 1.0 —— 越近的时效权重越小。
实测 t=10, decay=0.2 时为 ``[5e-7, ..., 0.2, 1.0]``，前 9 个时效的梯度
几乎被完全抹掉，与"预报越远越不可信、应降权"的物理含义相反。
改为 ``decay ** (k / max(t-1, 1))``：k=0 为 1.0，k=t-1 为 ``decay``，
单调衰减。默认 ``lead_decay=1.0`` 时全为 1，新旧逐位一致，**不改变既有
训练配方**。

保留的三个性质
--------------
* 幅值项：泊松负对数似然 ``p - y*log(p)``，保证量级定标不被事件项带偏；
* 事件项：多阈值 BCE，正例权重可由 batch 内的正负比自动推导；
* 逐时效加权：``reduction="none"`` 后逐元素乘 ``decay`` 再求均值
  （原式 ``bce * decay.mean()`` 会让 ``lead_decay`` 退化成标量）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LossBreakdown", "ThresholdAwareLoss", "threshold_logits",
           "lead_weights"]


@dataclass
class LossBreakdown:
    total: torch.Tensor
    amount: torch.Tensor
    event: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "loss_amount": float(self.amount.detach()),
            "loss_event": float(self.event.detach()),
        }


def threshold_logits(pred: torch.Tensor, thr: float) -> torch.Tensor:
    """把预测雨量转成以 ``thr`` 为中心对齐的 logit。

    ``pred == thr`` ⇒ 返回 0 ⇒ sigmoid = 0.5，与该档的判定分水岭一致。
    ``pred > thr`` ⇒ 正数 ⇒ 正例；``pred < thr`` ⇒ 负数 ⇒ 负例。
    """
    p = pred.clamp(min=1e-4)
    return torch.log(p) - math.log(float(thr))


def lead_weights(t: int, decay: float) -> list[float]:
    """逐时效权重：第一个时效 1.0，最后一个平滑降到 ``decay``。

    单独暴露成纯函数，是为了让 ``qlfn2.checks`` 直接**测量**方向，
    而不是靠读代码推断。t <= 1 时返回单个 1.0。
    """
    if t < 1:
        raise ValueError("t must be >= 1")
    if t == 1:
        return [1.0]
    d = float(decay)
    return [d ** (k / (t - 1)) for k in range(t)]


class ThresholdAwareLoss(nn.Module):
    """泊松幅值损失 + 逐阈值对齐的加权事件损失。

    Args:
        thresholds: 降雨阈值，单位 mm / 6 min。2.0 是承重的那一档。
        weight_amount / weight_event: 两项权重。
        event_positive_weight: 正例权重；None 表示按 batch 内正负比推导。
        lead_decay: 逐时效折扣，第一个时效为 1.0、最后一个为其值，
            中间按 ``decay**(k/(t-1))`` 平滑衰减。

    ``pred`` / ``target`` 允许 4D ``(B, T, H, W)`` 或 5D
    ``(B, T, 1, H, W)``（上游模型输出契约），两者结果逐位相同。
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        weight_amount: float = 0.5,
        weight_event: float = 1.0,
        event_positive_weight: float | None = None,
        lead_decay: float = 1.0,
    ) -> None:
        super().__init__()
        if not thresholds or any(float(t) <= 0.0 for t in thresholds):
            raise ValueError("thresholds must be non-empty and strictly positive")
        self.thresholds = tuple(float(t) for t in thresholds)
        self.weight_amount = float(weight_amount)
        self.weight_event = float(weight_event)
        self.event_positive_weight = event_positive_weight
        self.lead_decay = float(lead_decay)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> LossBreakdown:
        """计算损失；支持 4D 与 5D 输入，且两者逐位一致。"""
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.dim() < 2:
            raise ValueError(
                f"need at least (B, T, ...) tensors, got {tuple(pred.shape)}")
        t = pred.shape[1]
        # 时效轴恒在 dim=1；其余维全部为 1。写死 (1,t,1,1) 会在 5D 输入下
        # 把 T 对到通道轴，静默把 1 个通道广播成 T 个通道。
        decay_shape = [1, t] + [1] * (pred.dim() - 2)
        decay = torch.tensor(
            lead_weights(t, self.lead_decay),
            dtype=pred.dtype, device=pred.device,
        ).view(*decay_shape)

        p = pred.clamp(min=1e-4)
        y = target.clamp(min=0.0)

        # 1. 幅值项：泊松负对数似然（保住降水量级定标）
        amount = ((p - y * torch.log(p)) * decay).mean()

        # 2. 事件项：每个阈值单独做对数居中对齐
        event = pred.new_zeros(())
        for thr in self.thresholds:
            score = threshold_logits(pred, thr)
            pos_weight = self.event_positive_weight
            if pos_weight is None:
                n_pos = (y >= thr).sum()
                n_neg = (y < thr).sum()
                pos_weight = (float((n_neg / n_pos.clamp(min=1)).item())
                              if n_pos > 0 else 1.0)
            bce = F.binary_cross_entropy_with_logits(
                score,
                (y >= thr).to(p.dtype),
                pos_weight=torch.tensor(pos_weight, dtype=p.dtype,
                                        device=p.device),
                reduction="none",
            )
            # 逐元素加权后求均值，lead_decay 才真正按时效起作用
            event = event + (bce * decay).mean()
        event = event / len(self.thresholds)

        total = self.weight_amount * amount + self.weight_event * event
        return LossBreakdown(total=total, amount=amount, event=event)

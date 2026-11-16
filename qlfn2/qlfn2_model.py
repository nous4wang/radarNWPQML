"""qlfn2_model.py · qlfn2 的唯一模型实现（自包含）。

2026-11-11: 八模块合并为自包含文件 + 修掉 F30 NWP 断联。
2026-11-14 (F36): 修复 D1 通道错位 (2,3,0,14,23)、D2 逐样本掩码、D3 逐样本半平面、M1 正交开关。
2026-11-15 (F37): 针对 F36 逐时效剖面揭示的「t0 初值腰斩 (-0.1239) + 第二片 NWP 未消费」，
新增 t>0 拉格朗日演进双通路（通过 --model-kwargs 开启，与 t0 调制正交）：
1. ``extract_nwp_evolution_tensor``：同时消费 ``x_nwp[:, 0]``（issue_hour）与
   ``x_nwp[:, 1] - x_nwp[:, 0]``（forecast_hour 1h 演变趋势），提取 10 个核心动力/热力通道拼成 20 通道空间场；
2. ``use_nwp_vel``：预测空间慢变的平流速度修正场 (dvx, dvy)，按 w(k) = k*(k+1)/20 作用在 k>=1 (lead 1..9)，
   lead 0 (+6min) 权重严格为 0，100% 锁死纯雷达 lead 0 高起点；
3. ``use_nwp_source``：预测空间零均值 (zero-mean) 的对流生消项，按 tau(k) = k/9 仅在平流后的湿区包络内生效，
   lead 0 权重严格为 0，且结构上禁止全局压低或抬高全图平均雨量。
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # geometry
    "RADAR_SHAPE", "NWP_SHAPE", "NWP_TO_RADAR_SCALE", "RADAR_VARS",
    "CAPPI_LEVELS_KM", "HALO_EXPECTED", "HaloGeometry",
    "bilinear_upsample", "center_crop", "nwp_to_radar",
    "symmetric_patch_origins",
    # units
    "ARCHIVE_SCALE", "DBZ_VARS", "OTHER_VARS", "net_to_physical",
    "physical_to_net", "net_to_archive", "archive_to_net",
    "archive_to_physical", "physical_to_archive",
    # cappi
    "CAPPI_INDICES", "CR_INDEX", "ET_INDEX", "HBR_INDEX", "VIL_INDEX",
    "VERTICAL_QUANTITY_NAMES", "VERTICAL_SCALES", "VerticalQuantities",
    "vertical_quantities", "scaled_vertical_quantities", "phase_inputs",
    # operator
    "PHASE_STATES", "OperatorOutput", "PhaseRouter",
    "PhaseRoutedZROperator", "StructuredRainHead",
    # motion
    "Motion", "MotionField", "ECHO_THRESHOLD_DBZ", "CHAIN_STEPS",
    "estimate_motion_field", "estimate_motion", "warp", "advect_sequence",
    # halo & NWP evolution
    "NWP_VARS", "HaloMasks", "HALO_CONTEXT_VARS", "HALO_CONTEXT_INDICES",
    "NWP_EVOL_VARS", "NWP_EVOL_INDICES", "extract_nwp_evolution_tensor",
    "build_masks", "reachable_distance_px", "summarise", "build_halo_context",
    # NWP wiring
    "nwp_context_from_patch", "nwp_context_from_native", "build_nwp_context",
    # model / seam
    "ModelOutput", "QLFN2Model", "QLFN2Adapter",
    "BENCH_CANDIDATES", "add_bench_to_path", "require_bench",
    "register_qlfn2",
]


# =============================================================================
# 1. geometry —— 网格与配准的唯一真相源
# =============================================================================
RADAR_SHAPE: tuple[int, int] = (301, 301)
NWP_SHAPE: tuple[int, int] = (167, 167)
NWP_TO_RADAR_SCALE: int = 3
RADAR_RES_DEG: float = 0.01
NWP_RES_DEG: float = 0.03
CAPPI_LEVELS_KM: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
RADAR_VARS: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
    "CR", "ET", "HBR", "VIL",
)
HALO_EXPECTED: dict[str, float] = {
    "crop_px_per_side": 100.0,
    "halo_native_cells_per_side": 100.0 / 3.0,
    "halo_degrees_per_side": 1.0,
    "halo_area_fraction": 1.0 - (301 * 301) / (501 * 501),
}


@dataclass(frozen=True)
class HaloGeometry:
    native: int = NWP_SHAPE[0]
    scale: int = NWP_TO_RADAR_SCALE
    radar: int = RADAR_SHAPE[0]

    @property
    def upsampled(self) -> int:
        return self.native * self.scale

    @property
    def crop(self) -> int:
        return (self.upsampled - self.radar) // 2

    @property
    def halo_native_cells_per_side(self) -> float:
        return self.crop / self.scale

    @property
    def halo_degrees_per_side(self) -> float:
        return self.halo_native_cells_per_side * NWP_RES_DEG

    @property
    def halo_area_fraction(self) -> float:
        total = self.upsampled * self.upsampled
        centre = self.radar * self.radar
        return 1.0 - centre / total

    def describe(self) -> dict:
        return {
            "native": self.native, "scale": self.scale, "radar": self.radar,
            "upsampled": self.upsampled, "crop_px_per_side": self.crop,
            "halo_native_cells_per_side": self.halo_native_cells_per_side,
            "halo_degrees_per_side": self.halo_degrees_per_side,
            "halo_area_fraction": self.halo_area_fraction,
        }


def bilinear_upsample(field: torch.Tensor, scale: int = NWP_TO_RADAR_SCALE):
    """复现上游 ``coordinates._bilinear_upsample``：reflect 边界 + 像素半偏置。"""
    if field.shape[-2:] != NWP_SHAPE:
        raise ValueError(f"expected NWP field {NWP_SHAPE}, got {tuple(field.shape[-2:])}")
    squeeze = field.dim() == 2
    x = field[None] if squeeze else field
    h, w = NWP_SHAPE
    dev = x.device

    def coords(n: int):
        dst = torch.arange(n * scale, device=dev, dtype=torch.float64)
        src = (dst + 0.5) / scale - 0.5
        i0 = torch.floor(src)
        weight = (src - i0).to(torch.float32)
        period = 2 * (n - 1)
        i0r = torch.remainder(i0, period)
        i0r = torch.where(i0r >= n, period - i0r, i0r)
        i1r = torch.remainder(i0 + 1, period)
        i1r = torch.where(i1r >= n, period - i1r, i1r)
        return i0r.long(), i1r.long(), weight

    r0, r1, rw = coords(h)
    c0, c1, cw = coords(w)
    xf = x.float()
    row0, row1 = xf[..., r0, :], xf[..., r1, :]
    top = row0[..., c0] * (1 - cw) + row0[..., c1] * cw
    bot = row1[..., c0] * (1 - cw) + row1[..., c1] * cw
    rw_col = rw.view(*([1] * (top.dim() - 2)), -1, 1)
    out = top * (1 - rw_col) + bot * rw_col
    return out[0] if squeeze else out


def center_crop(field: torch.Tensor, radar: int = RADAR_SHAPE[0]):
    up = field.shape[-1]
    if up < radar:
        raise ValueError(f"cannot crop {radar} out of {up}")
    r0 = (field.shape[-2] - radar) // 2
    c0 = (field.shape[-1] - radar) // 2
    return field[..., r0:r0 + radar, c0:c0 + radar]


def nwp_to_radar(field: torch.Tensor):
    return center_crop(bilinear_upsample(field))


def symmetric_patch_origins(grid_dim: int, patch_size: int,
                            nominal_stride: int) -> tuple[int, ...]:
    if patch_size >= grid_dim:
        return (0,)
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    return tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))


# =============================================================================
# 2. units —— 网络压缩尺度 <-> 档案原始单位的量纲桥（严格互逆）
# =============================================================================
ARCHIVE_SCALE: float = 10.0
NET_DIVISOR_DBZ: float = 6.0
NET_DIVISOR_OTHER: float = 4.0
DBZ_VARS: tuple[str, ...] = (
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70", "CR", "HBR",
)
OTHER_VARS: tuple[str, ...] = ("ET", "VIL")


def _divisor_vector(x: torch.Tensor,
                    var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    if x.shape[-3] != len(var_names):
        raise ValueError(f"expected {len(var_names)} channels, got {x.shape[-3]}")
    values = [NET_DIVISOR_DBZ if n in DBZ_VARS else NET_DIVISOR_OTHER
              for n in var_names]
    div = torch.tensor(values, dtype=x.dtype, device=x.device)
    return div.view(*([1] * (x.dim() - 3)), -1, 1, 1)


def net_to_physical(x: torch.Tensor,
                    var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return torch.expm1(x.clamp(min=0.0) * _divisor_vector(x, var_names))


def physical_to_net(x: torch.Tensor,
                    var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return torch.log1p(x.clamp(min=0.0)) / _divisor_vector(x, var_names)


def net_to_archive(x: torch.Tensor,
                   var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return net_to_physical(x, var_names) * ARCHIVE_SCALE


def archive_to_net(x: torch.Tensor,
                   var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return physical_to_net(x / ARCHIVE_SCALE, var_names)


def archive_to_physical(x: torch.Tensor,
                        var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return x / ARCHIVE_SCALE


def physical_to_archive(x: torch.Tensor,
                        var_names: tuple[str, ...] = RADAR_VARS) -> torch.Tensor:
    return x * ARCHIVE_SCALE


# =============================================================================
# 3. cappi —— 六个高度层 -> 十个具名垂直量
# =============================================================================
CAPPI_INDICES: tuple[int, ...] = tuple(
    RADAR_VARS.index(f"CAP{d}") for d in (20, 30, 40, 50, 60, 70))
CR_INDEX: int = RADAR_VARS.index("CR")
ET_INDEX: int = RADAR_VARS.index("ET")
HBR_INDEX: int = RADAR_VARS.index("HBR")
VIL_INDEX: int = RADAR_VARS.index("VIL")

VERTICAL_QUANTITY_NAMES: tuple[str, ...] = (
    "cap_top_km", "cap_top_dbz", "cap_max_dbz", "cap_argmax_km",
    "cap_bright_dbz", "cap_nz_levels", "cap_col_dbz_sum",
    "cap_slope_dbz_km", "cap_et_gap_km", "cap_low_dbz",
)
VERTICAL_SCALES: tuple[float, ...] = (
    8.0, 60.0, 60.0, 8.0, 40.0, 6.0, 300.0, 10.0, 6.0, 60.0,
)


@dataclass
class VerticalQuantities:
    values: torch.Tensor
    levels_km: tuple[float, ...] = CAPPI_LEVELS_KM

    def __getitem__(self, name: str) -> torch.Tensor:
        i = VERTICAL_QUANTITY_NAMES.index(name)
        return self.values[:, i:i + 1]

    @property
    def n_quantities(self) -> int:
        return len(VERTICAL_QUANTITY_NAMES)


def vertical_quantities(radar: torch.Tensor) -> VerticalQuantities:
    if radar.dim() != 4:
        raise ValueError(f"expected (B, 10, H, W), got {tuple(radar.shape)}")
    if radar.shape[1] < len(RADAR_VARS):
        raise ValueError(f"expected {len(RADAR_VARS)} channels, got {radar.shape[1]}")
    levels = torch.tensor(CAPPI_LEVELS_KM, dtype=radar.dtype, device=radar.device)
    cap = radar[:, CAPPI_INDICES] / 10.0
    et_km = radar[:, ET_INDEX] / 10.0
    present = cap > 0.0
    masked = torch.where(present, cap, torch.zeros_like(cap))
    n_levels = present.sum(dim=1).to(radar.dtype)
    col_max = masked.max(dim=1).values
    col_sum = masked.sum(dim=1)
    low_dbz = cap[:, 0]
    idx = torch.arange(cap.shape[1], dtype=radar.dtype, device=radar.device)
    idx = idx.view(1, -1, 1, 1).expand_as(cap)
    top_idx = torch.where(present, idx, torch.full_like(idx, -1.0)).max(dim=1).values
    has_echo = top_idx >= 0
    top_km = torch.where(has_echo, top_idx + 2.0, torch.zeros_like(top_idx))
    argmax_km = cap.argmax(dim=1).to(radar.dtype) + 2.0
    bright = col_max - low_dbz
    centred = levels - levels.mean()
    denom = float((centred ** 2).sum())
    slope = (masked * centred.view(1, -1, 1, 1)).sum(dim=1) / denom
    et_gap = et_km - top_km
    top_dbz = torch.gather(
        cap, 1, top_idx.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    top_dbz = torch.where(has_echo, top_dbz, torch.zeros_like(top_dbz))
    stacked = torch.cat([
        top_km.unsqueeze(1), top_dbz.unsqueeze(1), col_max.unsqueeze(1),
        argmax_km.unsqueeze(1), bright.unsqueeze(1), n_levels.unsqueeze(1),
        col_sum.unsqueeze(1), slope.unsqueeze(1), et_gap.unsqueeze(1),
        low_dbz.unsqueeze(1),
    ], dim=1)
    return VerticalQuantities(values=stacked)


def scaled_vertical_quantities(radar: torch.Tensor) -> torch.Tensor:
    squeeze = radar.dim() == 4
    x = radar[None] if squeeze else radar
    values = vertical_quantities(x.reshape(-1, *x.shape[2:])).values
    values = values.reshape(*x.shape[:2], *values.shape[1:])
    scales = torch.tensor(VERTICAL_SCALES, dtype=values.dtype, device=values.device)
    scales = scales.view(*([1] * (values.dim() - 3)), -1, 1, 1)
    out = values / scales
    return out[0] if squeeze else out


def phase_inputs(radar: torch.Tensor) -> torch.Tensor:
    vq = vertical_quantities(radar)
    names = ("cap_argmax_km", "cap_bright_dbz", "cap_nz_levels",
             "cap_slope_dbz_km", "cap_low_dbz", "cap_et_gap_km")
    return torch.cat([vq[n] for n in names], dim=1)


# =============================================================================
# 4. operator —— 相态路由 Z-R 算子 + 结构化雨量头
# =============================================================================
PHASE_STATES: tuple[str, ...] = ("warm_rain", "mixed", "ice")


@dataclass
class OperatorOutput:
    rain: torch.Tensor
    weights: torch.Tensor
    params: dict[str, torch.Tensor]


class PhaseRouter(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 24,
                 n_states: int = len(PHASE_STATES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_inputs, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv2d(hidden, n_states, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(phase), dim=1)


class PhaseRoutedZROperator(nn.Module):
    def __init__(self, z_min: float = 5.0,
                 init_a: tuple[float, ...] = (1.600, 1.600, 1.600),
                 init_b: tuple[float, ...] = (0.660, 0.660, 0.660),
                 residual_scale: float = 2.0) -> None:
        super().__init__()
        if len(init_a) != len(PHASE_STATES) or len(init_b) != len(PHASE_STATES):
            raise ValueError("one (a, b) pair per phase state is required")
        self.z_min = float(z_min)
        self.router = PhaseRouter(n_inputs=6)
        self.raw_a = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(init_a, dtype=torch.float32))))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.residual_scale = float(residual_scale)

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.raw_a)

    def forward(self, radar: torch.Tensor,
                z_dbz: Optional[torch.Tensor] = None) -> OperatorOutput:
        if z_dbz is None:
            z_dbz = radar[:, CR_INDEX] / 10.0
        weights = self.router(phase_inputs(radar))
        z = z_dbz.clamp(min=self.z_min)
        zk = z.unsqueeze(1).expand_as(weights)
        a, b = self.a.view(1, -1, 1, 1), self.b.view(1, -1, 1, 1)
        rain_h = (weights * (a * torch.pow(zk, b))).sum(dim=1)
        rain = rain_h / 10.0
        rain = torch.where(z_dbz > self.z_min, rain, torch.zeros_like(rain))
        return OperatorOutput(
            rain=rain, weights=weights,
            params={"a": self.a.detach(), "b": self.b.detach()})


class StructuredRainHead(nn.Module):
    def __init__(self, n_features: int, z_min: float = 5.0, hidden: int = 32,
                 residual_scale: float = 2.0) -> None:
        super().__init__()
        self.operator = PhaseRoutedZROperator(
            z_min=z_min, residual_scale=residual_scale)
        self.residual = nn.Sequential(
            nn.Conv2d(n_features, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, radar: torch.Tensor, features: torch.Tensor,
                z_dbz: Optional[torch.Tensor] = None) -> OperatorOutput:
        op = self.operator(radar, z_dbz=z_dbz)
        delta = self.residual(features).squeeze(1)
        bound = self.operator.residual_scale + op.rain.detach()
        rain = torch.clamp(op.rain + torch.tanh(delta) * bound, min=0.0)
        if z_dbz is None:
            z_dbz = radar[:, CR_INDEX] / 10.0
        rain = torch.where(z_dbz > self.operator.z_min, rain,
                           torch.zeros_like(rain))
        return OperatorOutput(rain=rain, weights=op.weights, params=op.params)


# =============================================================================
# 5. motion —— 从雷达回波实测平流 + F37 空间速度场平流支持
# =============================================================================
@dataclass
class Motion:
    vx: float = 0.0
    vy: float = 0.0
    usable: bool = False

    @property
    def speed(self) -> float:
        return float((self.vx ** 2 + self.vy ** 2) ** 0.5)

    @property
    def upstream(self) -> tuple[float, float]:
        return (-self.vx, -self.vy)

    def as_tuple(self) -> tuple[float, float]:
        return (self.vx, self.vy)


@dataclass
class MotionField:
    vx: torch.Tensor
    vy: torch.Tensor
    usable: torch.Tensor
    method: str = "corr"

    def scalar(self, index: int = 0) -> Motion:
        return Motion(vx=float(self.vx[index]), vy=float(self.vy[index]),
                      usable=bool(self.usable[index]))


ECHO_THRESHOLD_DBZ: float = 20.0
CHAIN_STEPS: int = 5


def _echo_field(radar: torch.Tensor, thresh_dbz: float):
    cr = radar[:, :, CR_INDEX] / 10.0
    return cr, cr > thresh_dbz


def _search_shift(a: torch.Tensor, b: torch.Tensor,
                  dy_cand: torch.Tensor, dx_cand: torch.Tensor):
    bsz, h, w = a.shape
    s = dy_cand.shape[1]
    rows = torch.arange(h, device=a.device).view(1, 1, h, 1)
    cols = torch.arange(w, device=a.device).view(1, 1, 1, w)
    ys = rows + dy_cand.view(bsz, s, 1, 1)
    xs = cols + dx_cand.view(bsz, s, 1, 1)
    valid = ((ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)).expand(bsz, s, h, w)
    idx_y = ys.expand(bsz, s, h, w).clamp(0, h - 1).long()
    idx_x = xs.expand(bsz, s, h, w).clamp(0, w - 1).long()
    a_sel = a.view(bsz, 1, h, w).expand(bsz, s, h, w).gather(2, idx_y)
    a_sel = a_sel.gather(3, idx_x)
    b_sel = b.view(bsz, 1, h, w).expand(bsz, s, h, w)
    mask = valid.to(a.dtype)
    n = mask.sum(dim=(2, 3)).clamp(min=1.0)
    a_mean = (a_sel * mask).sum(dim=(2, 3)) / n
    b_mean = (b_sel * mask).sum(dim=(2, 3)) / n
    da = (a_sel - a_mean.view(bsz, s, 1, 1)) * mask
    db = (b_sel - b_mean.view(bsz, s, 1, 1)) * mask
    cov = (da * db).sum(dim=(2, 3))
    va = (da * da).sum(dim=(2, 3))
    vb = (db * db).sum(dim=(2, 3))
    score = cov / torch.sqrt(va * vb + 1e-6)
    best = score.argmax(dim=1)
    return (dy_cand.gather(1, best.view(-1, 1)).squeeze(1),
            dx_cand.gather(1, best.view(-1, 1)).squeeze(1))


def _estimate_pair_shift(a: torch.Tensor, b: torch.Tensor, max_shift: int):
    bsz = a.shape[0]
    span = torch.arange(-max_shift, max_shift + 1, 2, device=a.device)
    dys, dxs = torch.meshgrid(span, span, indexing="ij")
    cdy = dys.flatten().view(1, -1).expand(bsz, -1)
    cdx = dxs.flatten().view(1, -1).expand(bsz, -1)
    dy0, dx0 = _search_shift(a, b, cdy, cdx)
    fine = torch.arange(-1, 2, device=a.device)
    fy, fx = torch.meshgrid(fine, fine, indexing="ij")
    return _search_shift(a, b, dy0.view(-1, 1) + fy.flatten().view(1, -1),
                         dx0.view(-1, 1) + fx.flatten().view(1, -1))


def _centroid_motion(cr: torch.Tensor, mask: torch.Tensor):
    b, t = cr.shape[0], cr.shape[1]
    m = mask.to(cr.dtype)
    total = m.flatten(2).sum(dim=2)
    h, w = cr.shape[-2:]
    rows = torch.arange(h, dtype=cr.dtype, device=cr.device).view(1, 1, h, 1)
    cols = torch.arange(w, dtype=cr.dtype, device=cr.device).view(1, 1, 1, w)
    cy = (m * rows).flatten(2).sum(dim=2) / total.clamp(min=1.0)
    cx = (m * cols).flatten(2).sum(dim=2) / total.clamp(min=1.0)
    idx = torch.arange(t, dtype=cr.dtype, device=cr.device).view(1, t).expand(b, t)
    valid = total > 10.0
    n = valid.sum(dim=1).clamp(min=1).to(cr.dtype)
    sx, sxx = (idx * valid).sum(dim=1), (idx * idx * valid).sum(dim=1)
    denom = n * sxx - sx * sx
    sy, sxy = (cy * valid).sum(dim=1), (idx * cy * valid).sum(dim=1)
    slope_y = torch.where(denom.abs() > 1e-6, (n * sxy - sx * sy) / denom,
                          torch.zeros_like(denom))
    sy_x, sxy_x = (cx * valid).sum(dim=1), (idx * cx * valid).sum(dim=1)
    slope_x = torch.where(denom.abs() > 1e-6, (n * sxy_x - sx * sy_x) / denom,
                          torch.zeros_like(denom))
    return slope_x, slope_y, valid.sum(dim=1) >= 4


def estimate_motion_field(radar: torch.Tensor,
                          thresh_dbz: float = ECHO_THRESHOLD_DBZ,
                          mode: str = "corr", max_shift: int = 8,
                          chain_steps: int = CHAIN_STEPS) -> MotionField:
    squeeze = radar.dim() == 4
    x = radar[None] if squeeze else radar
    if x.dim() != 5:
        raise ValueError(f"expected (B, T, 10, H, W), got {tuple(radar.shape)}")
    b = x.shape[0]
    cr, mask = _echo_field(x, thresh_dbz)
    if mode == "none":
        zero = torch.zeros(b, dtype=x.dtype, device=x.device)
        return MotionField(zero, zero.clone(),
                           torch.zeros(b, dtype=torch.bool, device=x.device),
                           method="none")
    if mode == "centroid":
        vx, vy, usable = _centroid_motion(cr, mask)
        return MotionField(vx, vy, usable, method="centroid")
    if mode != "corr":
        raise ValueError(f"unknown motion mode {mode!r}")
    t = x.shape[1]
    steps = max(1, min(int(chain_steps), t - 1))
    vx = torch.zeros(b, dtype=x.dtype, device=x.device)
    vy = torch.zeros(b, dtype=x.dtype, device=x.device)
    counted = torch.zeros(b, dtype=x.dtype, device=x.device)
    echo_count = mask.to(x.dtype).flatten(2).sum(dim=2)
    for k in range(steps):
        earlier = x[:, t - 1 - k - 1]
        later = x[:, t - 1 - k]
        a = later[:, CR_INDEX] / 10.0
        bfr = earlier[:, CR_INDEX] / 10.0
        dy, dx = _estimate_pair_shift(a, bfr, max_shift)
        pair_ok = ((echo_count[:, t - 1 - k - 1] > 10.0)
                   & (echo_count[:, t - 1 - k] > 10.0))
        vx = vx + dx.to(x.dtype) * pair_ok.to(x.dtype)
        vy = vy + dy.to(x.dtype) * pair_ok.to(x.dtype)
        counted = counted + pair_ok.to(x.dtype)
    n = counted.clamp(min=1.0)
    vx, vy = vx / n, vy / n
    usable = counted >= 2
    return MotionField(torch.where(usable, vx, torch.zeros_like(vx)),
                       torch.where(usable, vy, torch.zeros_like(vy)),
                       usable, method="corr")


def estimate_motion(radar: torch.Tensor,
                    thresh_dbz: float = ECHO_THRESHOLD_DBZ,
                    mode: str = "corr", max_shift: int = 8) -> Motion:
    return estimate_motion_field(
        radar, thresh_dbz=thresh_dbz, mode=mode, max_shift=max_shift).scalar(0)


def warp(field: torch.Tensor, vx, vy, steps: float,
         dx_extra: Optional[torch.Tensor] = None,
         dy_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
    if field.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(field.shape)}")
    b, c, h, w = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=field.dtype, device=field.device),
        torch.arange(w, dtype=field.dtype, device=field.device), indexing="ij")
    if torch.is_tensor(vx):
        vx_b = vx.to(field.dtype).reshape(b, 1, 1)
        vy_b = vy.to(field.dtype).reshape(b, 1, 1)
        src_x = xx.unsqueeze(0) - vx_b * steps
        src_y = yy.unsqueeze(0) - vy_b * steps
    else:
        src_x = (xx - float(vx) * steps).unsqueeze(0)
        src_y = (yy - float(vy) * steps).unsqueeze(0)
    if dx_extra is not None:
        src_x = src_x - dx_extra.to(field.dtype).reshape(b, h, w)
    if dy_extra is not None:
        src_y = src_y - dy_extra.to(field.dtype).reshape(b, h, w)
    gx = (src_x / (w - 1)) * 2.0 - 1.0
    gy = (src_y / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    if grid.shape[0] == 1 and b > 1:
        grid = grid.expand(b, *grid.shape[1:])
    return F.grid_sample(field, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def advect_sequence(field: torch.Tensor, motion, n_steps: int,
                    dvx_field: Optional[torch.Tensor] = None,
                    dvy_field: Optional[torch.Tensor] = None,
                    source_field: Optional[torch.Tensor] = None,
                    ) -> torch.Tensor:
    """外推 n_steps 个未来时效 (k=0..n_steps-1 对应 lead 0..9, +6..+60 min)。

    F37 关键保底机制：
    * k=0 (lead 0, +6 min) 时，vel_weight = 0.0，tau_k = 0.0 -> **严格退化为纯雷达刚性平流**，
      lead 0 与纯雷达逐位恒等，绝不允许 NWP 演进项污染初始帧。
      （0.2499 是 F36 某一轮的观测值，不是模型常数，不得写进规格。）
    * k>=1 时，二次轨迹修正项 w_vel(k) = k*(k+1)/20 与零均值湿区生消项 tau(k) = k/(n_steps-1)
      随预报时效渐进接管，专门针对 +30~+60 min 的非线性轨迹偏折与对流生消。
    """
    frames = []
    denom = float(max(n_steps - 1, 1))
    for k in range(n_steps):
        steps = float(k + 1)
        if dvx_field is not None and dvy_field is not None and k > 0:
            w_vel = float(k * (k + 1)) / 20.0
            dx_e = dvx_field * w_vel
            dy_e = dvy_field * w_vel
        else:
            dx_e, dy_e = None, None
        adv_k = warp(field, motion.vx, motion.vy, steps,
                     dx_extra=dx_e, dy_extra=dy_e)
        if source_field is not None and k > 0:
            tau_k = float(k) / denom
            # 5x5 湿区邻域包络：只在平流回波附近允许生消调制，不在晴空区凭空造雨
            wet_env = (F.max_pool2d(adv_k, kernel_size=5, stride=1, padding=2)
                       > 0.05).to(adv_k.dtype)
            scale_k = 0.5 * adv_k.detach() + 0.5
            adv_k = torch.clamp(
                adv_k + tau_k * wet_env * source_field * scale_k, min=0.0)
        frames.append(adv_k)
    return torch.stack(frames, dim=1)


# =============================================================================
# 6. halo & F37 双时效 NWP 物理演变场提取
# =============================================================================
#: 上游 radar_nwp_bench/data/index.py::NWP_VARS 的权威顺序（共 27 通道）
#: 上游 ``radar_nwp_bench/data/index.py::NWP_VARS`` 的权威顺序（27 通道）。
#: 2026-11-14 F36 已三方核对（上游 index.py / qlfn/contract.py / 真实档案 nwp_vars）。
#: 2026-11-15 F37 合并稿曾退回一套自造表（HT20/LPL/MRH57/...），使 TTdMean74
#: 落到索引 22、实际取到 TTdMax74，HALO_CONTEXT_INDICES 变 (2,3,0,14,22)。
#: 该表必须逐字保持上游顺序，禁止二次自造或按位置兜底（坑 15）。
NWP_VARS: tuple[str, ...] = (
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000",
    "Q700", "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74",
    "TTdMean74", "WS500", "WS700", "WS925",
)


@dataclass
class HaloMasks:
    centre: torch.Tensor
    upstream: torch.Tensor
    downstream: torch.Tensor
    geometry: HaloGeometry


def _half_plane(geom: HaloGeometry, vx: float, vy: float) -> torch.Tensor:
    n = geom.upsampled
    c = (n - 1) / 2.0
    idx = torch.arange(n, dtype=torch.float32)
    dy = (idx - c).view(-1, 1)
    dx = (idx - c).view(1, -1)
    return (dx * (-vx) + dy * (-vy)) > 0


def build_masks(motion: Motion, max_distance_px: float,
                geometry: Optional[HaloGeometry] = None) -> HaloMasks:
    geom = geometry or HaloGeometry()
    n = geom.upsampled
    centre = torch.zeros((n, n), dtype=torch.bool)
    c0 = geom.crop
    centre[c0:c0 + geom.radar, c0:c0 + geom.radar] = True
    ring = ~centre
    up_half = _half_plane(geom, motion.vx, motion.vy)
    idx = torch.arange(n, dtype=torch.float32)
    dy = (idx - (n - 1) / 2.0).abs()
    dx = (idx - (n - 1) / 2.0).abs()
    outside = torch.maximum(dy.view(-1, 1), dx.view(1, -1)) - geom.radar / 2.0
    near = outside <= float(max_distance_px)
    return HaloMasks(centre=centre, upstream=ring & up_half & near,
                     downstream=ring & (~up_half) & near, geometry=geom)


def reachable_distance_px(motion: Motion, lead_minutes: float = 60.0,
                          step_minutes: float = 6.0) -> float:
    return motion.speed * (lead_minutes / step_minutes)


def summarise(field_upsampled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if field_upsampled.dim() != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(field_upsampled.shape)}")
    sel = field_upsampled[:, mask]
    if sel.shape[1] == 0:
        return field_upsampled.new_zeros((field_upsampled.shape[0], 4))
    q10 = torch.quantile(sel, 0.10, dim=1)
    return torch.stack([
        sel.mean(dim=1), q10, sel.min(dim=1).values,
        (sel < 0).to(sel.dtype).mean(dim=1),
    ], dim=1)


HALO_CONTEXT_VARS: tuple[str, ...] = (
    "DVG850", "DVG925", "CAPE", "PWAT", "TTdMean74",
)
HALO_CONTEXT_INDICES: tuple[int, ...] = tuple(
    NWP_VARS.index(v) for v in HALO_CONTEXT_VARS
)

#: F37 用于驱动 t>0 平流与生消的 10 个核心物理通道（动力 6 + 热力水汽 4）
NWP_EVOL_VARS: tuple[str, ...] = (
    "DVG850", "DVG925", "DVG200", "WS925", "WS700", "WS500",
    "CAPE", "PWAT", "TTdMean74", "KI",
)
NWP_EVOL_INDICES: tuple[int, ...] = tuple(
    NWP_VARS.index(v) for v in NWP_EVOL_VARS
)


def extract_nwp_evolution_tensor(x_nwp: torch.Tensor) -> torch.Tensor:
    """提取双时效 NWP 空间演变张量 (B, 20, H, W)：10 通道分析场 + 10 通道 1h 趋势差分场。

    首次真正消费 ``x_nwp[:, 1]``（FORECAST 模式下的 forecast_hour 切片）。
    """
    if x_nwp.dim() == 5:
        nwp0 = x_nwp[:, 0]
        nwp1 = x_nwp[:, 1] if x_nwp.shape[1] > 1 else nwp0
    elif x_nwp.dim() == 4:
        nwp0 = x_nwp
        nwp1 = x_nwp
    else:
        raise ValueError(f"unexpected x_nwp shape {tuple(x_nwp.shape)}")
    # 通道数不足 / 变量名未登记时**禁止**退回位置索引 0..9（F24/F36 老坑）。
    if nwp0.shape[1] < len(NWP_VARS):
        raise ValueError(
            f"x_nwp has {nwp0.shape[1]} channels; the F37 evolution tensor "
            f"needs the full {len(NWP_VARS)}-channel upstream table. "
            f"A positional fallback would silently read the wrong variables.")
    f0 = nwp0[:, NWP_EVOL_INDICES]
    df = nwp1[:, NWP_EVOL_INDICES] - f0
    return torch.cat([f0, df], dim=1)         # (B, 20, H, W)


def build_halo_context(nwp_native: torch.Tensor, motion: Motion,
                       max_distance_px: Optional[float] = None,
                       variables: tuple[str, ...] = HALO_CONTEXT_VARS
                       ) -> dict[str, torch.Tensor]:
    if nwp_native.dim() != 4:
        raise ValueError(f"expected (B, 27, 167, 167), got {tuple(nwp_native.shape)}")
    geom = HaloGeometry()
    reach = (reachable_distance_px(motion) if max_distance_px is None
             else max_distance_px)
    masks = build_masks(motion, reach, geom)
    up = bilinear_upsample(nwp_native)
    context: dict[str, torch.Tensor] = {}
    for name in variables:
        if name not in NWP_VARS:
            raise ValueError(
                f"unknown NWP variable {name!r}; refusing the positional "
                f"fallback that caused the F35/F36 channel misalignment")
        ch_idx = NWP_VARS.index(name)
        if ch_idx >= up.shape[1]:
            raise ValueError(
                f"NWP variable {name!r} needs channel {ch_idx}, "
                f"but the field has only {up.shape[1]} channels")
        f = up[:, ch_idx]
        context[f"{name}_upstream"] = summarise(f, masks.upstream)
        context[f"{name}_downstream"] = summarise(f, masks.downstream)
        context[f"{name}_centre"] = summarise(f, masks.centre)
    context["mask_upstream"] = masks.upstream
    context["mask_downstream"] = masks.downstream
    context["reach_px"] = torch.tensor(float(reach))
    return context


# =============================================================================
# 7. NWP 上下文连线（保留供 F36 兼容与回归自检）
# =============================================================================
NWP_CONTEXT_DIM: int = 20


def nwp_context_from_patch(x_nwp: torch.Tensor) -> torch.Tensor:
    if x_nwp.dim() == 5:
        x = x_nwp[:, 0]
    elif x_nwp.dim() == 4:
        x = x_nwp
    else:
        raise ValueError(f"unexpected x_nwp shape {tuple(x_nwp.shape)}")
    b = x.shape[0]
    parts = []
    if x.shape[1] < len(NWP_VARS):
        raise ValueError(
            f"x_nwp has {x.shape[1]} channels; the halo context needs the full "
            f"{len(NWP_VARS)}-channel upstream table. A positional fallback "
            f"would silently read channels 0..4 (F24/F36 old bug).")
    for ch_idx in HALO_CONTEXT_INDICES:
        if ch_idx < x.shape[1]:
            flat = x[:, ch_idx].reshape(b, -1)
            parts.append(torch.stack([
                flat.mean(dim=1),
                torch.quantile(flat, 0.10, dim=1),
                flat.min(dim=1).values,
                (flat < 0).to(flat.dtype).mean(dim=1),
            ], dim=1))
        else:
            parts.append(x.new_zeros((b, 4)))
    while len(parts) < len(HALO_CONTEXT_VARS):
        parts.append(x.new_zeros((b, 4)))
    return torch.cat(parts, dim=1)


def nwp_context_from_native(x_nwp_native: torch.Tensor,
                            motion: Union[Motion, MotionField]) -> torch.Tensor:
    if isinstance(motion, Motion):
        ctx = build_halo_context(x_nwp_native, motion)
        return torch.cat([ctx[f"{v}_upstream"] for v in HALO_CONTEXT_VARS], dim=1)
    rows = []
    for bi in range(x_nwp_native.shape[0]):
        m_b = motion.scalar(bi)
        ctx_b = build_halo_context(x_nwp_native[bi:bi + 1], m_b)
        rows.append(torch.cat(
            [ctx_b[f"{v}_upstream"] for v in HALO_CONTEXT_VARS], dim=1))
    return torch.cat(rows, dim=0)


def build_nwp_context(x_nwp: Optional[torch.Tensor],
                      motion: Optional[Union[Motion, MotionField]] = None,
                      nwp_context: Optional[torch.Tensor] = None,
                      ) -> tuple[Optional[torch.Tensor], str]:
    if nwp_context is not None:
        return nwp_context, "external"
    if x_nwp is None:
        return None, "none"
    native = x_nwp[:, 0] if x_nwp.dim() == 5 else x_nwp
    if (native.dim() == 4 and tuple(native.shape[-2:]) == NWP_SHAPE
            and motion is not None):
        return nwp_context_from_native(native, motion), "native_ring"
    return nwp_context_from_patch(x_nwp), "in_patch"


# =============================================================================
# 8. model —— t0 结构化雨量场 + F37 拉格朗日演进外推
# =============================================================================
def _residual_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(1, channels), nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(1, channels),
    )


@dataclass
class ModelOutput:
    rain: torch.Tensor
    rain_t0: torch.Tensor
    rain_history: Optional[torch.Tensor] = None
    motion: Optional[MotionField] = None
    diagnostics: dict = field(default_factory=dict)


class QLFN2Model(nn.Module):
    """结构化 QPE + 平流，支持 F36 (t0 调制) 与 F37 (t>0 双时效拉格朗日演进)。"""

    def __init__(
        self,
        in_len: int = 11,
        out_len: int = 10,
        use_halo: bool = True,
        use_nwp: bool = True,
        use_nwp_gate: bool = True,
        use_nwp_vel: bool = False,
        use_nwp_source: bool = False,
        vel_max_delta: float = 1.2,
        use_history_head: bool = True,
        z_min: float = 5.0,
        width: int = 48,
        motion_mode: str = "corr",
        max_shift: int = 8,
    ) -> None:
        super().__init__()
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.use_nwp = bool(use_nwp)
        self.use_halo = bool(use_halo) and self.use_nwp
        self.use_nwp_gate = bool(use_nwp_gate) and self.use_nwp
        self.use_nwp_vel = bool(use_nwp_vel) and self.use_nwp
        self.use_nwp_source = bool(use_nwp_source) and self.use_nwp
        self.vel_max_delta = float(vel_max_delta)
        self.use_history_head = bool(use_history_head)
        self.motion_mode = str(motion_mode)
        self.max_shift = int(max_shift)
        self.n_features = len(RADAR_VARS) * 3 + 10  # net + 2 trends + 10 vq

        self.trunk = nn.Sequential(
            nn.Conv2d(self.n_features, width, 3, padding=1),
            nn.GroupNorm(1, width), nn.GELU(),
            _residual_block(width), nn.GELU(),
        )
        self.head = StructuredRainHead(width, z_min=z_min, hidden=width)

        if self.use_halo:
            self.halo_proj = nn.Linear(NWP_CONTEXT_DIM, self.n_features)
            nn.init.zeros_(self.halo_proj.weight)
            nn.init.zeros_(self.halo_proj.bias)
        else:
            self.register_parameter("halo_proj", None)

        if self.use_nwp_gate:
            self.nwp_gate = nn.Sequential(
                nn.Conv2d(27, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1),
            )
            nn.init.zeros_(self.nwp_gate[-1].weight)
            nn.init.zeros_(self.nwp_gate[-1].bias)
        else:
            self.register_parameter("nwp_gate", None)

        evol_in_c = len(NWP_EVOL_VARS) * 2  # 20 通道 (t0 + dt)
        if self.use_nwp_vel:
            self.nwp_vel_net = nn.Sequential(
                nn.Conv2d(evol_in_c, 16, 3, padding=1),
                nn.GroupNorm(1, 16), nn.GELU(),
                nn.Conv2d(16, 2, 3, padding=1),
            )
            nn.init.zeros_(self.nwp_vel_net[-1].weight)
            nn.init.zeros_(self.nwp_vel_net[-1].bias)
        else:
            self.register_parameter("nwp_vel_net", None)

        if self.use_nwp_source:
            self.nwp_source_net = nn.Sequential(
                nn.Conv2d(evol_in_c, 16, 3, padding=1),
                nn.GroupNorm(1, 16), nn.GELU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )
            nn.init.zeros_(self.nwp_source_net[-1].weight)
            nn.init.zeros_(self.nwp_source_net[-1].bias)
        else:
            self.register_parameter("nwp_source_net", None)

    @staticmethod
    def causal_features(net: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
        b, t = net.shape[:2]
        prev1 = torch.cat([net[:, :1], net[:, :-1]], dim=1)
        idx = torch.arange(t, device=net.device)
        prev5 = net[:, (idx - 5).clamp(min=0)]
        return torch.cat([net, net - prev1, net - prev5, vq], dim=2)

    @staticmethod
    def _nwp_sample_mask(x_nwp: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_nwp is None:
            return None
        reduce_dims = tuple(range(1, x_nwp.dim()))
        return (x_nwp.abs().amax(dim=reduce_dims) > 0).to(x_nwp.dtype)

    @staticmethod
    def _nwp_is_active(x_nwp: Optional[torch.Tensor]) -> bool:
        if x_nwp is None:
            return False
        return bool((x_nwp != 0).any().item())

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        halo_context: Optional[torch.Tensor] = None,
        nwp_context: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        if x_radar.dim() != 5:
            raise ValueError(f"expected (B, T, 10, H, W), got {tuple(x_radar.shape)}")
        b, t = x_radar.shape[:2]
        if t < 2:
            raise ValueError("the motion estimate needs at least two frames")

        net = archive_to_net(x_radar)
        vq = scaled_vertical_quantities(x_radar)
        feats = self.causal_features(net, vq)
        motion = estimate_motion_field(x_radar, mode=self.motion_mode,
                                       max_shift=self.max_shift)

        sample_mask = self._nwp_sample_mask(x_nwp) if self.use_nwp else None
        nwp_active = (sample_mask is not None
                      and bool((sample_mask > 0).any().item()))
        nwp_source = "disabled"
        ctx = halo_context if halo_context is not None else nwp_context
        if nwp_active and self.halo_proj is not None:
            ctx, nwp_source = build_nwp_context(
                x_nwp, motion, nwp_context=ctx)
        if self.halo_proj is not None and ctx is not None:
            cond = self.halo_proj(ctx).view(b, 1, self.n_features, 1, 1)
            if sample_mask is not None:
                cond = cond * sample_mask.view(b, 1, 1, 1, 1)
            feats = torch.cat([
                feats[:, :-1], feats[:, -1:] + cond,
            ], dim=1)

        t0_feats = feats[:, -1]
        trunk = self.trunk(t0_feats)
        op = self.head(x_radar[:, -1], trunk)
        rain_t0 = op.rain.unsqueeze(1)                       # (B, 1, H, W)

        if nwp_active and self.nwp_gate is not None and sample_mask is not None:
            nwp0 = x_nwp[:, 0] if x_nwp.dim() == 5 else x_nwp
            gate = torch.tanh(self.nwp_gate(nwp0))           # (B, 1, H, W)
            gate = gate * sample_mask.view(b, 1, 1, 1)
            rain_t0 = torch.clamp(rain_t0 * (1.0 + gate), min=0.0)
            if nwp_source == "disabled":
                nwp_source = "gate_only"

        # ---- F37: 双时效 NWP 驱动 t>0 (k>=1) 平流轨迹修正与零均值生消 ----
        dvx_field: Optional[torch.Tensor] = None
        dvy_field: Optional[torch.Tensor] = None
        src_field: Optional[torch.Tensor] = None
        if nwp_active and sample_mask is not None and (
                self.nwp_vel_net is not None or self.nwp_source_net is not None):
            evol = extract_nwp_evolution_tensor(x_nwp)       # (B, 20, H, W)
            m4 = sample_mask.view(b, 1, 1, 1)
            if self.nwp_vel_net is not None:
                raw_v = torch.tanh(self.nwp_vel_net(evol)) * self.vel_max_delta
                raw_v = raw_v * m4
                dvx_field = raw_v[:, 0]                      # (B, H, W)
                dvy_field = raw_v[:, 1]                      # (B, H, W)
            if self.nwp_source_net is not None:
                raw_s = self.nwp_source_net(evol)            # (B, 1, H, W)
                # 空间零均值化：禁止全局抬高或压低全图雨量，只允许空间重分配/局地生消
                zm_s = raw_s - raw_s.mean(dim=(-2, -1), keepdim=True)
                src_field = torch.tanh(zm_s) * m4
            if nwp_source == "disabled":
                nwp_source = "lagrangian_2slice"

        if self.motion_mode == "none":
            rain = rain_t0.unsqueeze(1).expand(
                -1, self.out_len, -1, -1, -1).contiguous()
        else:
            rain = advect_sequence(
                rain_t0, motion, self.out_len,
                dvx_field=dvx_field, dvy_field=dvy_field,
                source_field=src_field,
            )

        rain_hist = None
        if self.use_history_head:
            hist_trunk = self.trunk(feats.reshape(b * t, *feats.shape[2:]))
            hist_trunk = hist_trunk.reshape(b, t, *hist_trunk.shape[1:])
            hist_frames = [
                self.head(x_radar[:, k], hist_trunk[:, k]).rain
                .unsqueeze(1).unsqueeze(1)
                for k in range(t - 1)
            ]
            hist_frames.append(rain_t0.unsqueeze(1))   # (B,1,1,H,W)
            rain_hist = torch.cat(hist_frames, dim=1)

        diagnostics = {
            "motion_vx": float(motion.vx[0]),
            "motion_vy": float(motion.vy[0]),
            "motion_usable": bool(motion.usable[0]),
            "phase_weights_mean": op.weights.detach().mean(dim=(0, 2, 3)).tolist(),
            "operator_a": op.params["a"].tolist(),
            "operator_b": op.params["b"].tolist(),
            "nwp_active": nwp_active,
            "nwp_source": nwp_source,
            "halo_connected": bool(self.halo_proj is not None and ctx is not None),
            "gate_connected": bool(self.nwp_gate is not None and nwp_active),
            "vel_connected": bool(self.nwp_vel_net is not None and nwp_active),
            "source_connected": bool(self.nwp_source_net is not None and nwp_active),
        }
        return ModelOutput(rain=rain, rain_t0=rain_t0,
                           rain_history=rain_hist, motion=motion,
                           diagnostics=diagnostics)


# =============================================================================
# 9. bench seam —— 上游 harness 接缝
# =============================================================================
BENCH_CANDIDATES: tuple[str, ...] = (
    os.environ.get("RADAR_NWP_BENCH") or "",
    "/home/wf/radarNWP",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "..", "1radarNWPprediction", "radar-nwp-bench"),
    "R:/1radarNWPprediction/radar-nwp-bench",
)


def add_bench_to_path() -> Optional[str]:
    for root in BENCH_CANDIDATES:
        if root and os.path.isdir(os.path.join(root, "radar_nwp_bench")):
            root = os.path.abspath(root)
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    return None


def require_bench() -> str:
    root = add_bench_to_path()
    if root is None:
        raise RuntimeError(
            "the benchmark (radar_nwp_bench) is not importable here. "
            "Set RADAR_NWP_BENCH to its root.")
    return root


add_bench_to_path()

try:
    from radar_nwp_bench.models.base import (  # noqa: E402
        BasePrecipitationModel as _Base,
    )
except ImportError:  # pragma: no cover
    class _Base(nn.Module):  # type: ignore[no-redef]
        def __init__(self, in_radar_c=10, in_nwp_c=27, in_len=11,
                     out_len=10, out_channels=1, **unused):
            super().__init__()
            self.in_radar_c = in_radar_c
            self.in_nwp_c = in_nwp_c
            self.in_len = in_len
            self.out_len = out_len
            self.out_channels = out_channels
            self.nwp_time_len = 2

        def resolve_nwp(self, x_radar, x_nwp):
            if x_nwp is not None:
                return x_nwp
            b = x_radar.shape[0]
            h, w = x_radar.shape[-2:]
            return torch.zeros(b, self.nwp_time_len, self.in_nwp_c, h, w,
                               device=x_radar.device, dtype=x_radar.dtype)

        def __call__(self, x_radar, x_nwp=None, **kwargs):
            return super().__call__(x_radar, self.resolve_nwp(x_radar, x_nwp),
                                   **kwargs)


class QLFN2Adapter(_Base):
    """上游 ``scripts/train.py`` / ``evaluate.py`` 唯一认识的模型接缝。"""

    def __init__(
        self,
        in_radar_c: int = 10,
        in_nwp_c: int = 27,
        in_len: int = 11,
        out_len: int = 10,
        out_channels: int = 1,
        width: int = 48,
        z_min: float = 5.0,
        motion_mode: str = "corr",
        max_shift: int = 8,
        use_halo: bool = True,
        use_nwp: bool = True,
        use_nwp_gate: bool = True,
        use_nwp_vel: bool = False,
        use_nwp_source: bool = False,
        vel_max_delta: float = 1.2,
        use_history_head: bool = True,
        **unused: object,
    ) -> None:
        super().__init__(in_radar_c=in_radar_c, in_nwp_c=in_nwp_c,
                         in_len=in_len, out_len=out_len,
                         out_channels=out_channels)
        nwp_ready = bool(use_nwp) and int(in_nwp_c) == 27
        self.core = QLFN2Model(
            in_len=in_len, out_len=out_len,
            use_halo=bool(use_halo) and nwp_ready,
            use_nwp=nwp_ready,
            use_nwp_gate=bool(use_nwp_gate) and nwp_ready,
            use_nwp_vel=bool(use_nwp_vel) and nwp_ready,
            use_nwp_source=bool(use_nwp_source) and nwp_ready,
            vel_max_delta=vel_max_delta,
            use_history_head=use_history_head, z_min=z_min, width=width,
            motion_mode=motion_mode, max_shift=max_shift,
        )
        self.last_qpe_pred: Optional[torch.Tensor] = None
        self.last_diagnostics: dict = {}

    def forward(
        self,
        x_radar: torch.Tensor,
        x_nwp: Optional[torch.Tensor] = None,
        halo_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw = net_to_archive(x_radar)
        out = self.core(raw, x_nwp=x_nwp, halo_context=halo_context)
        self.last_qpe_pred = out.rain_history
        self.last_diagnostics = out.diagnostics
        return out.rain


def register_qlfn2(name: str = "qlfn2"):
    require_bench()
    from radar_nwp_bench.models import register_model  # noqa: E402
    register_model(name)(QLFN2Adapter)
    return QLFN2Adapter
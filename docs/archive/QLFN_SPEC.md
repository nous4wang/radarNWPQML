# QLag-FlowNet（QLFN）代码库规格 · v1.1

> 本文件是 `qlfn/` 的**需求来源**，供两轴代码审查的 Spec 轴比对。
> 日期：2026-11-08 · 上游基准：`radar-nwp-bench` @ `88fd03c`（MASTER v2.28）
> 关联评审：`docs/QLFN_BLUEPRINT_REVIEW.md` · ARIS 证据：`.aris/evidence/`

## 0. 定位（最重要的约束）

按用户授权构建 `QLagrangeFlowNet`（展示名 **`QLag-FlowNet`**，代码标识 **`QLFN`**），
定位为 **可审计参考实现（auditable reference implementation）**：

- 目标是把蓝图 §5 的四段代码变成**能运行、能自检、缺陷可定位**的真实代码库；
- **不训练、不评测、不改上游、不写 `configs/baselines.json`、不宣称任何分数**；
- 每个模块文件头必须标注：三族机制已被 `MASTER.md §9 / §11.8.2` 封口，
  本实现仅作工程参考，不构成研究路线或改进声明。

### 命名

| 用途 | 名称 |
|---|---|
| 论文/文档展示名 | `QLag-FlowNet` |
| 代码标识（包名、配置前缀、别名） | `QLFN`（`qlfn/`、`QLFNConfig`、`QLFN`） |
| 主类 | `QLagrangeFlowNet` |
| 弃用 | `QFN`（丢失 Lagrange 词根、与既有缩写撞车） |

风险留档：`FlowNet` 与 ICCV 2015 光流家族及上游封口的显式运动族撞名；
`Lag` 指向被 `MASTER.md:1048/1088` 判负的拉格朗日平流。作为参考实现是诚实标记。

### 依赖例外（相对 `CODING_STANDARDS.md` §1）

`CODING_STANDARDS.md` 的「零第三方依赖」约束针对 `qmlbench/` 静态扫描层。
本代码库分层遵守：

| 层 | 文件 | 依赖 | 运行位置 |
|---|---|---|---|
| 契约/审计层 | `qlfn/contract.py`、`qlfn/data_audit.py`、`qlfn/__init__.py` | **仅标准库** | 本机可运行 |
| 模型层 | `qlfn/geo_prior.py`、`scale_bridge.py`、`quantum_latent.py`、`flow_head.py`、`losses.py`、`model.py` | PyTorch（远端 conda） | 仅远端 |
| 自检层 | `scripts/verify_qlfn_contract.py`（零依赖）、`scripts/qlfn_smoke_test.py`（torch） | 分层 | 本机 / 远端 |

`qlfn/__init__.py` **不得**在 import 时加载 torch 子模块（用惰性 `__getattr__`），
否则本机契约自检会被 torch 缺失阻断。

## 1. 真实数据契约（不得臆造）

来源：`MASTER.md:51-54,74-76`、`models/base.py:10-19`、`data/index.py:14-24`、
`configs/default.yaml:29-46`、本项目 `configs/baselines.json`。

| 项 | 值 |
|---|---|
| `x_radar` | `(B, 11, 10, H, W)`，11 帧 × 6min；通道 `CAP20,CAP30,CAP40,CAP50,CAP60,CAP70,CR,ET,HBR,VIL`（CR=6、VIL=9） |
| `x_nwp` | `(B, 2, 27, H, W)`，**已重采样到雷达网格**；27 变量无 U/V、无风向、无 850hPa 风分量 |
| `y_rain` | `(B, 10, 1, H, W)`，单位 **mm per 6-min accumulation**（不是 mm/h） |
| 阈值 | 0.1 / 0.5 / 1.0 / 2.0 / 4.0 / 8.0 / 16.0 mm per 6min |
| 模型输入尺寸 | 训练默认 `patch_size=128`、`slide_stride=None`；301×301 是评测缝合后的**打分全场**，不是模型输入 |
| 主指标 | 全场缝合 `CSI@2.0` |
| 划分 | 洁净 23/8/8（test n=538）；robustness 23 case（n=962），两组不可相减 |

**禁止**：硬编码 301/303/101/167、假设 167 原生 NWP 输入、假设存在 U/V 风场、
把 `CAPE`/`DVG200` 当 U/V、把 `WS*`/`DVG850` 当风矢量。

## 2. 功能需求（验收编号）

> **v1.1（2026-11-08，ARIS W2 Round 1 + aihero 两轴复审后修订）**：
> R3 消除自相矛盾（原写「可被 3 整除」，实现与测试却用 64/128）；
> R4 改为真实数值路径验收；R6/R8 撤回 S1/T2 误报（见 §2.1）；
> R7 补先验一致/目标语义/cond 可达；R9 用上游真实格距；
> R10 明确「诊断量绝不门控输出」；R12 要求真实产物；R14 要求全仓扫描 + 检测器阳性对照。

| ID | 需求 | 验收方式 |
|---|---|---|
| R1 | 契约常量集中、带上游出处注释；`QLFNConfig` 为 frozen + 可哈希 dataclass；`WIND_SURROGATE_INDICES` 必须由 `NWP_VARIABLES` 名字导出并与上游 `data/index.py` 交叉核对 | `verify_qlfn_contract.py` 静态断言（含上游 provenance） |
| R2 | `x_radar`/`x_nwp`/`y_rain` 形状与通道校验；**训练批次缺 `y_rain` = FAIL**（`require_target=True`），推理允许 `y_rain=None` | 纯标准库 `validate_batch_shapes()`；三个分支都有正反用例 |
| R3 | **任意正整数 `H,W`** 经对称 padding 使 padded 尺寸可被 `fold` 整除后折叠/还原；**不得硬编码 301** | 零依赖 `fold_plan` 覆盖 64/66/128/301；冒烟 forward 覆盖 64/66/128；301 做 fold roundtrip + 一次 `flow_steps=1` 的轻量 forward（CPU 预算） |
| R4 | 折叠/还原数值可逆（无 encoder 参与时误差 < 1e-6）；**必须走真实 `fold_channels()` → `decode_subpixel()` 路径**，不得只自测 `PixelUnshuffle(PixelShuffle())` | 冒烟 `fold_roundtrip`，逐尺寸报告 `max_abs` |
| R5 | `forward(x_radar, x_nwp) -> (B,10,1,H,W)` 与非负输出 | 冒烟测试 |
| R6 | **S5 通道断言**：解码前校验 `subpixel_channels == T_out*fold²`，禁止无断言的 `view`；不得声称「修 S1」（已撤回为误报） | 静态 + 冒烟（错通道数必须抛 `ValueError`） |
| R7 | **T1 修复**：提供 CFM 损失 `‖v_θ(z_s,s,c) − (x₁−z₀)‖²`；且①训练/推理先验**一致**（zero prior）；② `x₁` 是 `body` 之上的**残差**而非完整目标；③ `cond` 必须来自模型内部可达路径（不得要求外部调用者手工构造 latent） | `flow_head.py`；`model.flow_training_loss(x_radar, x_nwp, target_rain)`；冒烟测梯度可回传 + 先验一致性 + 语义一致性 |
| R8 | **binary mask 契约**：`mask∈{0,1}` 有显式校验，全掩码批次 `tp=fp=fn=0`；pure-Python 与 torch 两式一致。不得声称「修 T2」（已撤回为误报，两式对 binary mask 代数等价） | 冒烟全掩码 + 两实现交叉比对 + 非 binary mask 抛错 |
| R9 | **S6 修复**：位移按上游真实格距 **1.11195 km/px**（`MASTER.md:1556`）换算，不写死 150；12.0 km 上限按 `H/W` 与格距推导，patch 输入下尺度正确 | 静态（无 `150.0`、无 `1000.0` 格距）+ 冒烟数值断言 |
| R10 | P1–P5 降级：docstring 明确「非密度矩阵、非偏迹、非物理纠缠熵」；**`norm_entropy` 只写入 `last_diagnostics`，绝不参与 `pred_rain` 计算** | 静态检查 `model.py` 无 `entropy` 参与输出；运行时断言移除 entropy 后输出 bit-exact 不变 |
| R11 | D1/D3/D4 不伪造数据：U/V 用显式声明的 `WS*/DVG*` 标量代理，不假设 167 输入，掩码协议作为独立工具 | 静态 + `data_audit.py` |
| R12 | 掩码协议工具与上游 evaluator 隔离：必须**真实生成**文件名含 `NOT_COMPARABLE` 的产物，且 `write_artifact` 有调用点 | `verify_qlfn_contract.py` 写出 `artifacts/QLFN_NOT_COMPARABLE_case_audit.json` 并校验文件名与 caveat |
| R13 | 每个模块文件头含 §11.8.2 封口标注 | 静态检查 |
| R14 | 不训练、不评测、不改上游、不写权威表 | **全仓**（`qlfn/`、`scripts/`）扫描训练/评测驱动；并对检测器做阳性对照，证明它确实能命中 `optimizer.step()` |

### 2.1 已撤回的审查结论（不得再作为需求来源）

| 原编号 | 撤回理由 |
|---|---|
| S1「96 通道 → `nn.Identity()` 必然 shape 崩」 | 蓝图 `out_ch = t_out*9 = 90`，`integrate_ode` 输出即 90 通道，路径合法。真正缺陷是 T1 与 S5 |
| T2「缺失帧虚假计数」 | 蓝图 `p_hit`/`t_hit` 已乘 binary `mask`，因 `m²=m` 与 QLFN 改写式严格等价。QLFN 只是等价重写 |

推导与原始核对见 `docs/QLFN_BLUEPRINT_REVIEW.md` §1.1、`.aris/rounds/round-01.md`。

## 3. Non-goals

- 不实现蓝图声称的「严格希尔伯特等价」「解析物理探针」「保范围不变量」为**结论**；
  相关量一律标注为诊断量，且评审文档已列为 invalidated / provisional。
- 不补丁上游 `evaluate.py` / `precipitation.py`；不新增 `valid_step_mask` 到上游。
- 不回收 NWP 外围环带（`MASTER.md:224` 禁止外推）。
- 不产生 `CSI@2.0` 或任何分数；不进入 `artifacts/master_table.md` 等权威表。
- 不引入 spec 未要求的抽象、钩子或参数（CODING_STANDARDS §3）。

## 4. 目录

```
qlfn/
  __init__.py           惰性导出，纯标准库
  contract.py           契约常量、QLFNConfig、形状校验、drift_scale（零依赖）
  data_audit.py         62-Case 审计与 valid-step mask 协议（零依赖）
  checks.py             check()/render() 两个自检脚本共用（零依赖）
  geo_prior.py          StaticGeoRadarPrior（torch，动态 H/W）
  scale_bridge.py       DualScaleBridge 折叠/还原（torch）
  quantum_latent.py     AngleLiftInteraction 诊断量（torch）
  flow_head.py          LatentConditionalFlowHead + CFM 训练损失（torch）
  losses.py             masked_coverage_extreme_loss + 掩码统计（torch）
  model.py              QLagrangeFlowNet / QLFN（torch）
scripts/
  verify_qlfn_contract.py   本机零依赖静态自检
  qlfn_smoke_test.py        远端 torch 冒烟测试（CPU 即可）
```

## 5. 验收命令

```bash
# 本机（零依赖）；裸命令必须 exit 0 —— 上游路径按 --bench-root >
# RADAR_NWP_BENCH_ROOT > 默认 R:/1radarNWPprediction/radar-nwp-bench 解析
python scripts/verify_qlfn_contract.py
python scripts/verify_qlfn_contract.py --bench-root R:/1radarNWPprediction/radar-nwp-bench

# 远端（conda python，CPU 冒烟）
/home/wf/.conda/envs/radar_nwp/bin/python scripts/qlfn_smoke_test.py
```

## 6. 口径纪律

- 所有阈值单位写 mm per 6min，不写 mm/h。
- 雷达格距用上游 **1.11195 km/px**（`MASTER.md:1556`），
  不用蓝图/常见近似的 1 km/px。
- 洁净与断帧划分分数禁止相减。
- 缺失证据一律 FAIL。
- 不写 `MASTER.md:1767-1770` 的禁写句式；区分「实测 / 机制层 / 未测 / 已证伪」。
- 结论强度不得超过证据：`supports_improvement_claim = False` 恒定。

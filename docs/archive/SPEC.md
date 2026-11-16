# qmlbench 规格说明书（SPEC）

本文件是 `R:/1radarNWPQML` 本次改动的**需求来源**，供两轴代码审查的 Spec 轴比对。
版本：v1.0 · 2026-11-07 · 上游基准：`radar-nwp-bench` @ `88fd03c`（MASTER v2.28）

## 0. 背景与任务边界

输入材料：

1. `from_gemini/radar_nwp_quantum_inspired_ml_blueprint_cvpr_icml.md` —— 外部生成的 Q-LagrangeFlow 蓝图，主张用「物理拉格朗日形变对齐 + 量子启发 Hilbert/PEPS 张量网络 + 频率解耦条件流匹配」冲击 CVPR/ICML 2027。
2. `R:/1radarNWPprediction/radar-nwp-bench` —— 用户自有的、已有 38 个注册实验（E01–E38）的雷达+NWP 近预报基准库。

要求：理解蓝图 → 接入代码库中**所有** baseline → 设计冲击 AI 顶会的方法 → 用代码审查工具检查产出。

**上游权威顺序（MASTER.md §13）**：`MASTER.md §10.2` > `RESULTS_FINAL.md §10` > `RESULTS_FINAL.md` 其余 > `THREE_DEFECTS.md` > `TGRS_EXPERIMENTS.md` / `HQMA_SCAN.md` / `GEMINI_HQMA_V2_REVIEW.md` > `DATA_AUDIT.md` / `FORMAT_AUDIT.md` > `.workbuddy/memory/MEMORY.md` > `README.md`。与早期文档冲突时一律以 `MASTER.md §10.2` 为准。

## 1. 功能需求

| ID | 需求 | 验收方式 |
|---|---|---|
| R1 | 接入上游**全部** baseline：25 个 `@register_model` 注册模型 + 1 个解析基线 `persistence`，共 26 个 | `configs/baselines.json` 有 26 条；`verify_integration.py` 断言 adapter 文件集合与上游一致 |
| R2 | 蓝图 §6.1 列出的 14 个 baseline 必须全部在册，并保留蓝图别名 | `BLUEPRINT_POOL` 与 catalogue 的 `blueprint_alias` 双向一致 |
| R3 | 记录并固化统一 IO 契约：`x_radar (B,11,10,H,W)` + `x_nwp (B,2,27,H,W)` → `y_rain (B,10,1,H,W)` mm/6min | `baselines.json.contract`；阈值必须写 mm per 6 min，不得写 mm/h |
| R4 | 把**标准损失表**（`*.eval.json`，19 行）与**权威 score-aware 表**（`RESULTS_FINAL.md` §10.1，18 行）分列，禁止合并成一张排名 | `master_table()` 两列并置且附口径声明；`results.py` 分别提供两个 loader |
| R5 | 所有承重数字常量必须在代码注释里标注来源文件与章节；不得手工转录表格数字 | `FLOOR_CSI2` / `ORACLE_CSI2` / `SEED_RANGE` / `BEST_TRAINED_CSI2` 等均有出处；score-aware 表由解析器从 markdown 读取 |
| R6 | 把 `MASTER.md §11.9` 的 P1–P6 资格门实现为**可执行的检查器**，并覆盖 §11.2 两条准入原则与 §9/§10.2 已封口方向 | `qmlbench/gate.py`；`gate_review.md` 对蓝图三个模块逐条给出裁定 |
| R7 | 零第三方依赖（本机无 torch / numpy / PyYAML）。上游仓库作为**数据**静态扫描，不得 import 上游包 | `python scripts/sync_baselines.py` / `verify_integration.py` / `export_tables.py` 在纯标准库下可运行 |
| R8 | 提供三个可重复运行的入口：同步清单、完整性验证、导出论文表格 | 三个脚本均支持 `--bench-root`；`sync --check` 可检测上游漂移 |
| R9 | 产出一份基于实测证据的 AI 顶会方法设计文档 | `docs/VENUE_METHOD.md`，每个关键数字标注来源 |
| R10 | 明确 non-goal：**不实现** Q-LagrangeFlow 的三个模块 | 见 §2 |

## 2. Non-goals（明确不做）

- **N1 不实现 Q-LagrangeFlow**。蓝图三个模块分别命中上游 `MASTER.md:1067-1090` §9 舍弃清单中的「显式运动/拉格朗日平流」「量子启发复数干涉 + 二维张量网络 + 相位对齐」「生成式/扩散族 + 频域干预派生方案」，且 `docs/radarNWP-QI-optimization.md:3-8` 已 `REJECTED / SEALED`。按 §11.8.2「被证否 = 停止派生，不是换个说法再试一次」，不投算力、不装依赖。
- **N2 不训练、不评测、不跑 forward pass**。本层只做元数据接入与证据协议；张量计算留在上游 harness。
- **N3 不修改上游 `radar-nwp-bench` 仓库**。全程只读。
- **N4 不复现蓝图声称的数字**。蓝图 §1.1 的「61% 结构 deficit 来自相位错配」是对上游 E33 的误读（`MASTER.md:1175, 1750-1753`）；蓝图 §1.1 的 `r=+0.17`、§7 的 `CSI@2.0 ≥ 0.1250` 目标等均需在方法文档中按上游权威证据校正。**澄清（2026-11-08）**：`r=+0.17` 本身是上游承重实测（`MASTER.md:278,282,1613`），并非误读；此处要求校正的是蓝图**据此推出的因果外推**（配准阶梯 ⇒ `r ≥ 0.35`）与 `CSI@2.0 > 0.125` 目标值（上游无出处、且 QPE 与 60min 临近预报是不同任务）。本条不把 `r=+0.17` 判为错误数字。
- **N5 不得写出 `MASTER.md:1767-1770` 的禁写句式**（「突破/破局/建设性解法奏效」「集成稳过地板」「插值容错有效」等）。

### 2.1 作用域与 QLFN 参考实现的范围豁免（2026-11-08 补）

本 SPEC v1.0 写于 2026-11-07，约束对象是 **`qmlbench/` 元数据与证据层**
（baseline 接入、权威表、P1–P6 gate、导出脚本）。上述 N1 / N2 中的
「不实现 Q-LagrangeFlow」「不跑 forward pass」均针对该层。

2026-11-08 用户**另行授权**构建第二个交付物：`qlfn/`
（`QLag-FlowNet` / `QLFN` 可审计参考实现），其需求来源是独立的
`docs/QLFN_SPEC.md`，与 N1/N2 存在字面张力。经 ARIS W2 Round 1 审查指出
该冲突未记录豁免，现明确：

| 约束 | `qmlbench/` 层（本 SPEC） | `qlfn/` 参考实现（`QLFN_SPEC.md`） |
|---|---|---|
| 实现三族机制 | N1 禁止 | 允许作为**工程复现**，文件头必须标注 §9 / §11.8.2 封口；不构成研究路线 |
| forward pass | N2 禁止 | 允许，仅限**形状/梯度自检**（`scripts/qlfn_smoke_test.py`），CPU、tiny patch |
| 训练 / 评测 / 产生分数 | 禁止 | **同样禁止** |
| 修改上游、写权威表 | N3 禁止 | **同样禁止** |

豁免边界：如需把 QLFN 转为训练或评测路线，必须重新立项并显式推翻 N1；
本豁免不覆盖该情形。研究建议的 `BLOCKED` 裁定不因构建参考实现而改变。

## 3. 数据契约

| 项 | 值 |
|---|---|
| 主指标 | `CSI@2.0` mm per 6-min，全场 301×301 汉宁缝合 |
| 阈值集合 | 0.1 / 0.5 / 1.0 / 2.0 / 4.0 / 8.0 / 16.0 mm per 6 min |
| 洁净划分 | n = 538 windows |
| 断帧鲁棒划分 | n = 962 windows，与洁净划分 **case 完全不相交、降水强度不同，禁止直接相减** |
| 地板 | `persistence` = 0.0924（`persistence.eval.json`） |
| Oracle | 完美 QPE + 不动 = 0.1653（E15，用真值起报场，特权信息、不可部署） |
| 可用空间 | 0.0729 |
| 单 seed 极差 | 0.1671 = 可用空间的 2.29×（E13 / §11.9 P1） |

## 4. 目录与交付物

```
qmlbench/           sources.py catalog.py results.py gate.py tables.py __init__.py
configs/            baselines.json（sync 生成）  claims.json（待审查候选）
scripts/            sync_baselines.py  verify_integration.py  export_tables.py
docs/               SPEC.md  CODING_STANDARDS.md  VENUE_METHOD.md
artifacts/          baselines.csv master_table.md coverage.md
                    robustness_table.md decidability.md gate_review.md
```

## 5. 验收命令

```bash
python scripts/sync_baselines.py --check
python scripts/verify_integration.py     # 期望 11/11 PASS，exit 0
python scripts/export_tables.py
```

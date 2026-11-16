# archive · 已退役的证据载体

> 本目录下的一切文件**只读、只作证据**。它们不再承载判断、禁令、准入条件或路线裁决。
> 唯一的结论与逻辑来源是项目根的 [`MASTER_CONTROL_ROOM.md`](../../MASTER_CONTROL_ROOM.md)。
> 归档日期：2026-11-11 · 依据：用户对"文档治理：必须彻底合并，只留一份活地图"的指令。

## 1. 已归档的旧决策文档

过去几周每个 `.md` 都在充当决策大脑，导致同一条「QPE–QPF 解耦拆分」换了名字被
测了三次。以下文件曾承载路线裁决或准入禁令，现全部退役：

| 原路径 | 归档路径 | 曾充当的角色 |
|---|---|---|
| `docs/CONTROL_ROOM.md` | `docs/archive/CONTROL_ROOM.md` | 第二份"主控台"（2026-11-08 快照） |
| `docs/SPEC.md` | `docs/archive/SPEC.md` | qmlbench 需求来源 / 权威顺序裁决 |
| `docs/QLFN_SPEC.md` | `docs/archive/QLFN_SPEC.md` | QLFN 规格，含已被 §1 撤销的 R14「禁止训练/评测」 |
| `docs/QLFN_BLUEPRINT_REVIEW.md` | `docs/archive/QLFN_BLUEPRINT_REVIEW.md` | 蓝图准入评审结论 |
| `docs/VENUE_METHOD.md` | `docs/archive/VENUE_METHOD.md` | 投稿路线与方法裁决 |
| `ARIS.md` | `docs/archive/ARIS.md` | W2 工作流配置与准入状态 |
| `CLAIMS_EVIDENCE_MATRIX.md` | `docs/archive/CLAIMS_EVIDENCE_MATRIX.md` | 主张–证据状态裁定表 |

**留在 `docs/` 的三份是纯证据/工程规范，不是决策文件**：
`CODING_STANDARDS.md`（编码约定）、`QLFN2_EVAL_AUDIT.md`（评测链路审计数值）、
`SLURM_PARTITION_REQUEST.md`（平台事实记录）。

## 2. 用户点名的三份「旧规划」：在上游仓库，本项目不动

用户要求把 `FOUR_TIERS_PLAN.md`、`FIVE_DIRECTIONS_ROADMAP.md`、旧版 `MASTER_*.md`
移进 `docs/archive/`。实测它们**不在本工作区**：

| 名称 | 实际位置 | 处理 |
|---|---|---|
| `FOUR_TIERS_PLAN.md` | `R:/1radarNWPprediction/radar-nwp-bench/docs/FOUR_TIERS_PLAN.md` | **不移动**。属上游仓库，四梯队计划已作废，仅作历史证据 |
| `MASTER.md` | `R:/1radarNWPprediction/radar-nwp-bench/docs/MASTER.md` | **不移动**。这是上游权威数字来源（§10.2），移动会毁掉唯一标尺 |
| `radarNWP-MASTER.md` | `R:/1radarNWPprediction/radar-nwp-bench/docs/radarNWP-MASTER.md` | **不移动**。上游旧版主文档 |
| `FIVE_DIRECTIONS_ROADMAP.md` | 全盘检索无此文件 | 无对象可归档；若日后出现，一律按"不作为现行指导"处理 |

依据：`MASTER_CONTROL_ROOM.md` 的「不改上游」约定，以及用户「要保证不干扰到其他
文件夹」的边界指令。远端同理：`/home/wf/radarNWP` 是上游 bench（只读参照），
本项目的同步落点始终是 `/home/wf/radarNWPQML`。

**结论**：这四份（含不存在的那份）**不得作为现行指导**。任何实验判据、对照、
轮数只能写在 `MASTER_CONTROL_ROOM.md`。

## 3. 合并前 qlfn2 的十个模块（纯代码证据）

`qlfn2_pre_merge/` 存放 2026-11-11 合并前的十个实现文件
（`geometry / units / cappi / operator / motion / halo / model / adapter / bench / losses`）。
自合并起，运行路径只认：

- `qlfn2/qlfn2_model.py` —— 自包含模型（八个模块内联 + NWP 真连线）
- `qlfn2/qlfn2_losses.py` —— 阈值对齐 + 逐时效衰减修复后的 `ThresholdAwareLoss`

另有 3 个**纯别名 shim**（F33 新增）：`qlfn2/adapter.py` /
`qlfn2/model.py` / `qlfn2/losses.py`，每个只把旧 import 路径转发到上面
两个实现文件，不含任何类或函数定义。`check_compat_aliases` 用 `is`
验证它们指向同一对象；**实现仍然只有两个文件**。

归档副本**不在 import 路径上**，不得被任何脚本引用；它们只用于核对
"合并是否改变了原有数学"。

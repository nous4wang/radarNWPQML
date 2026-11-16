# 评测口径审计 · qlfn2 第一轮 sweep

> 生成于 2026-11-09。审计对象：`runs/qlfn2_sweep/` 下第一轮四臂 × 2 seed 的评测。
> 结论：**该轮的绝对值不能与任何上游权威数字比较**，但缺陷的成因已全部定位并量化。
>
> 本文件只记录事实与证据，不写结论性主张。判读纪律见 `MASTER_CONTROL_ROOM.md` §5.3b。

## 一、一句话结论

第一轮 sweep 的四个数字，是用**两把与上游不同的尺子**量出来的：
**取错了 checkpoint 轮次**（评的是崩塌端），且**用错了评测几何**（非上游权威口径）。
两条缺陷方向相反、幅度都没抵消，于是分子停在一个两边都不属于的数上。

## 二、缺陷一：取错了 checkpoint 轮次

### 2.1 上游协议

上游 `scripts/train_keep_epochs.py` 的模块 docstring 写明它存在的理由：

> `scripts/train.py` writes one file, `last.pt`, and rewrites it at the end of
> every epoch. A run that is still training therefore has no stable
> checkpoint to point at: the file under a fixed name changes underneath the
> reader. This wrapper keeps a copy per epoch so the score can be read off a
> known epoch instead of off a moment.

上游 `scripts/epoch_curve_eval.sh` 的注释进一步说明为什么要逐轮评分：

> The old headline (CSI@2.0 = 0.118) came from a checkpoint that the
> training run was still overwriting. Whether the score-aware objective
> peaks early and then decays, or whether 0.118 was a fluke, is a question
> about the epoch, so the epoch is the variable that has to be swept.

### 2.2 上游记录的曲线形状（E19）

| epoch | CSI@0.1 | CSI@0.5 | CSI@2.0 | CSI@4.0 |
|---|---|---|---|---|
| ep00 | 0.4432 | 0.2188 | **0.1226** | 0.0506 |
| ep01 | 0.4818 | 0.1826 | 0.0839 | 0.0386 |
| ep02 | 0.2990 | 0.1286 | 0.0467 | 0.0131 |

ep01 已低于持久性地板 0.0924，ep02 只有地板的 51 %。**峰值在 ep0。**

### 2.3 本轮 sweep 的做法

`scripts/sbatch_qlfn2_exp.sbatch` 用 `--checkpoint last.pt` 评分。
第 10 轮是崩塌端。**每个臂报出的都是崩塌端。**

### 2.4 直接证据（作业 58）

用上游自己的 checkpoint 逐轮评分（协议、评测器、阈值全部是上游的）：

| 轮次 | CSI@2.0 实测（128/128） | CSI@2.0 上游记录（64/64） |
|---|---|---|
| ep00 | 0.1395 | 0.1226 |
| ep01 | 0.1030 | 0.0839 |

ep01/ep00 = 0.738（实测）vs 0.684（上游记录）—— **形状一致**。

## 三、缺陷二：用错了评测几何

### 3.1 几何是怎么进入分数的

`radar_nwp_bench/data/coordinates.py:90`：

```python
def symmetric_patch_origins(grid_dim, patch_size, nominal_stride):
    total_span = grid_dim - patch_size
    n_steps = max(1, math.ceil(total_span / nominal_stride))
    origins = tuple(round(i * total_span / n_steps) for i in range(n_steps + 1))
```

**patch_size 与 nominal_stride 都进入这个式子。**

`radar_nwp_bench/metrics/stitching.py::PatchStitcher` 用 2D Hann 窗加权把 tile
融回 301×301 全场。tile 的大小与位置一变 ⇒ 融合场的值变 ⇒ 列联计数
（hits / misses / false alarms）变 ⇒ CSI 变。

### 3.2 三套几何（网格 301，test 538 窗口）

| patch / stride | 原点 | tiles/窗 | n_samples | 谁在用 |
|---|---|---|---|---|
| **64 / 64** | 0, 59, 118, 178, 237 | **25** | **13450** | **上游全部权威脚本** |
| 128 / 64 | 0, 58, 115, 173 | 16 | 8608 | 本轮 sweep |
| 128 / 128 | 0, 86, 173 | 9 | 4842 | `evaluate.py` 回退 |

### 3.3 上游权威几何 = 64/64 的取证

| 证据 | 内容 |
|---|---|
| `scripts/epoch_curve_eval.sh:37` | `--patch-size 64 --slide-stride 64` |
| `scripts/epoch_check.sh:53` | 同上 |
| `scripts/run_four_tiers_stage2.sh:50` | 同上 |
| `tools/_oracle_advection_ceiling.py:120-121` | 默认值就是 64 / 64 |
| `tools/_eval_four_tiers.py:157` | 显式传 64 / 64 |
| 上游 `physical_qpe/config.json` | `patch_size=64, slide_stride=64` |

### 3.4 陷阱：不传 `--slide-stride` ≠ 默认值

`scripts/evaluate.py:360`：

```python
if (args.slide_stride is None and args.patch_size < min(grid_h, grid_w)
        and not args.centre_only):
    dataset = _build_dataset(args, nwp_vars, radar_vars, args.patch_size)
```

不传 `--slide-stride` 时，评测器**自己按 patch 大小滑**。这是一次静默的口径切换。

### 3.5 量化：同一权重，换几何

| checkpoint | 64/64 | 128/128 | 相对差 |
|---|---|---|---|
| `physical_qpe` ep00 | 0.1226（上游 E19 记录） | 0.1395（作业 58 实测） | **+13.8 %** |
| `physical_qpe` ep01 | 0.0839（上游 E19 记录） | 0.1030（作业 58 实测） | **+22.8 %** |

**越接近地面，几何抬得越多。**

### 3.6 模型不是原因

第 17 条自检（`qlfn2/checks.py`）：patch 128 与 patch 64 的预测在**内部区域
相对差 0.000** ⇒ qlfn2 与 patch 尺寸无关。**几何差异是评测口径问题，不是模型问题。**

## 四、缺陷三（已排除）：真值泄漏

### 4.1 探针设计（三条诊断都"干窗无法伪造"）

1. **汇总计数，不取逐窗 CSI 均值** —— 无雨窗口 hits=misses=false_alarms=0，
   逐窗 CSI 是 0/0=NaN，均值会把结果洗掉；汇总计数不会。
2. **每窗报真值最大值** —— 真值全干的窗口误差恒为 0，**零误差本身什么都不能证明**；
   只有有雨的真值才能见证"复现"。
3. **同一探针跑上游对照 `physical_qpe`** —— 一个已知不读标签的模型。
   对照也出现的模式 = 任务性质，不是泄漏。**这一条承重。**

### 4.2 实测（作业 61，test，64/64，阈值 2.0 mm，24 窗）

| 指标 | qlfn2（臂 0） | physical_qpe（臂 3，上游对照） |
|---|---|---|
| 湿窗 / 干窗 | 9 / 15 | 9 / 15 |
| **湿窗中误差恰为 0 的个数** | **0** | **0** |
| MAE lead 0 | 0.1034 mm | 0.0941 mm |
| MAE lead 9 | 0.1667 mm | 0.1632 mm |
| **误差增长 lead9/lead0** | **1.613×** | **1.734×** |
| pooled CSI@2.0 逐时效 | 0.0204 / 0.0112 / 0.0009 / 0.0027 / 0.0317 / 0.0268 / 0.0141 / 0.0098 / 0.0031 / 0.0000 | 十时效全 0.0000 |

**判读**：误差随时效单调增长，且与对照同量级 ⇒ 真预报的行为，与"读答案"相反。
**泄漏排除。**

### 4.3 附带发现

`physical_qpe` 在 2 mm 档**十个时效的 pooled CSI 全为 0**，而 MAE 曲线与 qlfn2 几乎一样
⇒ 它不是"预测得差"，是**在 2 mm 档完全不报**（欠报塌陷）。
这与"崩塌端"判断互相印证，且来自另一个测量通道。

## 五、三条假说的完整状态

| 假说 | 状态 | 证据 |
|---|---|---|
| (a) 取哪一轮 | **成立** | 上游自己的模型逐轮曲线：ep00 峰值、ep01 掉 26 % |
| (b) 真值泄漏 | **排除** | 误差随时效增长 1.61×，湿窗零误差 0 个 |
| (c) 几何口径 | **成立，且是主因** | 同权重同协议，换几何 +13.8 %～22.8 % |

## 六、修正后的协议（已落地）

要跟上游权威数字比较，**必须同时满足**：

1. **几何 = `--patch-size 64 --slide-stride 64`**
2. **逐轮保存 + 逐轮评分 + 取峰值**（峰值只允许来自 epoch 轮次）

| 部件 | 文件 |
|---|---|
| 逐轮保存 checkpoint | `scripts/train_qlfn2_keep.py` |
| 逐轮评分取峰值 | `tools/score_epoch_curve.py` |
| 正式实验作业 | `scripts/sbatch_qlfn2_curve.sbatch` |
| 提交脚本 | `scripts/run_qlfn2_curve.sh` |
| 端到端复现验证 | `scripts/sbatch_qlfn2_repro.sbatch` |
| 几何对照 | `scripts/sbatch_qlfn2_geom.sbatch` |
| 几何取证探针 | `tools/probe_eval_geometry.py` |
| 几何审计 | `tools/audit_eval_geometry.py` |
| 曲线汇总 | `tools/summarize_qlfn2_curve.py` |
| 几何汇总 | `tools/summarize_geometry.py` |
| 旗标传递自检 | `scripts/check_score_flags.py` |
| 第 17 条自检 | `qlfn2/checks.py` |

**端到端验证（作业 63）**：用上游自己的 checkpoint 在 64/64 下复现 **0.1226**。
能复现 ⇒ 评测链路与上游同源；不能复现 ⇒ 仪器不是上游那台，全部比较作废。

## 七、写死的教训

1. **"跑完了" ≠ "数字对"**。评测取哪一轮、什么几何，都是承重决定，
   且**最容易在提交时被忽略**——脚本里没提，就默默用了默认值。
2. **评测几何必须显式写进作业脚本**；`--slide-stride` 不传 ≠ 默认值。
   日志必须核对 `n_samples` 与 `tiles per window` 两行。
3. **探针脚本先在登录会话跑通再提交**（30 秒 vs 一个 6–11 秒的空作业）。

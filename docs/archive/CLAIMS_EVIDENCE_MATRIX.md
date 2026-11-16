# Claims–Evidence Matrix · 2026-11-08 蓝图审查

> 由 `.aris/templates/CLAIMS_EVIDENCE_MATRIX.md` 填写。`gate claims` 逐行核验。
> Status 枚举以 `aris.py check_claims` 的实现为准，只有
> `supported` / `invalidated` / `reported` / `provisional` 四种；
> 「缺失证据」不是第五种状态，而是登记为 `invalidated` 且语义上 BLOCKED。
> 本轮为**审查**而非投稿，`assurance=draft`，不标注 submission-ready。
>
> **Evidence 列必须是单个、无反引号、无尾注的相对路径**：
> `gate claims` 只对「整格以文件扩展名结尾」的单元格做存在性校验。
> 基线 `b9268d9` 的 C1–C11 已全部是裸路径（实测 11/11 被校验、0 缺失）；
> 本轮修订初稿（未提交）一度写成 `` `.aris/evidence/01.md` §2 `` 这类单元格，
> 会使 17 行全部绕过文件检查，已在提交前自查改回。此处记录的是
> **工具行为约束与本轮自查**，不是基线既存缺陷。补充说明一律移入
> Claim 列或本页脚注。
>
> 修订记录（ARIS W2 Round 1 复审后落地）：
> - 消除「枚举不含 BLOCKED」问题：不发明工具不支持的状态，
>   改为把缺失证据登记为 `invalidated` + BLOCKED 标记；
> - C7 从 `provisional` 改判 `invalidated`（实现中根本没有密度矩阵/偏迹，
>   不是「尚未证实」而是与声称相矛盾）；
> - C9 不再用决策文档自证，改指向独立的 `06-implementation-audit.md`；
> - C10/C11 收窄措辞、指向本机已归档的原始 JSON；
> - 登记 `03-unsupported-claims.md` 中此前遗漏的 6 条（C12–C17）；
> - 规则 3 的计数按修订后的实际条目重算。

## 本机证据路径（本轮新增）

远端 `/home/wf/radarNWPQML/runs/seeds_10_19/` 的 6 份原始产物已下载归档到
`artifacts/remote_seeds_10_19/`：`summary.json`、`independence.json`、
`sensitivity.json`、`queue_config.txt`、`train_status.txt`、`eval_status.txt`。
C10/C11 只引用这些文件，不再依赖远端自述。

## 主表

| ID | Claim | Evidence | Status |
|---|---|---|---|
| C1 | 蓝图三族机制命中上游 §9 / §11.8.2 封口，且例外条款（含「原对照不成立」分支）未触发 | .aris/evidence/01-fact-check.md | supported |
| C2 | 62/39/23 case、BIAS@0.5 1.7820 与 0.9048、r=+0.17、地板 0.0924 / Oracle 0.1653 均有上游实测出处 | .aris/evidence/01-fact-check.md | supported |
| C3 | 「一刀切丢弃 23 个断帧 case 造成条件分布撕裂」 | .aris/evidence/01-fact-check.md | invalidated |
| C4 | 「BIAS 1.78 vs 0.90 是 epoch0 触顶崩塌的根本原因」 | .aris/evidence/01-fact-check.md | invalidated |
| C5 | 27 个 NWP 通道含 850hPa U/V 风矢量，足以支撑风漂移对齐（见 02 §D1） | .aris/evidence/02-engineering-audit.md | invalidated |
| C6 | 上游 evaluator 支持 valid_step_mask，全 62-Case 可与 26 个基线同表比较（见 02 §D4） | .aris/evidence/02-engineering-audit.md | invalidated |
| C7 | 「rank-M 谱压缩与原始非线性动力学严格等价且线性复杂度」（Stone-Weierstrass 只保证一致逼近；实现无偏迹/Schmidt。复审由 provisional 改判） | .aris/evidence/02-engineering-audit.md | invalidated |
| C8 | QLag-FlowNet 能在 10-seed 两级 Bootstrap 下显著超越地板与全部基线（见 03 第 1 条） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C9 | QLFN 代码库可作为不训练 / 不评测 / 不改上游的可审计参考实现；仅由规格与两个自检的结构性结果支撑，属规范性判断而非性能判断 | .aris/evidence/06-implementation-audit.md | provisional |
| C10 | seed 10–19 的 20 份 epoch_00.pt md5 两两互异、与 0–9 零交集、分数多重集非旧副本；这是 20 个互不相同的确定性配置，不是随机抽样 | artifacts/remote_seeds_10_19/independence.json | supported |
| C11 | 该 seed 集合上通过率由 5/10 vs 4/10 变为 7/10 vs 4/10；Fisher 单侧 p=0.18492498213860442，重采样差值 95% 区间 [-0.1, 0.7] 跨 0，supports_improvement_claim=false（另见同目录 sensitivity.json） | artifacts/remote_seeds_10_19/summary.json | reported |
| C12 | 「配准阶梯把 r 从 +0.17 提升到 ≥ +0.35」（见 03 第 2 条） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C13 | 「单步 QPE CSI@2.0 > 0.125」（见 03 第 3 条） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C14 | 「从数学结构上保证 CSI@0.1 / 0.5 列联表计数完全不被极值分支破坏」（见 03 第 4 条、02 §T3） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C15 | 「S_vN 是解析物理探针，无需 SHAP 即可给出物理贡献率」（见 03 第 5 条、02 §P2/P3） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C16 | 「62-Case 44/9/9 分层划分产生 ≥2800 有效滑窗」（见 03 第 6 条） | .aris/evidence/03-unsupported-claims.md | invalidated |
| C17 | 「恒等重构误差 <1e-6、显存降 ≥50%、吞吐提速 ≥2.2×」（见 03 第 7 条、02 §G1/G2） | .aris/evidence/03-unsupported-claims.md | invalidated |

## 已撤回的审查结论（不计入上表）

| 原编号 | 撤回理由 | 归档位置 |
|---|---|---|
| 旧 S1「`decode_101_to_301(z_extreme_101, nn.Identity())` 96 通道 → 必然 shape 崩」 | **硬误报**。蓝图 `LatentConditionalFlowHead.out_ch = t_out * 9 = 90`，`integrate_ode` 返回 90 通道；`view(-1, 9, H, W)` 得到 `(B*10, 9, H, W)`，`PixelShuffle` 后 `view(B,-1,1,301,301)` 合法，不会抛错。真正问题是 T1（CFM 无训练目标）与缺少通道数断言 | `docs/QLFN_BLUEPRINT_REVIEW.md` §1.1；`.aris/rounds/round-01.md` |
| 旧 T2「蓝图 masked fp/fn 在缺失帧虚假计数」 | **硬误报**。蓝图 `p_hit`/`t_hit` 已先乘 binary `mask`，对 m∈{0,1} 两式代数等价（见 §1.1 推导）。QLFN 的改写只是等价重写，不是修缺陷 | 同上 |

## 规则

1. 每条 claim 必须有证据；缺失证据一律登记为 `invalidated` 并在 Claim 列
   写明 `BLOCKED`，不得进入正文。不发明 `aris.py` 不认识的第五种状态。
2. 措辞强度不得超过证据允许的强度；`reported` 不得写成 `supported`，
   `provisional` 不得写成 `supported`。
3. **不得用于任何改进声明的条目共 13 条**：12 条 `invalidated`
   （C3–C8、C12–C17）+ 1 条 `provisional`（C9）。C11 为 `reported`，
   只允许按原始 JSON 的数字引用，禁止外推为「qpe_0 优于 base_concat」
   或「seed 编号决定通过率」的因果结论。
4. 本轮为**同族审查**，结论等级为 `provisional`，只写「在当前证据与上游规则下
   不予立项」，不写「已证明不可行」。
5. C10 的「独立」仅指与 0–9 批次无 byte/score 重叠；固定 seed 在本管线是
   bit-exact 确定性映射，因此**不是统计意义上的随机抽样**
   （`sensitivity.json.caveat`）。重采样区间只描述 seed 集合敏感性，不是置信区间。
6. `supports_improvement_claim = False` 恒定：目前只有两臂基线，
   §11.9 P4 要求的候选臂、逐 seed 配对差值与两级配对 bootstrap 仍缺。

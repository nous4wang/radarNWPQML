# ARIS 项目配置

> 由 `aris.py init` 生成，等价上游 `CLAUDE.md` 的配置职责；可自由修改。

- 研究方向：2026-11-08 新蓝图研究建议审查：Q-LagrangeFlow / QLag-FlowNet（QLFN）的可实现性与证据合规性
- 当前工作流：W2
- 目标会议 / 期刊：CVPR/ICML main track (蓝图主张)
- effort：balanced · reviewer：auto
- 建立日期：2026-11-08

## 计算环境

- GPU 目标：local / remote SSH / vast（未配置时，实验类修复一律标记为人工跟进）
- 远程主机 / 环境路径 / 数据目录：
- 单次实验 GPU 小时上限：4（超过即跳过并标记）

## 数据源

- 允许来源：本地 PDF、Zotero、Obsidian、arXiv、WebSearch / WebFetch、知识库
- 引用策略：只使用真实 DOI / arXiv ID 与 DBLP / CrossRef 元数据，禁止生成式 BibTeX
- 敏感数据：不得外传到未经批准的外部服务

## 验收偏好

- AUTO_PROCEED：true（选择点报告后继续；false 则逐点确认）
- human checkpoint：false
- 提交标准：assurance=submission 时强制 proof / claim / citation 三项审计全部 PASS

## 不可协商的约束

1. 不编造文献、实验结果或评审意见。
2. 不隐藏弱点以抬高评分，不降低验收标准。
3. 同族审查结论一律标记 provisional。
4. 每个结论必须能指向本地证据文件。
5. 审查可以驱动流程，但不得宣告"无罪"。

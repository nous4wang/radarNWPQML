# 编码标准（qmlbench）

本文件是 `R:/1radarNWPQML` 的编码约定，供两轴代码审查的 Standards 轴比对。

## 1. 依赖

- **零第三方依赖**。本机无 torch / numpy / PyYAML，所有模块必须只依赖 Python 标准库。
- 上游 `radar-nwp-bench` 只能作为**数据**读取（扫描 decorator、读 YAML 文本、读 JSON 报告、读 Markdown 表），**不得 import** 其任何模块。
- 上游路径只能经 `BenchSource.locate()` 解析：`--bench-root` > 环境变量 `RADAR_NWP_BENCH_ROOT` > 默认路径。禁止在库代码里硬编码绝对路径。

## 2. 数值与证据纪律

- **每个承重数字常量必须有来源注释**，写明上游文件与章节（例如 `E15-oracle-advection-ceiling.md` §4.15.2）。禁止把表格数字手工转录进代码。
- **score-aware 表必须由 `RESULTS_FINAL.md` §10.1 解析得到**，不得硬编码 18 行数值。
- **标准损失表与 score-aware 表禁止合并排名**；任何表格输出必须附口径声明。
- **阈值单位一律写 mm per 6-min accumulation**，不得写 mm/h；`2.0` 对应 20 mm/h。
- **洁净划分与鲁棒划分的分数禁止相减**（两组 case 不相交、强度不同）。
- **缺失证据 = FAIL，不得视为 PASS**。资格门里 `None` 一律按未报告处理。
- 不得写出 `MASTER.md:1767-1770` 的禁写句式；结论须区分「实测」「机制层」「推测/未测」「已证伪」。

## 3. 代码风格

- Python 3.10+ 语法，`from __future__ import annotations`，类型注解完整。
- 数据记录用 `@dataclass(frozen=True)`；记录必须**可哈希**（不得含可变 `list` 字段，用 `tuple`）。
- 路径操作一律 `pathlib.Path`，读写显式 `encoding="utf-8"`。
- 公开函数/类的 docstring 说明 `Args` / `Returns` / `Raises`；模块 docstring 说明该模块的职责边界。
- 常量集中放在模块顶部或专门模块，不散落在函数体内。
- 不引入 spec 未要求的抽象、钩子或参数（无 speculative generality）。
- 标识符与代码注释用英文；面向用户的文档用简体中文。
- 脚本须支持从任意工作目录运行（自行把项目根插入 `sys.path`），并提供 `--bench-root`。

## 4. 模块边界

| 模块 | 只负责 | 明确不做 |
|---|---|---|
| `sources.py` | 定位上游、静态扫描、解析权威 Markdown 表 | 不解释分数、不实现判据 |
| `catalog.py` | 类型化清单、分组、结构校验 | 不读上游报告 |
| `results.py` | 分数提取、排名、可判定性度量 | 不裁定方法是否合格 |
| `gate.py` | P1–P6 与 §11.2 规则判定 | 不格式化输出、不替用户下科学结论 |
| `tables.py` | Markdown / CSV 渲染 | 不计算新的分数或判据 |

渲染与判定必须分离：任何模块不得既算出裁定又格式化它。

## 5. 可重复性

- `sync_baselines.py` 幂等，`--check` 能发现上游漂移且返回非零。
- `verify_integration.py` 覆盖：结构完整性、蓝图 14 个、adapter/config/报告覆盖、地板常量一致性、资格门自检；失败返回非零。
- 生成物放 `artifacts/`，不放进库代码目录。

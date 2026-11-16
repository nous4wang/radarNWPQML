# 管理员操作说明 · 开放 shared 分区以启用 GPU 批作业

> **状态（2026-11-09 17:20 更新）**：**分区在提交时已是 `State=UP`，本文件
> 保留为应急预案，不再需要执行任何命令。**
> 证据：作业 29 起全部提交成功，作业 34/36/37 均正常取到 GPU 并跑完
> （`CUDA_VISIBLE_DEVICES=0`，RTX 4090，23.5 GiB）。
> 若将来分区再次被关闭（例如 `State=INACTIVE`），再按下述说明操作。

**申请人**：wf（研究账号）
**日期**：2026-11-09
**平台**：nbdx / Slurm 23.11.4 / 集群 zywt / 分区 shared

---

## 一句话请求

```bash
sudo scontrol update PartitionName=shared State=UP
```

执行后本项目立即提交 GPU 作业，不需要其他改动。

---

## 背景：为什么 GPU 在登录会话里不可见

平台规则 §2.3 写明"登录会话不能直接访问 GPU"，§11.1 记录了"单卡作业仅能
访问已分配显卡，PyTorch 运算成功"。本项目实测与之一致：

```
$ python -c "import os; os.open('/dev/nvidiactl', os.O_RDWR)"
/dev/nvidiactl: [Errno 1] Operation not permitted
```

`/dev/nvidia0..7` 的权限位是 `crw-rw-rw-`，设备节点本身正常；驱动侧也正常
（内核 NVRM 595.84 与用户态 `libcuda.so.595.84` 一致，`cuDriverGetVersion`
返回 rc=0 / ver=13020）。**GPU 硬件与驱动均无故障**，只是被 Slurm 的设备
cgroup 挡在登录会话之外。

因此此前"驱动层故障、需 `nvidia-smi -pm 1` / `rmmod` / 重启"的判断不成立，
**不需要任何驱动或内核层面的改动**。

---

## 需要开放的原因

当前 `shared` 分区为 `State=INACTIVE`，任何提交都被拒：

```
$ sbatch test.sbatch
sbatch: error: Batch job submission failed: Required partition not available (inactive or drain)
```

按平台 §10.3，临时开放用 `scontrol update PartitionName=shared State=UP`
即可；§10.3 也提到配置文件中 `State=INACTIVE` 改 `State=UP` +
`scontrol reconfigure` 是持久做法。**本项目只请求试运行期间的临时开放**，
并在作业结束后按 §10.3 的 `sudo scontrol update PartitionName=shared State=INACTIVE`
恢复——请管理员在方便时告知是否需要我们自行恢复，或由您代为恢复。

---

## 提交前的验证（已完成，无需管理员参与）

| 检查 | 结果 |
|---|---|
| `sbatch --test-only scripts/sbatch_qlfn2_pilot.sbatch` | `Job 27 to start at 2026-11-09T17:06:56 using 8 processors on nodes nbdx in partition shared` |
| `sbatch --test-only --array=0-2 scripts/sbatch_qlfn2_exp.sbatch` | `Job 28 to start at 2026-11-09T17:06:56 using 8 processors on nodes nbdx in partition shared` |
| 账号关联 | `sacctmgr show assoc where user=wf` → `wf / research / standard` ✓ |
| 解释器 | `/home/wf/.conda/envs/radar_nwp/bin/python`（个人环境，未改公共 Conda） |
| 依赖 | torch 2.5.1（CUDA 12.4 构建）、numpy 1.26.4、scipy 1.17.1、sklearn 1.9.1 |
| 数据 | `/home/wf/data/radarNWP/BJ_mmap`（只读，62 个 case） |

---

## 我们承诺的资源用量（在平台限额内）

| 项 | 本项目用量 | 平台限额 |
|---|---|---|
| 单作业 GPU | 1 | ≤6 |
| 单作业 CPU | 8 逻辑（4 物理核） | ≤96 逻辑 |
| 单作业内存 | 32 GiB | ≤384 GiB |
| 单作业时长 | pilot 40 min；正式 6 h | ≤30 h |
| 同时运行作业 | ≤4（4 个臂各 1 张卡） | ≤4 |
| 已提交作业 | 1（pilot）+ 4（正式队列的 4 个数组作业） | ≤12 |
| GPU 合计 | ≤4 | ≤6 |

正式实验用 **4 个数组作业**（每个臂一个数组），数组子任务按平台规则计入
提交数量。3 个 seed × 4 臂 = 12 个子任务，正好在"同一用户已提交 ≤12"的
边界内；若管理员希望更保守，可先只跑 pilot。

---

## 作业规格

```
#SBATCH --partition=shared
#SBATCH --account=research
#SBATCH --qos=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00        # pilot；正式实验 06:00:00
#SBATCH --output=run_logs/slurm/%x-%j.out
```

作业脚本遵守平台规则：用自己的 Conda 环境、日志目录提交前创建、
保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`、不修改 GPU 可见性、
不使用 nohup 或后台进程。

---

## 联系方式

作业号与日志路径：项目仓库 `/home/wf/radarNWPQML`，日志 `run_logs/slurm/`。
若作业异常终止，请把作业号、`run_logs/slurm/<jobname>-<jobid>.err` 的内容
和发生时间发回。

# 交接说明 —— 终端断开后如何继续

> 生成时间：2026-09-21 21:31
> 场景：你的终端即将断电，所有长任务已挂到服务器后台，**脱离终端运行**。

---

## 0. 「我关了终端，Claude 还会继续跑吗？」—— 分成两件事

| | 会继续吗 | 说明 |
|---|---|---|
| **后台进程** | ✅ **会** | 已 `setsid` + `nohup` + `ppid=1`，与终端、与 Claude 会话都无关 |
| **Claude 我** | ❌ **不会** | 终端一关会话即结束，我无法再自己发起任何动作 |

所以：**光挂着下载是不够的** —— 下载完之后没人去做解压/抽取/去污染。
为此我已经把**整条链**也挂到后台了（`scripts/run_pipeline.sh`），
它会自己等下载完成，再依次跑完后续步骤。**你什么都不用做。**

---

## 1. 现在正在跑什么（已脱离终端，断电不影响）

所有进程 `ppid=1`（被 init 收养），**关闭终端不会中断**。

| 脚本 | 内容 | 预计 |
|---|---|---|
| `run_pipeline.sh` | **总控**：等下载 → 解压 → 视频抓取 → 抽取 400K → 去污染 → 统计 | 数小时，全自动 |
| `01b_images.sh` | COCO 19.3G + GQA 21.8G + TextVQA 7.1G + VG ~19G | 约 1–1.5 小时 |
| `00_model.sh` | Qwen2-VL-2B-Instruct (~4.4G) | 较慢，随其它流结束加速 |
| `04_bench.sh` | MME / MathVista / Video-MME 标注（主体已完成） | 快 |

**查看进度**：

```bash
cd /home/ml-user/workdir/project-2
bash scripts/download/resume_all.sh --status   # 各数据集完成情况
tail -30 logs/pipeline.log                     # 总控流水线走到哪一步了
```

**万一进程被杀 / 机器重启，一条命令补齐**（幂等，可反复跑）：

```bash
bash scripts/download/resume_all.sh            # 补齐下载
setsid nohup env CUDA_VISIBLE_DEVICES="" bash scripts/run_pipeline.sh \
  > logs/pipeline.log 2>&1 < /dev/null &       # 重新挂上总控
```

---

## 2. ⚠️ 关于你的训练进程（重要）

`project-1/baseline/` 下的两个训练任务**全程未被触碰**：

| PID | 任务 | GPU |
|---|---|---|
| 38325 | `run_train.py --config configs/formal250.json` | 0,1,2,3 |
| 109792 | `red/scripts/run_red.py --config configs/red250.json` | 4,5,6 |

本次全部工作在 CPU 上完成：**没有 import torch、没有创建 CUDA context、
没有申请任何显存**。所有下载脚本入口都强制 `CUDA_VISIBLE_DEVICES=""`。

> 注意：`project-1/baseline/AGENTS.md` 里写的存储根目录
> （`/mnt/shared-storage-user/...`、`/data/VPO-RM`）**在本机不存在**，
> 那份 AGENTS.md 应该是从别的环境带过来的。本项目产出都在 `project-2/` 内。

---

## 3. 已经完成的（可直接用）

| 产出 | 位置 | 状态 |
|---|---|---|
| CPU 数据环境 | `.venv/`（无 torch-cuda） | ✅ |
| 数据清单 | `data/manifests/datasets.yaml` | ✅ |
| **RLVR 数据 25,748 条** | `data/processed/rlvr/*.parquet` | ✅ |
| **DPO 偏好对 5,731 条** | `data/processed/dpo_10k.jsonl` | ✅（见下方缺口说明） |
| 奖励函数 + 单测 | `configs/reward_geo.py` | ✅ 7/7 用例通过 |
| 训练配置 ×4 | `configs/`（严格按 PLAN 的 4×L40） | ✅ |
| 环境脚本 3 套 | `env/server_l40_{sft,rl,bench}.sh` | ✅ |
| 环境自检 | `env/check_env.py`（默认不占显存） | ✅ |
| 下载/续传工具 | `scripts/download/resume_all.sh` | ✅ |
| 视频抓取器 | `scripts/prepare/fetch_videos.py` | ✅ 可用率实测 80% |
| 文档 00–05 | `docs/` | ✅（06 待补） |

### 3.1 DPO 缺口：5,731 / 10,000

`zhiqings/LLaVA-RLHF-Data` **实测不含显式的 chosen/rejected 字段** ——
它给的是每张图的 `captions[]` 候选列表，没有标注哪个更好，README 也没有排序约定。

**我没有凭空假设「第一条最好」**（错误偏好标签会直接毁掉 DPO）。
缺口 4,269 对需要你决策，三个选项：

1. **接受 5,731 对**，如实报告（RLHF-V 质量高，5.7K 对做 DPO 是够的）；
2. **换一个有显式标注的偏好数据集**（我可以帮你找）；
3. 加 `--allow-llava-rlhf-heuristic` 强行推导 —— **会污染偏好信号，不推荐**，
   脚本会在输出里加 `assumption` 字段标记。

---

## 4. 还没做的（需要你决策或等下载完）

| # | 事项 | 触发条件 |
|---|---|---|
| 1 | 解压图像归档 | COCO/GQA/TextVQA/VG 下完之后 |
| 2 | SFT 400K 抽取 | 图像解压完之后 |
| 3 | 质量过滤 + 去污染 | 上一步之后 |
| 4 | 视频抓取（<3 分钟） | 可随时开始，与下载并行 |
| 5 | 训练配置验证 | **本机无空闲卡，从未跑过** |

### 4.1 数据管线一键脚本（下载完成后按顺序跑）

```bash
cd /home/ml-user/workdir/project-2

bash scripts/download/01c_extract_images.sh                 # 解压+完整性校验
.venv/bin/python scripts/prepare/to_sharegpt.py             # 抽 400K
.venv/bin/python scripts/prepare/quality_filter.py          # 过滤+去污染
.venv/bin/python scripts/prepare/stats.py                   # 统计报告
less data/manifests/stats_report.md                         # ★ 必看
```

### 4.2 视频（你的需求：VideoInstruct 中 <3 分钟的视频）

```bash
# 先小样本验证（12 个）
.venv/bin/python scripts/prepare/fetch_videos.py --limit 12 --stage probe

# 全量：探测 13,303 个视频时长 -> 筛 <3min -> 下载
setsid nohup env CUDA_VISIBLE_DEVICES="" .venv/bin/python \
  scripts/prepare/fetch_videos.py --max-minutes 3 --workers 4 \
  > logs/fetch_videos.log 2>&1 < /dev/null &
```

**实测数据**：20 个抽样视频中 **16 个可用（80%）**，4 个失败
（下架 2 / 私享 1 / 超时 1）。所以 13,303 个视频预计可拿到约 1 万个。
失败原因会分类统计，不会隐藏。

> ⚠️ 一个已修复的坑：`video_id` 形如 `v_k_ZXmr8pmrs`，其中 `v_` 只是
> ActivityNet 的标记，**真正的 YouTube ID 是 `v_` 之后的部分**。不剥前缀
> 会让**每一个**视频都返回 "This video is unavailable"，看起来像数据全废，
> 实际只是拼错了 URL（实测 0% → 80%）。

---

## 5. 必须知道的三个风险

### 5.1 训练配置从未在 GPU 上验证过

受「不占用显存」约束 + 本机 7 张卡全被占用，`configs/` 下的四个配置
**只做了结构与取值检查，从未真实运行**。上机后请务必先跑 100 step 吞吐校准：

```bash
NPROC=4 bash scripts/train/launch_4xl40.sh sft --max_steps 100
```

### 5.2 训练前必查 reward 是否恒为 0

RLVR 的答案格式在各数据集间不统一（`answer` vs `solution`，裸值 vs `<answer>` 标签）。
若不处理，GRPO 会**照常跑满 200 step、loss 看着正常、但奖励恒为 0**。
`configs/reward_geo.py` 已处理并有单测，但上机后仍建议抽样人工核对几条 reward。

### 5.3 去污染可能未生效

`quality_filter.py` 依赖从 MME/MathVista parquet 里提取图像指纹。
若提取数为 0，脚本会在报告中显式写「**未生效**」——
**此时不得声称已去污染**（分数会虚高）。

---

## 6. 一键恢复清单（新终端打开后）

```bash
cd /home/ml-user/workdir/project-2

# 1) 看下载状态
bash scripts/download/resume_all.sh --status

# 2) 补齐未完成的下载（幂等）
bash scripts/download/resume_all.sh

# 3) 看训练进程是否还活着（不要动它们）
ps -o pid,args -p 38325,109792
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 4) 看本文档
less HANDOFF.md
```

---

## 7. 问题记录

本次遇到的 15 个问题（含 3 个静默失败）全部记录在
**`docs/05-下载准备搭建记录.md`**，其中最重要的三个：

| 编号 | 问题 | 为什么危险 |
|---|---|---|
| **P1** | 三个 SFT 数据源只有标注、没有图像/视频 | PLAN 的下载策略**根本跑不通** |
| **P6** | RLVR 答案列名/格式不统一 | **静默失败**：奖励恒 0 但训练"正常"跑完 |
| **P7** | `images` 是 numpy.ndarray 不是 list | **静默失败**：2,101 条数据被全部丢弃 |

另见 `docs/02-踩坑清单.md`（症状 → 原因 → 解法，含上机自查清单）。

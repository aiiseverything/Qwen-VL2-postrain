# Postrain — 基于 Qwen2-VL-2B 的 SFT / DPO / GRPO

数据准备、训练配置、评测与指南的完整工程。执行依据为仓库内的 `PLAN.md`。

> ⚠️ **先读 [`HANDOFF.md`](HANDOFF.md)** —— 后台任务状态、如何续跑、必须知道的风险。

---

## 快速开始

```bash
# 1) 看下载/流水线状态
bash scripts/download/resume_all.sh --status
tail -30 logs/pipeline.log

# 2) 补齐未完成的下载（幂等）
bash scripts/download/resume_all.sh

# 3) 数据流水线（若总控没在跑）
bash scripts/run_pipeline.sh

# 4) 训练（★ 先跑 100 step 校准吞吐）
NPROC=4 bash scripts/train/launch_4xl40.sh sft --max_steps 100
NPROC=4 bash scripts/train/launch_4xl40.sh sft
NPROC=4 bash scripts/train/launch_4xl40.sh dpo
NPROC=4 bash scripts/train/launch_4xl40.sh grpo
```

---

## 目录结构

```
project-2/
├── HANDOFF.md              ★ 交接说明（先读这个）
├── PLAN.md                 原始计划
│
├── docs/                   ★ 指南本体
│   ├── 00-执行路线图.md      怎么做（含 go/no-go 判据）
│   ├── 01-学习路径.md        学什么（按依赖顺序）
│   ├── 02-踩坑清单.md        注意什么（症状→原因→解法）
│   ├── 03-服务器部署手册.md   逐条命令
│   ├── 04-实验记录模板.md     消融记录
│   └── 05-下载准备搭建记录.md ★ 本次实况 + 15 个问题
│
├── configs/
│   ├── sft_qwen2vl_2b.yaml          SFT（冻结 ViT，2 epoch）
│   ├── dpo_qwen2vl_2b.yaml          DPO（基于 SFT ckpt，1 epoch）
│   ├── grpo_qwen2vl_2b.yaml         GRPO（200 step，EasyR1）
│   ├── ds_zero2.json                DeepSpeed ZeRO-2
│   ├── reward_geo.py                RLVR 奖励函数（含单测）
│   └── dataset_info.json            LLaMA-Factory 数据集注册
│
├── scripts/
│   ├── download/           00_model / 01_sft / 01b_images / 01c_extract
│   │                       02_dpo / 03_rlvr / 04_bench / resume_all
│   ├── prepare/            common / to_sharegpt / build_dpo_pairs
│   │                       build_rlvr / quality_filter / stats / fetch_videos
│   ├── train/              launch_4xl40
│   ├── eval/               (待补)
│   ├── run_pipeline.sh     ★ 无人值守总控
│   └── status
│
├── env/                    三套独立 venv（规避依赖冲突）
│   ├── server_l40_{sft,rl,eval}.sh
│   ├── requirements-server-*.txt
│   └── check_env.py        环境自检（默认不占显存）
│
├── data/
│   ├── raw/                原始下载（不进 git）
│   ├── processed/          统一格式数据
│   ├── samples/            小样本
│   └── manifests/
│       ├── datasets.yaml   数据清单（ID/配比/许可/风险）
│       └── stats_report.md 统计报告（自动生成）
│
└── results/
    ├── checkpoints/        训练产物
    └── report.md           最终对比表
```

---

## 数据规模

| 阶段 | 目标 | 实际 |
|---|---|---|
| SFT | 400,000 | 待流水线产出 |
| DPO | 10,000 | **5,731**（缺口原因见 `docs/05` P15） |
| RLVR | — | **25,748**（geometry3k 2,101 + geoqa 8,031 + MMK12 15,616） |
| 视频 | VideoInstruct 中 <3 min | 待抓取（实测可用率 **80%**） |

---

## 你需要知道的三个硬事实

1. **PLAN 的环境描述与实际不符。** PLAN 写的是「无 GPU 的 Mac + 4×L40 服务器」，
   实际是一台 **7×L20 的 Linux 机器**（`docs/05` 第 1 节 E1–E6）。

2. **三个 SFT 数据源只有标注、没有图像/视频。** 图像要从 COCO/GQA/TextVQA/VG
   单独下载，视频要从 YouTube 抓。PLAN 的下载策略照原样跑不通（`docs/05` P1/P2）。

3. **训练配置从未在 GPU 上验证过。** 本机 7 张卡全部被别的任务占用，
   且用户要求不得占用显存，因此只做了结构级检查。
   上机务必先跑 100 step 吞吐校准（`docs/03` 第 5.1 节）。

---

## 环境要求

| 组件 | 版本 |
|---|---|
| Python | 3.12（数据侧）/ 3.11（训练侧 venv） |
| 训练 | LLaMA-Factory (SFT/DPO) + EasyR1 (GRPO) + VLMEvalKit (评测) |
| 硬件 | 训练目标 4×L40；显存估算见 `docs/03` 第 8 节 |

⚠️ **三个 venv 必须分开装**：LLaMA-Factory 与 EasyR1 对
`transformers`/`vllm` 的版本要求冲突（PLAN K2）。

---

## 许可风险

| 数据集 | 许可 | 风险 |
|---|---|---|
| Lin-Chen/ShareGPT4V | CC-BY-NC-4.0 | **禁商用** |
| openbmb/RLHF-V-Dataset | CC-BY-NC-4.0 | **禁商用** |
| zhiqings/LLaVA-RLHF-Data | CC-BY-NC-4.0 | **禁商用** |
| liuhaotian/LLaVA-Instruct-150K | CC-BY-4.0（受 OpenAI 条款约束） | 中 |
| MBZUAI/VideoInstruct-100K | CC-BY-SA-4.0 | 中 |

完整清单见 `data/manifests/datasets.yaml`。

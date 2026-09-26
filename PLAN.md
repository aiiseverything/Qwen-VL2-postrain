# 执行计划：Postrain — 基于 Qwen2-VL-2B 的 SFT 与强化学习

> **本文档状态：待审批。截至本文写成，`Project-1/` 内的任何文件都未被改动、未被删除。**
> 本文件 `PLAN.md` 是我唯一新建的文件，它本身会在 Phase 0 随目录重建一起被清理。
>
> 文档生成日期：2026-09-20
> 所有环境数据与仓库路径均为**实测核实**结果，非记忆或假设；核实方式见附录 A。

---

## 目录

- [1. 项目目标拆解](#1-项目目标拆解)
- [2. 事实核查结果](#2-事实核查结果)
- [3. 关键设计决策](#3-关键设计决策)
- [4. 风险登记表](#4-风险登记表)
- [5. 目标目录结构](#5-目标目录结构)
- [6. 分阶段执行步骤](#6-分阶段执行步骤)
- [7. 数据方案详表](#7-数据方案详表)
- [8. 训练超参方案](#8-训练超参方案)
- [9. 评测方案](#9-评测方案)
- [10. 指南文档交付大纲](#10-指南文档交付大纲)
- [11. 需要你提供的东西](#11-需要你提供的东西)
- [12. 时间线与里程碑](#12-时间线与里程碑)
- [13. 审批清单](#13-审批清单)
- [附录 A：核实方法](#附录-a核实方法)

---

## 1. 项目目标拆解

把项目描述逐条拆成可验收的交付物。**「谁执行」一列很重要**：本机无 GPU，带 ★ 的必须在 4×L40 上跑，我只能交付代码而不能代你验证。

| # | 项目描述中的要求 | 交付物 | 谁执行 |
|---|---|---|---|
| R1 | 从 LLaVA-665K、ShareGPT4V、Valley 取约 30 万条做 SFT | 下载脚本 + 配比采样脚本 + 统一格式 jsonl | 我（本机小样本）+ ★服务器（全量） |
| R2 | 基于 RLHF-V、LLaVA-RLHF 构建约 1 万条偏好对做 DPO | 偏好对构建脚本 + 1 万条 DPO 数据 | 我（本机全量，数据量小） |
| R3 | 用 Geometry3K、GEOQA-8K、MMK12 构建可验证奖励数据做 RLVR | RLVR 格式转换 + 奖励函数 | 我（本机全量） |
| R4 | 数据格式统一与质量过滤 | 过滤脚本 + 数据统计报告 + 去污染检查 | 我 |
| R5 | DeepSpeed ZeRO-2，4×L40 分布式训练 | `ds_zero2.json` + 多卡启动脚本 + 显存估算 | 我（写）/ ★你（跑） |
| R6 | SFT 阶段冻结视觉编码器，30 万数据训 2 epoch | `sft_qwen2vl_2b.yaml` | 我（写）/ ★你（跑） |
| R7 | 基于 SFT checkpoint 用 DPO 训 1 epoch | `dpo_qwen2vl_2b.yaml` | 我（写）/ ★你（跑） |
| R8 | RLVR 阶段用 GRPO 训 200 step | `grpo_qwen2vl_2b.yaml` + 奖励函数 | 我（写）/ ★你（跑） |
| R9 | 在 MME、Video-MME、MathVista 上评测 pretrained / SFT / RL 三个模型 | VLMEvalKit 接入 + 一键脚本 + 对比表生成器 | 我（写+本机小规模验证）/ ★你（全量） |
| R10 | **指南：怎么做、学什么、注意什么** | `docs/` 下 5 篇文档 | 我 |

---

## 2. 事实核查结果

### 2.1 本机环境（实测）

| 项 | 实测值 | 对计划的约束 |
|---|---|---|
| 机型 | `Darwin ... arm64 T8142`，MacBook Air, Apple Silicon | **无 CUDA。训练完全不可能在本机进行** |
| Python | 仅 `/usr/bin/python3` = **3.9.6**；无 conda / uv / pyenv | LLaMA-Factory 与 verl 要求 3.10+，**必须先装独立 Python** |
| 磁盘 | 460 GB 总量，**剩余 306 GB** | 够放小样本与评测集，不够放全量 SFT 图像 |
| 下载工具 | 有 `curl` / `git`；**无 `wget`、无 `aria2c`** | 下载脚本不能依赖 aria2 多线程加速 |
| 沙箱 | 文件写入被限制在 `Project-1/` 内 | **不能写 `~/.local`、不能改系统 Python**；工具必须装进工作区 `.tools/` |

### 2.2 网络连通性（实测 HTTP 状态码 / 耗时）

| 站点 | 结果 |
|---|---|
| `huggingface.co` | **200**, 1.25 s（可直连，无需镜像） |
| `hf-mirror.com` | **200**, 0.40 s（更快，作为加速备选） |
| `modelscope.cn` | 302（可用） |
| `pypi.org` | **200**, 0.39 s |

> 结论：**直连可用**。脚本默认走官方源，用环境变量 `HF_ENDPOINT` 一键切镜像，不写死。

### 2.3 仓库路径核查（⚠️ 有 6 处已改名，用旧 ID 会跳转或失败）

**模型**

| 用途 | 已核实 ID | 状态 |
|---|---|---|
| 基座 | `Qwen/Qwen2-VL-2B-Instruct` | ✅ 200，下载量 2,050,601，非 gated |

**训练数据**

| 阶段 | 已核实 ID | 状态 |
|---|---|---|
| SFT | `liuhaotian/LLaVA-Instruct-150K` | ✅ 665K 混合的 `llava_v1_5_mix665k.json` 在此仓库内 |
| SFT | `Lin-Chen/ShareGPT4V` | ✅ 200；**许可 CC BY-NC 4.0，禁止商用** |
| SFT | ⚠️ `luoruipu1/Valley-Instruct-**65k**` | 旧 ID `Valley-Instruct-73k` 会 **307 跳转**到 `-65k`；**`gated=auto`，需 HF token** |
| DPO | ⚠️ `openbmb/RLHF-V-Dataset` | 常见的 `HaoyeZhang/RLHF-V-Dataset` 已 **307 迁移**至此 |
| DPO | `zhiqings/LLaVA-RLHF-Data` | ✅ 200 |
| RLVR | `hiyouga/geometry3k` | ✅ 200，下载量 41,495（EasyR1 官方示例数据） |
| RLVR | `leonardPKU/GEOQA_R1V_Train_8K` | ✅ 200 |
| RLVR | `FanqingM/MMK12` | ✅ 200 |

**评测数据**

| Benchmark | 已核实 ID | 状态 |
|---|---|---|
| MME | ⚠️ `lmms-lab-encoder/MME` | `lmms-lab/MME` 已 **307 迁移**至此 |
| Video-MME | ⚠️ `lmms-eval/Video-MME` | `lmms-lab/Video-MME` 已 **307 迁移**至此 |
| MathVista | `AI4Math/MathVista` | ✅ 200，非 gated |

**框架**

| 框架 | 已核实 | 状态 |
|---|---|---|
| LLaMA-Factory | ⚠️ `hiyouga/LlamaFactory`（74,902 ★） | 旧名 `LLaMA-Factory` **301 跳转** |
| EasyR1 | `hiyouga/EasyR1`（5,166 ★） | ✅ 活跃，最近推送 2026-09-19 |
| verl | ⚠️ `verl-project/verl`（23,500 ★） | `volcengine/verl` **301 跳转** |
| VLMEvalKit | `open-compass/VLMEvalKit` | ✅ 200 |
| ~~`QwenLM/Qwen2-VL`~~ | ⚠️ **已 301 跳转到 `QwenLM/Qwen3-VL`** | Qwen2-VL 独立仓库不再维护，需从 git 历史或 HF 取资料 |

### 2.4 框架能力确认（这一步很关键，避免选了框架才发现不支持）

- **EasyR1 支持 Qwen2-VL** —— README 第 31 行原文：`Qwen2-VL/Qwen2.5-VL/Qwen3-VL vision language models`。
  - ⚠️ 但 `examples/` 下**只有 Qwen2.5-VL 和 Qwen3-VL 的示例脚本，没有 Qwen2-VL 的**。意味着 Qwen2-VL 路径的示例需要我自己从 Qwen2.5-VL 脚本改写，且**属于支持但非官方重点验证的路径**，这是一个需要预留调试时间的真实风险（见 [风险 K3](#4-风险登记表)）。
  - 依赖要求：`transformers>=4.54.0`、`vllm>=0.8.3`；官方 Docker：`hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`。
  - 自带基线脚本含 **GeoQA-8k** 与 CLEVR-70k，与你的 RLVR 数据正好对口。
- **VLMEvalKit 三个 benchmark 全覆盖**（已定位到具体实现文件）：
  - MME → `vlmeval/dataset/image_yorn.py`
  - MathVista → `vlmeval/dataset/image_vqa.py`
  - Video-MME → `vlmeval/dataset/videomme.py`（另有 `videommev2.py`，对应升级版 Video-MME-v2）

---

## 3. 关键设计决策

每条决策给出**理由**和**备选方案**，你不同意可以在审批时改。

### D1. SFT + DPO 用 LLaMA-Factory，GRPO 用 EasyR1

- **理由**：LLaMA-Factory 一套配置吃下 SFT 和 DPO 两个阶段，原生支持 `template: qwen2_vl`、DeepSpeed ZeRO-2、以及 `freeze_vision_tower`（正好对应 R6 要求的「冻结视觉编码器」），不必自己写训练循环。GRPO 需要高效 rollout（vLLM 推理 + 训练权重同步），这是 verl/EasyR1 的核心能力，LLaMA-Factory 在 VLM GRPO 上不成熟。
- **代价**：需要装两套环境（LLaMA-Factory 与 EasyR1 的 transformers/vllm 版本可能冲突）。**应对：两个独立 venv，不混装**，这一点会写进部署手册。
- **备选**：全程用 `ms-swift`（一套框架覆盖 SFT/DPO/GRPO）。优点是环境唯一；缺点是 GRPO 实现不如 verl 成熟、社区排错资料少。**如果你更看重环境简单而非 RL 效果，可以改选这个，告诉我即可。**

### D2. 用 ZeRO-2，不用 ZeRO-3

- **理由**：L40 是 PCIe 卡，**无 NVLink**。ZeRO-3 每个 forward/backward 都要 all-gather 完整参数，在 PCIe 带宽下会成为瓶颈；而 Qwen2-VL-2B 在 ZeRO-2 下显存完全够用（估算见 [§8.1](#81-sft-阶段)），没有必要付 ZeRO-3 的通信代价。这也与项目描述要求的 ZeRO-2 一致。

### D3. 冻结 ViT，但**解冻 merger（projector）**

- **理由**：R6 只要求冻结视觉编码器。Qwen2-VL 的视觉侧由 ViT + merger（把 2×2 patch 合并投影到 LLM 维度）组成。冻结 ViT 省显存和算力、防止小数据破坏视觉表征；但保留 merger 可训练，让视觉特征能适配新的指令分布。
- **备选**：merger 也冻结（更保守，更省显存）。我会把两者都写成配置开关，默认解冻 merger，并在文档里说明如何做这个消融。

### D4. 双机分工：Mac 开发 / L40 训练

| 机器 | 职责 |
|---|---|
| MacBook Air | 写代码、下载与转换数据（小样本）、跑 CPU/MPS 烟测、写文档、生成评测报告 |
| 4×L40 | 全量数据、SFT / DPO / GRPO 训练、全量评测 |

所有脚本通过**环境变量注入路径**（`PROJECT_ROOT` / `DATA_ROOT` / `MODEL_ROOT`），同一份代码两边都能跑，不需要改代码。

### D5. 三个 benchmark 统一用 VLMEvalKit

- **理由**：如果 MME 用官方脚本、MathVista 用另一套、Video-MME 用第三套，四个模型 × 三个 benchmark 的分数会因为 prompt 模板、答案抽取方式、抽帧策略不同而**不可横向比较**，整个「验证后训练有效性」的结论就站不住。统一工具是保证结论可信的前提。

### D6. 先跳过 Valley（按你的选择），但这有一个必须知道的后果

⚠️ **Valley 是视频指令数据集**（Valley = Video Assistant with Large Language model Enhanced abilitY）。它是你 SFT 配方里**唯一的视频数据来源**。

跳过它意味着：**SFT 训练数据将全部是静态图像，但你要在 Video-MME 上评测视频理解能力。** 预期结果是 Video-MME 分数**持平或下降**（因为 SFT 会让输出分布偏向单图问答），而不是上升。这会直接影响 R9「验证后训练流程有效性」的结论。

**三个应对选项**（请在审批时选一个）：

1. **接受并如实报告**：把「SFT 无视频数据 → Video-MME 不涨甚至掉分」作为一个**诚实的实验发现**写进报告。这在学术上是完全站得住的，甚至比硬凑出涨分更有说服力。
2. **补 token 拿回 Valley**：你在 HF 网页点同意条款后给我 token。
3. **换等价的开放视频数据**（我已核实以下 ID 全部可直连、非 gated）：
   - `lmms-eval/VideoChatGPT`（原 `lmms-lab/VideoChatGPT`，307 跳转）
   - `MBZUAI/VideoInstruct-100K`（✅ 200，VideoChatGPT 的官方训练集）
   - `ShareGPT4Video/ShareGPT4Video`（✅ 200，下载量 11,584）
   - （`OpenGVLab/VideoChat2-IT` 也可用，但同样是 `gated=auto`）

> **我的推荐：选 3，用 `MBZUAI/VideoInstruct-100K` 替代 Valley。** 它开放、体量合适、与 Valley 同属视频指令数据，能保住 Video-MME 这条评测轴的意义。

---

## 4. 风险登记表

| ID | 风险 | 影响 | 应对 |
|---|---|---|---|
| **K1** | 本机无 GPU，训练脚本**我无法验证** | 上服务器可能报错 | 本机跑 CPU/MPS 烟测覆盖「数据→训练→存档→评测」全链路逻辑；依赖版本全部钉死；准备报错对照表 |
| **K2** | LLaMA-Factory 与 EasyR1 依赖冲突（transformers/vllm 版本） | 环境互相破坏 | **两个独立 venv**，部署手册强制分装 |
| **K3** | EasyR1 无 Qwen2-VL 官方示例 | GRPO 阶段可能踩坑 | 从 `qwen2_5_vl_3b_geo3k_grpo.sh` 改写；预留调试时间；失败时的降级方案是先用 Qwen2.5-VL-3B 验证管线再换回 2B |
| **K4** | 跳过 Valley → 无视频 SFT 数据 | Video-MME 结论失效 | 见 [D6](#d6-先跳过-valley按你的选择但这有一个必须知道的后果)，需你决策 |
| **K5** | **训练集与评测集重叠（数据污染）** | 分数虚高，结论无效 | 在质量过滤阶段做图像级去重检查（MME/MathVista 的图像 vs SFT 图像），并在报告中如实披露 |
| **K6** | MathVista 官方协议需 GPT 做答案抽取判分 | 无 API key 则分数与论文不可比 | 默认配规则抽取退化方案，**并在报告中明确标注「非官方判分」**；有 key 时切换 |
| **K7** | Video-MME 视频体量数百 GB | 服务器磁盘可能不够 | 上机后先 `df -h`；不够则只评 short 子集并如实标注 |
| **K8** | 2B 模型上强制 `<think></think>` CoT 格式奖励**会掉分** | GRPO 白跑 | 这是已知现象（R1-V 在 Qwen2-VL-2B 上的自述）。奖励函数默认**不强制 CoT 格式**，把「加/不加格式奖励」设计成一次消融 |
| **K9** | 数据许可不干净 | 成果不能商用 | ShareGPT4V 为 **CC BY-NC 4.0（禁商用）**；在数据清单中逐项标注许可与风险等级 |
| **K10** | 磁盘：本机 306 GB | 下载中途爆盘 | 本机严格限制在小样本（目标 < 20 GB）；每个下载脚本前置空间检查 |

---

## 5. 目标目录结构

```
Project-1/
├── README.md                      # 项目总览 + 快速开始
├── PLAN.md                        # 本文档（Phase 0 后归档到 docs/）
├── .gitignore                     # 忽略 权重/原始数据/venv/.tools
│
├── .tools/                        # ★ uv + 独立 CPython 3.11（装在工作区内，不污染系统）
│
├── env/
│   ├── local_mac.sh               # 本机开发环境（CPU/MPS，无 DeepSpeed）
│   ├── server_l40_sft.sh          # 服务器 venv-1：LLaMA-Factory + DeepSpeed + flash-attn 2.x
│   ├── server_l40_rl.sh           # 服务器 venv-2：EasyR1 + vLLM
│   ├── server_l40_eval.sh         # 服务器 venv-3：VLMEvalKit
│   ├── requirements-local.txt     # 版本全部钉死
│   ├── requirements-server-*.txt
│   └── check_env.py               # 环境自检：CUDA/显存/NCCL/版本一致性
│
├── data/
│   ├── raw/                       # 原始下载（.gitignore）
│   ├── processed/                 # 统一格式后的 jsonl（.gitignore）
│   ├── samples/                   # ★ 小样本，进 git，供烟测与 code review
│   └── manifests/
│       ├── datasets.yaml          # 数据清单：ID / 配比 / 许可 / 风险等级
│       └── stats_report.md        # 数据统计与过滤报告（自动生成）
│
├── scripts/
│   ├── download/
│   │   ├── 00_model.sh            # Qwen2-VL-2B-Instruct
│   │   ├── 01_sft.sh   02_dpo.sh   03_rlvr.sh   04_bench.sh
│   │   └── _common.sh             # 空间检查 / 断点续传 / HF_ENDPOINT 切换
│   ├── prepare/
│   │   ├── to_sharegpt.py         # 各来源 → 统一 ShareGPT 格式
│   │   ├── build_dpo_pairs.py     # RLHF-V + LLaVA-RLHF → 1 万偏好对
│   │   ├── build_rlvr.py          # Geometry3K/GEOQA-8K/MMK12 → 可验证奖励格式
│   │   ├── quality_filter.py      # 去重/空答案/缺图/超长/去污染
│   │   └── stats.py               # 生成统计报告
│   ├── train/
│   │   ├── sft.sh   dpo.sh   grpo.sh
│   │   └── launch_4xl40.sh        # 多卡启动 + 断点续训
│   ├── eval/
│   │   ├── run_mme.sh   run_mathvista.sh   run_videomme.sh
│   │   ├── run_all.sh             # 4 个模型 × 3 个 benchmark
│   │   └── collect_results.py     # → Markdown + CSV 对比表
│   └── smoke/
│       ├── local_smoke_data.sh    # 数据转换烟测
│       ├── local_smoke_sft.sh     # CPU/MPS LoRA 几十 step
│       └── local_smoke_eval.sh    # 评测脚本跑 20 题
│
├── configs/
│   ├── sft_qwen2vl_2b.yaml        # LLaMA-Factory
│   ├── dpo_qwen2vl_2b.yaml        # LLaMA-Factory
│   ├── grpo_qwen2vl_2b.yaml       # EasyR1
│   ├── reward_geo.py              # RLVR 奖励函数（答案匹配 + 可选格式奖励）
│   └── ds_zero2.json              # DeepSpeed ZeRO-2
│
├── third_party/                   # clone 的框架（.gitignore，用脚本拉取并钉 commit）
│   ├── LLaMA-Factory/  EasyR1/  VLMEvalKit/
│
├── docs/                          # ★★ 你要的「指南」本体
│   ├── 00-执行路线图.md
│   ├── 01-学习路径.md
│   ├── 02-踩坑清单.md
│   ├── 03-服务器部署手册.md
│   └── 04-实验记录模板.md
│
└── results/
    ├── checkpoints/               # (.gitignore)
    ├── eval/                      # 原始评测输出
    └── report.md                  # 最终对比表
```

---

## 6. 分阶段执行步骤

### Phase 0 — 清空与骨架（~10 分钟）

| 动作 | 说明 |
|---|---|
| 0.1 | 生成 `legacy-manifest.md`：记录被清理内容的**完整清单 + 原始下载命令**（含 `Qwen3-VL-4B-Instruct` 的 `huggingface-cli` 命令），万一要找回有据可依 |
| 0.2 | **将现有全部内容移入 macOS 废纸篓**（`mv ~/.Trash/`，瞬时完成、可恢复）。除非你在审批时要求改为 `rm -rf` 不可恢复硬删除 |
| 0.3 | `git init`，写 `.gitignore`（忽略 `models/ data/raw/ data/processed/ third_party/ .tools/ results/checkpoints/`） |
| 0.4 | 按 [§5](#5-目标目录结构) 创建目录骨架与占位 README |

- **产出**：干净的骨架仓库 + 首次 commit
- **验收**：`git log` 有初始提交；`tree` 结构与 §5 一致
- **不可逆性**：0.2 是本计划中唯一不可逆的步骤（走废纸篓则可逆）

---

### Phase 1 — 环境准备（~40 分钟）

| 动作 | 说明 |
|---|---|
| 1.1 | 把 `uv` 安装到 `.tools/bin`（`UV_INSTALL_DIR` 指向工作区），CPython 3.11 安装到 `.tools/python`（`UV_PYTHON_INSTALL_DIR`）。**全程不写系统目录、不写 `~/.local`**，符合沙箱限制 |
| 1.2 | 建本机 venv，装 torch（MPS 版）+ transformers>=4.54 + qwen-vl-utils + accelerate + datasets + pillow。**本机不装 DeepSpeed / flash-attn / vLLM**（Mac 上装不了且用不到） |
| 1.3 | 下载 `Qwen/Qwen2-VL-2B-Instruct`（约 4.4 GB）到 `models/` |
| 1.4 | 跑一次真实图文推理，确认基座可用（这是**第一个真实验收点**） |
| 1.5 | 写服务器三套环境脚本 `env/server_l40_{sft,rl,eval}.sh` + 版本钉死的 requirements + `check_env.py` 自检脚本 |

- **产出**：可用的本机 Python 环境、基座模型、服务器环境脚本
- **验收**：本机能对一张图输出合理描述
- **风险**：K2（依赖冲突）—— 通过三个独立 venv 规避

---

### Phase 2 — 数据（后台任务，~2–3 小时，不阻塞后续阶段）

| 动作 | 说明 |
|---|---|
| 2.1 | 写 `data/manifests/datasets.yaml`：所有数据源的 ID / 配比 / 许可 / 风险等级（路径用 [§2.3](#23-仓库路径核查️-有-6-处已改名用旧-id-会跳转或失败) 已核实的正确 ID） |
| 2.2 | 下载（详见 [§7](#7-数据方案详表)）：本机只取小样本 + 全量小体积数据集；全量 SFT 图像留给服务器脚本 |
| 2.3 | 格式统一：全部转成 ShareGPT 格式 jsonl（`to_sharegpt.py`） |
| 2.4 | 构建 1 万条 DPO 偏好对（`build_dpo_pairs.py`） |
| 2.5 | 构建 RLVR 数据 + 奖励函数（`build_rlvr.py`），答案字段可被程序验证 |
| 2.6 | 质量过滤 + **训练/评测去污染检查**（K5），生成 `stats_report.md` |

- **产出**：`data/processed/` 下的统一格式数据 + `data/samples/` 小样本 + 统计报告
- **验收**：统计报告给出每个来源的条数、平均长度、被过滤条数及原因、与评测集的图像重叠数
- **并行**：下载走**后台任务**，同时我推进 Phase 3/4

---

### Phase 3 — 训练框架（~2 小时）

| 动作 | 说明 |
|---|---|
| 3.1 | clone LLaMA-Factory 与 EasyR1 到 `third_party/`，**钉住 commit hash**（避免上游更新导致配置失效） |
| 3.2 | 写 `configs/sft_qwen2vl_2b.yaml`（冻结 ViT、2 epoch、ZeRO-2），超参见 [§8.1](#81-sft-阶段) |
| 3.3 | 写 `configs/dpo_qwen2vl_2b.yaml`（基于 SFT ckpt、1 epoch），见 [§8.2](#82-dpo-阶段) |
| 3.4 | 写 `configs/grpo_qwen2vl_2b.yaml` + `reward_geo.py`（200 step），见 [§8.3](#83-grpo--rlvr-阶段)。**从 EasyR1 的 `qwen2_5_vl_3b_geo3k_grpo.sh` 改写**（K3） |
| 3.5 | 写 `ds_zero2.json` 与 `launch_4xl40.sh`（含断点续训） |
| 3.6 | **给出每个阶段的显存估算与推导过程**，写进部署手册，让你上机前就知道该不该调 batch size |

- **产出**：三份可直接运行的训练配置 + 启动脚本
- **验收**：配置能通过框架的参数解析（本机 `--help` / dry-run 级别校验）
- **诚实声明**：这一步**我无法验证真实多卡训练**（K1）

---

### Phase 4 — 评测框架（~1 小时）

| 动作 | 说明 |
|---|---|
| 4.1 | clone VLMEvalKit，钉 commit，配置 Qwen2-VL-2B 模型入口 |
| 4.2 | 接入三个 benchmark：MME / MathVista / Video-MME，统一 prompt 与判分协议（D5） |
| 4.3 | 写 `run_all.sh`：**4 个模型（pretrained / SFT / DPO / GRPO）× 3 个 benchmark = 12 次评测**，支持断点续跑 |
| 4.4 | 写 `collect_results.py`：输出 Markdown + CSV 对比表，**自动标注判分方式**（官方 GPT 判分 or 规则退化，对应 K6） |

- **产出**：一键评测脚本 + 报告生成器
- **验收**：本机用 pretrained 模型跑 MME 的 20 题子集，产出格式正确的报告

---

### Phase 5 — 本机烟测（~1 小时）

**这个阶段的唯一目的：保证上服务器时不是代码第一次运行。**

| 动作 | 验收标准 |
|---|---|
| 5.1 数据烟测 | 小样本走完 `to_sharegpt → filter → stats`，输出合法 jsonl |
| 5.2 训练烟测 | CPU/MPS 上用 LoRA + 几十条样本跑几十 step，**loss 下降**，能存出 checkpoint |
| 5.3 评测烟测 | 用上一步的 checkpoint 跑 20 题，产出报告 |
| 5.4 全链路 | 上述三步串起来一次跑通 |

> ⚠️ 烟测用 LoRA 和极小样本，**只验证代码路径连通，不代表训练效果**。服务器上是全参微调，配置不同。

---

### Phase 6 — 指南文档（~2 小时）

产出 `docs/` 下 5 篇文档，大纲见 [§10](#10-指南文档交付大纲)。这是你最核心的诉求之一。

---

### Phase 7 — 服务器部署（需要你提供 SSH 信息）

| 动作 | 说明 |
|---|---|
| 7.1 | 上机环境自检：`nvidia-smi`（确认 4×L40 与驱动）、`df -h`（磁盘，对应 K7）、CUDA 版本 |
| 7.2 | 按 `03-服务器部署手册.md` 装三套 venv |
| 7.3 | 同步代码；服务器端跑全量数据下载（约 100 GB+） |
| 7.4 | **先跑 SFT 的前 100 step 测吞吐**，据此校准总时长与超参，再放全量 |
| 7.5 | 依次跑 SFT → DPO → GRPO → 全量评测 |

> 这一阶段**你可以选择让我通过 SSH 代为执行，或你自己照手册执行**。如果是我执行，我需要你提供连接方式。

---

## 7. 数据方案详表

### 7.1 SFT（目标 ≈ 30 万条）

| 来源 | 配比 | 本机 | 服务器 | 许可与风险 |
|---|---|---|---|---|
| `liuhaotian/LLaVA-Instruct-150K`（665K 混合） | 采样 **200K** | 标注 json + 2K 样本图 | 全量图像 ~100 GB | mixed，受 OpenAI 条款约束 |
| `Lin-Chen/ShareGPT4V` | **65K** | 标注 + 样本 | 全量 | ⚠️ **CC BY-NC 4.0，禁商用** |
| `luoruipu1/Valley-Instruct-65k`（视频） | **35K** | — | — | ⚠️ **gated，已按你选择跳过** |
| **替代建议**：`MBZUAI/VideoInstruct-100K` | 35K | 标注 + 样本 | 全量 | 开放，非 gated（见 [D6](#d6-先跳过-valley按你的选择但这有一个必须知道的后果)） |

> 若最终确定跳过视频数据，则配比调整为 LLaVA-665K **235K** + ShareGPT4V **65K** = 30 万，并在报告中声明 Video-MME 无对应训练数据。

### 7.2 DPO（目标 ≈ 1 万条偏好对）

| 来源 | 数量 | 说明 |
|---|---|---|
| `openbmb/RLHF-V-Dataset` | ~5.7K | 细粒度人工修正的幻觉偏好对 |
| `zhiqings/LLaVA-RLHF-Data` | ~4.3K | 补足到 1 万 |

本机**全量下载**（体量小，约 5 GB）。

### 7.3 RLVR（GRPO 200 step）

| 来源 | 数量 | 角色 |
|---|---|---|
| `hiyouga/geometry3k` | 2,101 train | **冒烟**：EasyR1 官方示例数据，最快跑通管线 |
| `leonardPKU/GEOQA_R1V_Train_8K` | ~8K | 主训练集（EasyR1 有官方基线脚本） |
| `FanqingM/MMK12` | 采样 | 补充多学科可验证题目 |

奖励设计：**答案精确匹配为主**；格式奖励设为开关，默认关闭（K8）。

### 7.4 评测数据

| Benchmark | 本机 | 服务器 |
|---|---|---|
| MME (`lmms-lab-encoder/MME`) | ✅ 全量（~2 GB） | 同步 |
| MathVista (`AI4Math/MathVista`) | ✅ testmini 1000 题 | 同步 |
| Video-MME (`lmms-eval/Video-MME`) | 仅标注，**视频不下** | 全量（数百 GB，先查磁盘，K7） |

### 7.5 质量过滤规则

1. 精确去重 + 近似去重（文本 hash）
2. 丢弃空答案 / answer 长度 < 2
3. 丢弃图片缺失或无法解码的样本
4. 超长对话按 `cutoff_len` 截断或丢弃
5. **去污染**：SFT/RLVR 图像 vs MME/MathVista 评测图像做重叠检查，命中则剔除并**在报告中披露数量**

---

## 8. 训练超参方案

> 以下为**初始推荐值**，需在服务器上用前 100 step 实测吞吐与显存后校准（Phase 7.4）。

### 8.1 SFT 阶段

```yaml
model_name_or_path: models/Qwen2-VL-2B-Instruct
stage: sft
finetuning_type: full
freeze_vision_tower: true              # R6 要求：冻结视觉编码器
freeze_multi_modal_projector: false    # D3：merger 保持可训练
deepspeed: configs/ds_zero2.json       # R5
template: qwen2_vl
cutoff_len: 2048
image_max_pixels: 262144               # 512×512 ≈ 334 visual tokens
per_device_train_batch_size: 4
gradient_accumulation_steps: 8         # global batch = 4×8×4卡 = 128
learning_rate: 1.0e-5
num_train_epochs: 2.0                  # R6 要求
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
```

**显存估算（单卡，ZeRO-2，冻结 ViT 后可训练参数 ≈ 1.5B）**

| 项 | 计算 | 占用 |
|---|---|---|
| 参数 bf16（全模型，不分片） | 2.21B × 2 B | ~4.4 GB |
| 梯度 bf16（仅可训练，/4 分片） | 1.5B × 2 / 4 | ~0.8 GB |
| 优化器 AdamW fp32（/4 分片） | 1.5B × 12 / 4 | ~4.5 GB |
| **静态小计** | | **≈ 9.7 GB** |
| 激活值（开 checkpointing） | 视 batch 与图像 token 而定 | 余量约 38 GB |

> 结论：L40 48 GB **显存非常充裕**，`per_device_train_batch_size` 可能还能往上调。这正是要在前 100 step 实测的原因。

**耗时**：600K 样本次（30 万 × 2 epoch）。**这是需要实测的量，我不给拍脑袋的数字**；Phase 7.4 会用前 100 step 外推出准确值。

### 8.2 DPO 阶段

```yaml
stage: dpo
model_name_or_path: results/checkpoints/sft   # R7：基于 SFT checkpoint
pref_beta: 0.1
pref_loss: sigmoid
learning_rate: 5.0e-7                  # 远低于 SFT，DPO 对 lr 极敏感
num_train_epochs: 1.0                  # R7 要求
per_device_train_batch_size: 1
gradient_accumulation_steps: 16        # global batch = 64
freeze_vision_tower: true
deepspeed: configs/ds_zero2.json
```

> ⚠️ DPO 需要同时载入 **policy + reference 两个模型**，显存约为 SFT 的 2 倍（静态 ≈ 15 GB），48 GB 仍然够用。

### 8.3 GRPO / RLVR 阶段

```yaml
# EasyR1，从 examples/qwen2_5_vl_3b_geo3k_grpo.sh 改写（K3）
worker.actor.model.model_path: results/checkpoints/dpo
worker.rollout.n: 5                    # 每个 prompt 采样 5 条
worker.actor.global_batch_size: 128
worker.actor.optim.lr: 1.0e-6
algorithm.kl_coef: 1.0e-2
trainer.max_steps: 200                 # R8 要求
```

奖励函数 `reward_geo.py`：答案精确匹配（主）+ 格式奖励（默认关闭，K8）。

---

## 9. 评测方案

### 9.1 评测矩阵

|  | MME | MathVista | Video-MME |
|---|---|---|---|
| pretrained（基线） | ✅ | ✅ | ✅ |
| SFT | ✅ | ✅ | ✅ |
| DPO | ✅ | ✅ | ✅ |
| GRPO | ✅ | ✅ | ✅ |

共 12 次评测。项目描述只要求 pretrained/SFT/RL 三个模型，**我多加了 DPO 一列**，这样才能区分「DPO 的贡献」与「GRPO 的贡献」——否则 RL 涨了分你不知道是哪一步带来的。

### 9.2 各 benchmark 要点

| Benchmark | 指标 | 注意事项 |
|---|---|---|
| **MME** | Perception（满分 2000）+ Cognition（满分 800） | 是/否二元题，分数 = accuracy + accuracy+；对 prompt 模板敏感，必须四个模型完全一致 |
| **MathVista** | testmini 1000 题准确率 | ⚠️ 官方用 GPT 抽取答案判分（K6）。无 API key 则用规则抽取，**报告中必须标注「非官方判分，不可与论文数字直接比较」** |
| **Video-MME** | short/medium/long × 有/无字幕 | ⚠️ 抽帧数量显著影响分数，四个模型必须用同一抽帧策略；视频体量大（K7）；⚠️ 若跳过视频 SFT 数据，预期不涨（K4） |

### 9.3 报告形态

`results/report.md` 自动生成，包含：模型 × benchmark 对比表、相对基线的增减、**判分方式标注**、**数据污染检查结果**、以及未达预期项的诚实说明。

---

## 10. 指南文档交付大纲

> 这是你诉求里「**我该怎么做、该学什么、该注意什么**」的正式交付物。

### `docs/00-执行路线图.md` — 怎么做

- 全流程时间线与依赖图
- 每个阶段的 **go / no-go 判据**（例如：SFT loss 应该落在什么区间、MME 相对基线涨多少才算正常、GRPO 的 reward 曲线该是什么形状；不满足时该回头查什么）
- 每阶段的产出物 checklist
- 失败时的回退路径

### `docs/01-学习路径.md` — 学什么

按**依赖顺序**排列，每项标注「必读 / 可跳过」，并指向**具体的论文章节或源码文件**而不是泛泛的链接：

1. **VLM 基础架构**：ViT + connector + LLM 的三段式，LLaVA 系列的演进
2. **Qwen2-VL 的两个关键设计**：
   - Naive Dynamic Resolution（动态分辨率 → 为什么图像 token 数是变的 → 为什么这会炸显存）
   - M-RoPE（多模态旋转位置编码 → 为什么它能处理视频时序）
3. **SFT**：指令微调的数据格式、loss mask（为什么只在 answer 上算 loss）、为什么冻结 ViT
4. **DPO**：从 RLHF 到 DPO 的推导、`beta` 的物理含义、为什么 DPO 对学习率极其敏感
5. **GRPO**：与 PPO 的差别（去掉 value model）、group 内优势估计、KL 约束的作用
6. **RLVR**：可验证奖励 vs 奖励模型，为什么小模型更吃 RLVR
7. **DeepSpeed ZeRO**：Stage 1/2/3 分别分片什么、为什么无 NVLink 时 ZeRO-3 会慢
8. **评测方法论**：benchmark 的判分协议、为什么分数不可跨工具比较、数据污染

### `docs/02-踩坑清单.md` — 注意什么

按「症状 → 原因 → 解法」组织，覆盖 [§4 风险登记表](#4-风险登记表)的全部条目，另加：

- 2B 模型上**强制 CoT 格式奖励会掉分**（K8）
- 视觉 token 数量失控导致 OOM（`image_max_pixels` 的作用）
- DPO 学习率设成 SFT 量级 → 模型崩溃
- 训练集/评测集重叠（K5）
- 数据集许可陷阱：ShareGPT4V 禁商用（K9）
- **HF 仓库改名**（本文 §2.3 的 6 处）
- 多卡训练卡住 / NCCL 超时的排查顺序
- checkpoint 保存与恢复的坑

### `docs/03-服务器部署手册.md`

逐条命令、三套 venv 分装、数据同步、多卡启动、断点续训、**常见报错对照表**。

### `docs/04-实验记录模板.md`

消融实验记录表，避免跑完不知道哪个改动有效。

---

## 11. 需要你提供的东西

| # | 事项 | 何时需要 | 不提供的后果 |
|---|---|---|---|
| 1 | **Valley/视频数据的决策**（[D6](#d6-先跳过-valley按你的选择但这有一个必须知道的后果) 的三选一） | Phase 2 前 | 默认按选项 1（如实报告 Video-MME 不涨） |
| 2 | **SSH 连接信息**（host/user/密钥） | Phase 7 | 你自己照手册执行，我无法代跑 |
| 3 | **OpenAI API key**（MathVista 官方判分，K6） | Phase 4 | 用规则判分，分数不可与论文比较（会标注） |
| 4 | 服务器磁盘余量（K7） | Phase 7.1 | 上机第一步就查 |

---

## 12. 时间线与里程碑

| Phase | 内容 | 预计耗时 | 阻塞关系 |
|---|---|---|---|
| 0 | 清空与骨架 | 10 min | — |
| 1 | 环境准备 | 40 min | 依赖 0 |
| 2 | 数据下载与处理 | 2–3 h（**后台**） | 依赖 1，与 3/4 并行 |
| 3 | 训练框架 | 2 h | 依赖 1 |
| 4 | 评测框架 | 1 h | 依赖 1 |
| 5 | 本机烟测 | 1 h | 依赖 2/3/4 |
| 6 | 指南文档 | 2 h | 可与 3/4 并行 |
| **本机合计** | | **约 5–7 h**（下载在后台并行） | |
| 7 | 服务器部署与训练 | 取决于你的服务器 | 需你提供 SSH |

**里程碑**
- **M1**（Phase 1 末）：本机能用 Qwen2-VL-2B 做真实图文推理
- **M2**（Phase 2 末）：拿到统一格式数据 + 数据统计与污染检查报告
- **M3**（Phase 5 末）：本机全链路烟测通过 → **可以上服务器了**
- **M4**（Phase 6 末）：指南文档交付
- **M5**（Phase 7 末）：四个模型 × 三个 benchmark 的完整对比表

---

## 13. 审批清单

请逐条确认（不同意的直接说，我改完再执行）：

| # | 待确认事项 | 我的默认方案 |
|---|---|---|
| A1 | **清空方式** | 移入废纸篓（可恢复）。要不可恢复硬删除请明说 |
| A2 | **视频数据决策**（D6） | 推荐用 `MBZUAI/VideoInstruct-100K` 替代 Valley，保住 Video-MME 这条评测轴 |
| A3 | **框架选型**（D1） | LLaMA-Factory（SFT+DPO）+ EasyR1（GRPO）+ VLMEvalKit（评测）。备选：全程 ms-swift |
| A4 | **评测多加 DPO 一列**（§9.1） | 加。否则分不清 DPO 与 GRPO 各自的贡献 |
| A5 | **merger 是否解冻**（D3） | 默认解冻，并作为一次消融 |
| A6 | **Phase 7 谁来执行** | 你决定：我通过 SSH 代跑，或你照手册自己跑 |

---

## 附录 A：核实方法

本文档中所有环境数据与仓库路径均通过以下方式**实测**获得，未依赖记忆：

- 本机环境：`uname -a`、`df -h`、`python3 -V`、`which` 系列命令
- 网络：`curl -o /dev/null -w "%{http_code} %{time_total}"` 对各站点实测
- HF 仓库：`https://huggingface.co/api/{models,datasets}/<id>`，用 `curl -sL` 跟随重定向确认**最终 ID**、`gated` 状态与下载量
- GitHub 仓库：`https://api.github.com/repos/<id>`，确认 `full_name`（改名）、star 数、默认分支、最近推送时间
- 框架能力：直接抓取 `README.md` 与 `vlmeval/dataset/` 目录清单，定位到具体实现文件

> 凡本文未能核实的数字（如训练耗时），已明确标注为「需实测」，不提供估计值冒充事实。

# Postrain 评测报告

> 自动生成：`scripts/eval/collect_results.py`


## 1. 模型 × Benchmark 对比

| 模型 | MME | MathVista | Video-MME |
|---|---|---|---|
| **base(未指令微调)** | — | — | — |
| **base+sft** | 1418.310 (-74.403) | 43.600 (-5.600) | 58.200 (-1.800) |
| **instruct(官方)** | 1492.712 (+0.000) | 49.200 (+0.000) | 60.000 (+0.000) |
| **sft** | 1536.296 (+43.584) | 47.100 (-2.100) | 57.300 (-2.700) |
| **sft_ep3** | 1505.771 (+13.059) | 47.500 (-1.700) | 57.600 (-2.400) |
| **dpo** | 1547.457 (+54.745) | 47.300 (-1.900) | 57.700 (-2.300) |
| **grpo** | — | — | — |

> ⚠️ **各列刻度不同，别横向比绝对值**（2026-09-29 澄清）：
> - `MME` 是 **2000 分制**（perception 10 类 × 200；reasoning 4 类 × 200 = 800），**不是百分比**。例：perception 1492.71 → 1536.30 是 +43.6 **分**，换算仅 **+2.2 个百分点**。
> - `MathVista` / `Video-MME` 是**百分比**（Video-MME 取 3 分钟以内的 short 档）。
> - 括号内为相对基线的增减，**同刻度**之差。
> - 换判分模型不影响可比性（09-23 用 gpt-4o-mini、09-29 用 gpt-5.5，Video-MME short overall 同为 0.600 实测验证）。
> 
> **关于两条基线（2026-09-29 澄清）**：
> - `instruct(官方)` = `Qwen2-VL-2B-Instruct`，**官方已指令微调**的版本，也是本项目 SFT/DPO 的起点。**报告的主基线是它** —— 所以增益是在「已指令微调」之上再取得的。
> - `base(未指令微调)` = `Qwen2-VL-2B`（无 -Instruct），纯预训练。
>   ⚠️ **它只能当参照，不能当同口径基线**：预训练模型没有 chat template、不按对话格式回答，用同一套对话式 prompt 评测天然吃亏。它的分数低是**预期**的，不代表「训练没用」。


实测明细（聚合分口径）：
- **base_sft · MME**：perception 1418.31/2000; reasoning 377.86/800
- **base_sft · Video-MME**：Video-MME 分档正确率: short 58.2%; medium nan%; long nan%
- **pretrained · MME**：perception 1492.71/2000; reasoning 456.43/800
- **pretrained · Video-MME**：Video-MME 分档正确率: short 60.0%; medium nan%; long nan%
- **sft · MME**：perception 1536.30/2000; reasoning 457.14/800
- **sft · Video-MME**：Video-MME 分档正确率: short 57.3%; medium nan%; long nan%
- **sft_ep3 · MME**：perception 1505.77/2000; reasoning 436.79/800
- **sft_ep3 · Video-MME**：Video-MME 分档正确率: short 57.6%; medium nan%; long nan%
- **dpo · MME**：perception 1547.46/2000; reasoning 450.36/800
- **dpo · Video-MME**：Video-MME 分档正确率: short 57.7%; medium nan%; long nan%


## 2. 判分方式（必须逐项标注）

| Benchmark | 判分方式 |
|---|---|
| MME | 官方协议（是/否二元题，accuracy + accuracy+） |
| MathVista | ⚠️ **规则抽取判分（非官方）** —— 无 OpenAI API key，**分数不可与论文数字直接比较**（PLAN K6） |
| Video-MME | VLMEvalKit 默认抽帧（本项目只评 `short` 档，即 3 分钟以内）；⚠️ 四个模型必须用**同一抽帧策略**。**实测判分模型可换**：09-23 用 gpt-4o-mini、09-29 用 gpt-5.5，`short overall` 同为 0.600 ⇒ 换判分模型不改变可比性 |

## 3. 数据污染检查

✅ 已执行去污染：收集评测集图像指纹 6,742 张，从训练集剔除重叠样本 **140** 条。

## 4. 未达预期项的说明

> 实测数据见上表。以下逐条如实说明**没涨的部分**（2026-09-29 填）。

- **MathVista 全线小幅退步**（相对 instruct：sft −2.1 / sft_ep3 −1.7 / dpo −1.9）：
  40 万条混训数据以感知与描述类为主（llava665k 296k、sharegpt4v 50k，多为 caption/VQA 风格），
  数学与推理样本占比很低 ⇒ 预期本就有限，实测是「没涨、略降」。属**数据问题**，非训练失败。

- **Video-MME 同样小幅退步**（sft −2.7 / sft_ep3 −2.4 / dpo −2.3）：
  SFT 数据中视频样本仅 **13.4%**（受 YouTube 可用性限制），且只取 3 分钟以内片段，
  长时序理解无从获得 ⇒ 同样**数据决定**。

- **多训一轮 epoch（sft_ep3）是净亏**：MME perception 从 +43.6 掉到 +13.1，
  reasoning 从 457.14 掉到 **436.79（−19.6）**，而 MathVista / Video-MME 只各涨 0.3~0.4。
  ⇒ 2 epoch 已到拐点，第 3 轮开始退化（过拟合或调度重启所致）。**不建议再训第 3 轮。**

- **GRPO 一列全空 = 未完成**（环境问题已修复，但按用户决定暂缓），
  不是「跑了但没分」，不得与其他两项混为一谈。


## 5. 复现信息

- 训练数据: `data/processed/sft_400k.clean.jsonl`
- 评测工具: VLMEvalKit（四个模型统一）
- 原始输出: `results/eval/<bench>/<model>/T<时间戳>/`

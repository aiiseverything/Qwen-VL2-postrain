# Postrain 评测报告

> 自动生成：`scripts/eval/collect_results.py`


## 1. 模型 × Benchmark 对比

| 模型 | MME | MathVista | Video-MME |
|---|---|---|---|
| **pretrained** | — | — | — |
| **sft** | — | — | — |
| **dpo** | — | — | — |
| **grpo** | — | — | — |

> 括号内为相对 `pretrained` 基线的增减。


## 2. 判分方式（必须逐项标注）

| Benchmark | 判分方式 |
|---|---|
| MME | 官方协议（是/否二元题，accuracy + accuracy+） |
| MathVista | ⚠️ **规则抽取判分（非官方）** —— 无 OpenAI API key，**分数不可与论文数字直接比较**（PLAN K6） |
| Video-MME | VLMEvalKit 默认抽帧；⚠️ 四个模型必须用**同一抽帧策略** |

## 3. 数据污染检查

✅ 已执行去污染：收集评测集图像指纹 6,742 张，从训练集剔除重叠样本 **140** 条。

## 4. 未达预期项的说明

_（本节省略 = 未如实评估。请逐条填写：哪些指标没涨、为什么、是数据问题还是方法问题）_

- 例：**Video-MME 未提升** —— SFT 训练数据中视频占比低（受 YouTube 可用性限制），预期视频理解提升有限。这是数据决定的，不是训练失败。


## 5. 复现信息

- 训练数据: `data/processed/sft_400k.clean.jsonl`
- 评测工具: VLMEvalKit（四个模型统一）
- 原始输出: `results/eval/<model>/<bench>/`

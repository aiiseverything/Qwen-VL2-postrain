# Postrain 数据统计报告

> 自动生成 —— `scripts/prepare/stats.py`


## 1. SFT 数据集

- 文件: `data/processed/sft_400k.clean.jsonl`
- 总条数: **399,642**
- 平均轮数: 8.62 | 平均字符数: 699 | 平均答案字符数: 338 | 最长: 9,774

| 来源 | 条数 |
|---|---|
| llava665k | 296,548 |
| sharegpt4v | 49,944 |
| videoinstruct | 53,150 |

| 模态 | 条数 |
|---|---|
| image | 346,492 |
| video | 53,150 |

**配额达成率: 399,642 / 400,000 = 99.9%**

## 2. DPO 偏好对

- 文件: `data/processed/dpo_10k.jsonl`
- 总条数: **5,731** (目标 10,000)
- 平均字符数: 63

| 来源 | 条数 |
|---|---|
| rlhf-v | 5,731 |

## 3. RLVR 数据

| 数据集 | 文件 |
|---|---|
| geometry3k | `geometry3k.parquet` (2,101 行) |
| geoqa_r1v | `geoqa_r1v.parquet` (8,031 行) |
| mmk12 | `mmk12.parquet` (15,616 行) |

## 4. 过滤与丢弃原因

### 4.1 质量过滤 / 去污染 (`quality_filter.py`)

| 原因 | 条数 |
|---|---|
| dropped_decontaminated | 133 |
| eval_fingerprints | 6,742 |
| input | 399,775 |

> **去污染披露 (PLAN K5)**: 训练集中与 MME/MathVista 评测图像重合、已被剔除的样本共 **133** 条。

### 4.2 抽取阶段丢弃 (`to_sharegpt.py`)

| 原因 | 数量 |
|---|---|
| llava665k_dropped_bad_conversation | 1 |
| llava665k_dropped_duplicate | 224 |
| llava_alloc | `{"coco": 198452, "gqa": 39319, "textvqa": 11965, "vg": 47102}` |
| llava_dropped_excluded_ocr_vqa | 80,000 |
| llava_dropped_no_image | 40,688 |
| llava_pool_by_prefix | `{"coco": 364100, "gqa": 72140, "textvqa": 21953, "vg": 86417}` |
| llava_total_records | 665,298 |
| sharegpt4v_dropped_unmapped_image | 29,988 |
| sharegpt4v_pool | 80,027 |
| sharegpt4v_total_records | 102,025 |
| video_total_turns | 100,010 |
| video_turns_after_duration_filter | 58,809 |
| video_unique_ids | 13,303 |
| videoinstruct_dropped_duplicate | 530 |
| videoinstruct_dropped_missing_video | 5,129 |

## 5. 许可与风险

| 数据集 | 许可 | 风险 |
|---|---|---|
| liuhaotian/LLaVA-Instruct-150K | CC-BY-4.0 (受 OpenAI 条款约束) | medium |
| Lin-Chen/ShareGPT4V | **CC-BY-NC-4.0** | **HIGH (禁商用)** |
| MBZUAI/VideoInstruct-100K | CC-BY-SA-4.0 | medium |
| openbmb/RLHF-V-Dataset | **CC-BY-NC-4.0** | **HIGH (禁商用)** |
| zhiqings/LLaVA-RLHF-Data | **CC-BY-NC-4.0** | **HIGH (禁商用)** |
| hiyouga/geometry3k | MIT | low |
| leonardPKU/GEOQA_R1V_Train_8K | apache-2.0 | low |
| FanqingM/MMK12 | apache-2.0 | low |

#!/usr/bin/env python3
"""生成 VLMEvalKit 的 config JSON（4 个模型 × 3 个 benchmark）。

=== 为什么需要这个脚本 ====================================================
VLMEvalKit 的 `run.py` **没有 `--model-path` 参数**。要用本地自有权重，
必须写一个 config JSON，在 `model` 段里用 `class` 指定适配器类、
用 `model_path` 指定本地路径，然后 `run.py --config <cfg>`。

同时这里要处理一个**本机特有的补丁**：VLMEvalKit 的 Qwen2VLChat 把
`attn_implementation` 硬编码成 `flash_attention_2`，本机没装 flash-attn，
故显式传 `"attn_implementation": "sdpa"`（已给 VLMEvalKit 打了补丁使其可覆盖）。

数据集名必须是**已注册的键**（实测）:
    MME            -> class ImageYORNDataset
    MathVista_MINI -> class MathVista        (注意不是 'MathVista')
    Video-MME      -> class VideoMME
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "pretrained": "models/Qwen2-VL-2B-Instruct",
    "sft":        "results/checkpoints/sft",
    "dpo":        "results/checkpoints/dpo",
    "grpo":       "results/checkpoints/grpo",
}

# VLMEvalKit 的 data 段格式是 {<别名>: {"class": ..., "dataset": <真实数据集名>, ...}}
# ⚠️ **`dataset` 键是必需的** —— 只写 `class` 会报
#    `ValueError: 'dataset' must be a non-empty string for dataset config ...`
#    （见 docs/05 问题 P32）
#
# Video-MME 的别名里带抽帧数（`Video-MME_16frame`）:
#   PLAN §9.2 要求四个模型**必须用同一抽帧策略**, 否则分数不可横向比较。
#   把 nframe 固定写进别名, 四个模型的配置天然一致。
DATASETS = {
    "MME":               {"class": "ImageYORNDataset", "dataset": "MME"},
    "MathVista_MINI":    {"class": "MathVista",        "dataset": "MathVista_MINI"},
    "Video-MME_16frame": {"class": "VideoMME",         "dataset": "Video-MME", "nframe": 16},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default=str(ROOT / "configs/vlmeval_config.json"))
    ap.add_argument("--attn", default="sdpa", help="sdpa / eager / flash_attention_2")
    ap.add_argument("--allow-missing", action="store_true",
                    help="模型目录不存在时仍写入（默认跳过并在 stderr 说明）")
    args = ap.parse_args()

    models, skipped = {}, []
    for m in args.models:
        p = ROOT / MODEL_PATHS[m]
        if not p.is_dir():
            skipped.append((m, str(p)))
            if not args.allow_missing:
                continue
        models[m] = {
            "class": "Qwen2VLChat",
            "model_path": str(p),
            "attn_implementation": args.attn,   # ← 本机补丁：默认 sdpa
            "max_new_tokens": 2048,
            "verbose": False,
        }

    data = {d: DATASETS[d] for d in args.datasets if d in DATASETS}
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"未注册的数据集名: {unknown}；可用: {list(DATASETS)}")
    if not models:
        raise SystemExit("没有任何可用的模型目录；先跑完 SFT 生成 results/checkpoints/sft")

    cfg = {"model": models, "data": data}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"配置写入 {out}")
    print(f"  模型: {list(models)}")
    print(f"  数据集: {list(data)}")
    if skipped:
        print("  跳过（目录不存在）:")
        for m, p in skipped:
            print(f"    {m}: {p}")


if __name__ == "__main__":
    main()

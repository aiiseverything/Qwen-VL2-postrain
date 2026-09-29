#!/usr/bin/env python
"""把 datasets 的 map 缓存分片转成 `load_from_disk` 格式的 tokenized 数据集。

## 为什么要这个

LLaMA-Factory 的 `tokenized_path` 有两种用法：
  * 已经存在 → `load_from_disk` 直接加载，**完全不看指纹**
  * 不存在   → 正常走 `dataset.map(...)`，并把结果 `save_to_disk` 到该路径

而 datasets 的 map 缓存是**按"整对象指纹"命中**的：指纹 = `dill.dumps(transform)`
+ 输入指纹，其中 transform 是绑定方法 `dataset_processor.preprocess_dataset`，
于是 **整个 processor 实例（含 data_args / tokenizer / processor）都被序列化进去**。
后果：只要换 `model_name_or_path`，processor 变了 → 指纹变了 → **4 小时的 tokenize 缓存作废**
（实测：主 run 的 4h08m 缓存，换模型路径后完全用不上）。

本脚本把**已经存在的 map 缓存分片**拼成 `DatasetDict` 并 `save_to_disk` 到
`tokenized_path`，从而让后续任意跑批（换模型、换 epoch 数…）都能直接 `load_from_disk`，
**几分钟进训练**。等价性已单独验证：vocab / merges 完全一致，唯一差异是
`<|image_pad|>` `<|video_pad|>` 两个 added token —— 而它们在任何一次跑批里都会被自动加上。

## 用法
    .venv-lf/bin/python scripts/training/build_tokenized_cache.py <分片glob> <输出目录>
例：
    .venv-lf/bin/python scripts/training/build_tokenized_cache.py \
      '/home/ml-user/workdir/cache/hf/datasets/json/default-3d379ea0a8bd3d7f/*/*/cache-b29e*.arrow' \
      data/cache/sft_tok
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    pat, out = sys.argv[1], Path(sys.argv[2])

    shards = sorted(glob.glob(pat))
    if not shards:
        sys.exit(f"✗ 没匹配到分片: {pat}")
    print(f"分片 {len(shards)} 个")

    t0 = time.time()
    ds = concatenate_datasets([Dataset.from_file(p) for p in shards])
    print(f"拼接完成 {time.time() - t0:.0f}s  行数={len(ds):,}")
    print("列:", ds.column_names)
    print("特征:", {k: str(v)[:70] for k, v in list(ds.features.items())})
    if len(ds) and "input_ids" in ds.column_names:
        first = ds[0]
        n_mask = sum(1 for x in first["labels"] if x != -100)
        print(f"首样本: input_ids={len(first['input_ids'])} labels 非 -100={n_mask}")

    out.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(str(out))
    print(f"✓ 已落盘 -> {out}（用时 {time.time() - t0:.0f}s）")
    print("  之后把 configs 里的 tokenized_path 指到这个目录即可跳过 tokenize")


if __name__ == "__main__":
    main()

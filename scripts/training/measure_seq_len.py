#!/usr/bin/env python
"""从 tokenize 缓存里量样本长度分布, 并模拟 batch 的 padding 行为。

目的: 判断 OOM 的余量该从哪来 —— 是"最长的样本"还是"pad 到 batch 内最长"造成的。
读的是 datasets 的 map 缓存 (arrow 分片), 只取 input_ids 列, 不碰图像。
"""
import glob
import random

import pyarrow as pa
import pyarrow.compute as pc

CACHE = "/home/ml-user/workdir/cache/hf/datasets/json/default-3d379ea0a8bd3d7f/*/*/cache-b29e49275b12fcb0_*.arrow"
VOCAB = 152064

lengths = []
for path in sorted(glob.glob(CACHE)):
    t = pa.ipc.open_stream(path).read_all().column("input_ids")
    lengths.extend(pc.list_value_length(t).to_pylist())

lengths = [x for x in lengths if x]
n = len(lengths)
lengths_sorted = sorted(lengths)
print(f"样本数 {n}")
for q in (0.5, 0.9, 0.95, 0.99, 1.0):
    print(f"  p{q*100:>4.0f} = {lengths_sorted[min(n-1, int(n*q))]}")
print(f"  均值 {sum(lengths)/n:.0f}")


def sim(bs, grouped):
    """返回 (batch 内最大长度的均值, 该长度的 fp32 logits GiB)"""
    maxes = []
    if grouped:
        idx = list(range(n))
        for i in range(0, n - bs, bs):
            chunk = idx[i:i + bs]
            maxes.append(max(lengths[j] for j in chunk))
    else:
        rng = random.Random(42)
        for _ in range(20000):
            batch = rng.sample(lengths, bs)
            maxes.append(max(batch))
    avg = sum(maxes) / len(maxes)
    gib = bs * avg * VOCAB * 4 / 2**30
    print(f"  batch={bs} {'按长度分组' if grouped else '随机':<10}: "
          f"batch 内最长 token 均值 {avg:7.1f}  →  fp32 logits ≈ {gib:5.2f} GiB")
    return avg


print("\n=== 随机采样 (现状) ===")
sim(8, False)
sim(4, False)

print("\n=== 若开 group_by_length: true ===")
sim(8, True)
sim(4, True)

#!/usr/bin/env python3
"""精确统计数据集的 token 数（文本 + 视觉），用于训练笔记。

=== 为什么要专门写一个工具 ==================================================
训练笔记本里要记「这次训练烧了多少 token」，而 token 数**不能靠字符数估算**：
  * 文本：中英文 token/字符比差异极大；用字符数估会错得离谱
  * 视觉：Qwen2-VL 用 Naive Dynamic Resolution，**图越大 token 越多**，
          且受 `image_max_pixels` 约束 —— 必须按真实尺寸算

本工具的做法：
  文本 token -> 用 `Qwen2VLProcessor` 的 tokenizer **真实编码**
  视觉 token -> 调用 **真实的 image_processor**（就是我们训练时用的那个），
               读 `image_grid_thw` 直接算，不做任何手推公式

用法:
    python scripts/training/measure_tokens.py                    # 抽查 5000 条
    python scripts/training/measure_tokens.py --n 20000
    python scripts/training/measure_tokens.py --full             # 全量 (慢)
    python scripts/training/measure_tokens.py --file data/processed/dpo_10k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
MODEL = ROOT / "models/Qwen2-VL-2B-Instruct"

IMAGE_MAX_PIXELS = 262144      # 与训练配置一致
IMAGE_MIN_PIXELS = 1024
VIDEO_MAX_PIXELS = 16384
VIDEO_MAXLEN = 32


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/processed/sft_400k.clean.jsonl"))
    ap.add_argument("--n", type=int, default=5000, help="抽样条数")
    ap.add_argument("--full", action="store_true", help="全量统计(慢)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from PIL import Image
    import io

    proc = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    tok = proc.tokenizer
    # 与训练时一致：把 max_pixels 塞进 image_processor
    proc.image_processor.max_pixels = IMAGE_MAX_PIXELS
    proc.image_processor.min_pixels = IMAGE_MIN_PIXELS
    merge = proc.image_processor.merge_size

    def n_img_tokens(p: Path):
        """调用真实 image_processor, 从 grid 反推 token 数。"""
        try:
            im = Image.open(p)
            out = proc.image_processor(images=[im], return_tensors="pt")
            g = out["image_grid_thw"][0]
            return int(g[0] * g[1] * g[2]) // (merge ** 2)
        except Exception:
            return None

    rows = []
    t0 = time.time()
    with open(args.file) as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
    print(f"  载入 {len(rows):,} 条 ({time.time()-t0:.1f}s)")

    if not args.full and len(rows) > args.n:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]
        print(f"  抽样 {len(rows):,} 条 (种子 {args.seed})")

    txt_toks, img_toks, vis_toks = [], [], []
    n_img = n_vid = 0
    by_src = {}
    t0 = time.time()

    for i, r in enumerate(rows):
        # --- 文本 token (真实编码) ---
        text = ""
        for m in r.get("messages", []):
            text += m.get("content", "") + "\n"
        n_text = len(tok(text, add_special_tokens=False)["input_ids"])
        txt_toks.append(n_text)

        # --- 视觉 token (真实 image_processor) ---
        nv = 0
        if r.get("images"):
            n_img += 1
            for p in r["images"]:
                t = n_img_tokens(Path(p))
                if t:
                    nv += t
        if r.get("videos"):
            n_vid += 1
            # 视频: 帧数上限 × 每帧 token (video_max_pixels=16384 -> 约 20)
            nv += VIDEO_MAXLEN * (VIDEO_MAX_PIXELS // (28 * 28) // (merge ** 2) + 4)
        img_toks.append(nv)
        vis_toks.append(n_text + nv)

        s = r.get("source", "?")
        b = by_src.setdefault(s, [0, 0, 0])
        b[0] += 1; b[1] += n_text; b[2] += nv

        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"    {i+1:,}/{len(rows):,}  ({i/el+1:.1f} 条/s)")

    n = len(rows)
    tot_text = sum(txt_toks)
    tot_vis = sum(img_toks)
    tot_all = sum(vis_toks)

    print("\n" + "=" * 66)
    print(f"  统计文件: {Path(args.file).name}")
    print(f"  样本数:   {n:,}" + ("" if args.full else f" (抽样自全量)"))
    print("=" * 66)
    print(f"\n  文本 token / 样本:")
    print(f"    均值 {tot_text/n:,.0f}   中位 {pct(txt_toks,50):,}   "
          f"P95 {pct(txt_toks,95):,}   最大 {max(txt_toks):,}")
    print(f"\n  视觉 token / 样本:")
    print(f"    均值 {tot_vis/n:,.0f}   中位 {pct(img_toks,50):,}   "
          f"P95 {pct(img_toks,95):,}   最大 {max(img_toks):,}")
    print(f"    含图像样本 {n_img:,}   含视频样本 {n_vid:,}")
    print(f"\n  ★ 合计 token:")
    print(f"    均值 {tot_all/n:,.1f} / 样本")
    print(f"    抽样合计 {tot_all:,}")
    if not args.full:
        scale = 1
        print(f"    (抽样 {n:,}, 若按同分布外推全量需乘全量/抽样比)")
    print(f"\n  分来源:")
    print(f"    {'来源':<16}{'条数':>10}{'文本tok/条':>14}{'视觉tok/条':>14}")
    for s, (c, t, v) in sorted(by_src.items()):
        print(f"    {s:<16}{c:>10,}{t/c:>14,.0f}{v/c:>14,.0f}")

    out = {
        "file": str(args.file), "n": n, "full": args.full,
        "text_tokens": {"mean": tot_text/n, "p50": pct(txt_toks,50),
                        "p95": pct(txt_toks,95), "max": max(txt_toks), "total": tot_text},
        "visual_tokens": {"mean": tot_vis/n, "p50": pct(img_toks,50),
                          "p95": pct(img_toks,95), "max": max(img_toks), "total": tot_vis},
        "total_tokens": {"mean": tot_all/n, "total": tot_all},
        "n_image_samples": n_img, "n_video_samples": n_vid,
        "by_source": {s: {"n": c, "text_mean": t/c, "visual_mean": v/c}
                      for s, (c, t, v) in by_src.items()},
    }
    o = ROOT / "docs/training" / f"tokens_{Path(args.file).stem}.json"
    o.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  结果写入 {o.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

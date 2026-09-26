#!/usr/bin/env python3
"""SFT 数据抽取 + 统一为 ShareGPT 格式。

=== 「40 万条怎么抽」的决策 (用户授权由我决定) ===============================
配额 (总量 400,000):
    LLaVA-665K      220,000   ← 图像来自 COCO / GQA / TextVQA / VisualGenome
    ShareGPT4V       80,000   ← 取 cap100k 中 coco+llava 两组 (图像同为 COCO)
    VideoInstruct   100,000   ← 视频, 且仅取时长 < 3 分钟的片段
    ---------------------------------------------------------------
    合计            400,000

为什么这样分:
  1. **只选图像真能拿到的来源。** 实测 LLaVA-665K 里 ocr_vqa(80,000) 的上游
     约 250GB 且需逐图匹配, 成本远高于收益, 故整组剔除; 另有 40,688 条是
     **纯文本 ShareGPT 对话(无 image 字段)**, 不属于视觉数据, 一并剔除。
     可用池因此是 coco/gqa/textvqa/vg 共 544,610 条。
  2. **LLaVA 配额按可用池比例分层抽样**, 保证四个图像来源都被覆盖, 而不是
     被 coco 一家(364,100, 占 67%)淹没。
  3. **ShareGPT4V 只取 coco+llava 两组**, 因为它们复用 COCO 图像, 不需要
     再下 ShareGPT4V 自建图像包(sam/wikiart/web-* 共 21,998 条因此放弃)。
  4. **视频占 100,000**, 与图像数据形成模态互补 —— 这正是 PLAN D6 想保住
     Video-MME 评测轴意义的做法。

⚠️ 视频与图像的图像本体需先下载 (01b_images.sh / 视频抓取), 缺失的样本会被
   自动跳过并计入 statistics, 因此首次运行本脚本的产出可能少于 400,000 条。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW, PROCESSED, EXCLUDED_PREFIX_REASON, load_json,  # noqa: E402
                    resolve_image, to_sharegpt, write_jsonl, text_hash)

QUOTA = {"llava665k": 220_000, "sharegpt4v": 80_000, "videoinstruct": 100_000}
SEED = 42

ANN = RAW / "annotations"
IMAGE_TASK_PREFIXES = tuple(["coco", "gqa", "textvqa", "vg"])


def select_llava(quota: int, stats: dict):
    """从 LLaVA-665K 分层抽样。"""
    data = load_json(ANN / "llava_v1_5_mix665k.json")
    stats["llava_total_records"] = len(data)

    pool = collections.defaultdict(list)
    for rec in data:
        img = rec.get("image")
        if not img:                                   # 纯文本 ShareGPT 对话
            stats["llava_dropped_no_image"] += 1
            continue
        prefix = img.replace("\\", "/").lstrip("./").split("/", 1)[0]
        if prefix in EXCLUDED_PREFIX_REASON:
            stats[f"llava_dropped_excluded_{prefix}"] += 1
            continue
        if prefix not in IMAGE_TASK_PREFIXES:
            stats[f"llava_dropped_unknown_prefix_{prefix}"] += 1
            continue
        pool[prefix].append(rec)

    stats["llava_pool_by_prefix"] = {k: len(v) for k, v in sorted(pool.items())}
    total_pool = sum(len(v) for v in pool.values())
    rng = random.Random(SEED)

    # 按可用池比例分层, 末位用最大余数法补齐到 quota
    picked, alloc = [], {}
    for prefix, recs in sorted(pool.items()):
        n = int(quota * len(recs) / total_pool)
        alloc[prefix] = n
    short = quota - sum(alloc.values())
    for prefix in sorted(pool, key=lambda p: -len(pool[p]))[:short]:
        alloc[prefix] += 1

    for prefix, recs in sorted(pool.items()):
        rng.shuffle(recs)
        picked.extend(recs[: alloc[prefix]])
    stats["llava_alloc"] = alloc
    return picked


def select_sharegpt4v(quota: int, stats: dict):
    """ShareGPT4V cap100k 中只取 coco + llava 两组 (图像可复用 COCO)。"""
    data = load_json(ANN / "sharegpt4v_instruct_gpt4-vision_cap100k.json")
    stats["sharegpt4v_total_records"] = len(data)
    pool = [r for r in data
            if (r.get("image") or "").replace("\\", "/").lstrip("./")
            .split("/", 1)[0] in ("coco", "llava")]
    stats["sharegpt4v_pool"] = len(pool)
    rng = random.Random(SEED + 1)
    rng.shuffle(pool)
    return pool[:quota]


def select_video(quota: int, stats: dict, max_minutes: float,
                 fraction: float = 1.0, only_existing: bool = True):
    """VideoInstruct: 按**视频片段**抽样, 再取这些片段的全部 turn。

    ⚠️ 为什么按片段抽而不是按 turn 抽:
        每个片段平均有 7.5 个问题。若按 turn 随机抽, 会把同一片段的问题打散,
        模型看到的是"半截视频的零散问题"; 按片段抽则保留完整片段,
        视频侧的信号是连贯的。
        (实测: 7,113 个可用片段 覆盖 53,150 条 turn, 平均 7.5 问/片段)

    fraction: 取多少比例的**片段**。用户要求「视频只抽 1/3」即 fraction=1/3。

    时长过滤在抓取阶段完成 (见 fetch_videos.py), 这里只做片段级抽样。
    """
    data = load_json(ANN / "VideoInstruct100K.json")
    stats["video_total_turns"] = len(data)
    stats["video_unique_ids"] = len({r["video_id"] for r in data})

    kept_manifest = RAW / "videos" / "kept_videos.json"
    if not kept_manifest.exists():
        stats["video_skipped_reason"] = f"缺少 {kept_manifest} (视频尚未抓取)"
        return []
    kept = set(load_json(kept_manifest))

    # 只保留**磁盘上确实存在**的片段 (坏文件已在 scan_bad_videos.py 里清掉)
    if only_existing:
        clip_dir = RAW / "videos" / "clips"
        kept = {v for v in kept if (clip_dir / f"{v}.mp4").exists()}
    stats["video_clips_available"] = len(kept)

    # ★ 片段级抽样
    clips = sorted(kept)
    rng = random.Random(SEED + 2)
    rng.shuffle(clips)
    if fraction < 1.0:
        n_keep = max(1, int(round(len(clips) * fraction)))
        clips = clips[:n_keep]
    stats["video_clips_sampled"] = len(clips)
    stats["video_fraction"] = fraction
    picked = set(clips)

    rows = [r for r in data if r["video_id"] in picked]
    stats["video_turns_after_duration_filter"] = len(rows)
    rng2 = random.Random(SEED + 3)
    rng2.shuffle(rows)
    return rows[:quota]


def convert(records, source: str, stats: dict, media_kind: str):
    out, seen = [], set()
    for rec in records:
        msgs = to_sharegpt(rec.get("conversations") or []) if media_kind == "image" else \
               [{"role": "user", "content": "<video>\n" + rec["q"].strip()},
                {"role": "assistant", "content": rec["a"].strip()}]
        if not msgs:
            stats[f"{source}_dropped_bad_conversation"] += 1
            continue

        if media_kind == "image":
            p = resolve_image(rec["image"])
            if p is None:
                stats[f"{source}_dropped_unmapped_image"] += 1
                continue
            if not p.exists():
                stats[f"{source}_dropped_missing_file"] += 1
                continue
            # ⚠️ 必须写**绝对路径**。LLaMA-Factory 解析 sharegpt 的 images 字段时
            #    是相对**数据集文件所在目录**的, 而我们这里原本写的是相对项目根的
            #    路径 —— 两者拼起来会变成 data/processed/data/raw/... 而找不到文件。
            media = [str(p.resolve())]
        else:
            vp = (RAW.parent.parent / "data/raw/videos/clips" / f"{rec['video_id']}.mp4")
            if not vp.exists():
                stats[f"{source}_dropped_missing_video"] += 1
                continue
            media = [str(vp.resolve())]

        h = text_hash(json.dumps(msgs, ensure_ascii=False))
        if h in seen:
            stats[f"{source}_dropped_duplicate"] += 1
            continue
        seen.add(h)

        # ⚠️ LLaMA-Factory 对图像与视频用**两个不同的列** (见 converter.py):
        #    images 列 -> 对应内容里的 <image> 占位符
        #    videos 列 -> 对应内容里的 <video> 占位符
        #    把视频样本塞进 images 列会直接报
        #    「The number of images does not match the number of <image> tokens」。
        #    `media` 保留给本项目自己的过滤/统计脚本用。
        out.append({"id": str(rec.get("id") or rec.get("video_id")),
                    "source": source, "media_type": media_kind,
                    "messages": msgs,
                    "images": media if media_kind == "image" else [],
                    "videos": media if media_kind == "video" else [],
                    "media": media})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED / "sft_400k.jsonl"))
    ap.add_argument("--video-max-minutes", type=float, default=3.0)
    ap.add_argument("--target", type=int, default=400_000,
                    help="SFT 总目标条数; 缺口由 LLaVA 侧自动补足")
    ap.add_argument("--llava", type=int, default=-1,
                    help="LLaVA 配额; -1 = 自动补足到 --target")
    ap.add_argument("--sharegpt4v", type=int, default=QUOTA["sharegpt4v"])
    ap.add_argument("--video", type=int, default=QUOTA["videoinstruct"])
    ap.add_argument("--video-fraction", type=float, default=1.0,
                    help="视频**片段**的抽取比例 (1/3 即 0.333); 按片段抽以保留完整问答")
    args = ap.parse_args()

    stats = collections.Counter()

    # 先取「供给受限」的两路 —— 它们能拿到多少不由我们决定
    # (ShareGPT4V 受可用图像限制; 视频受 YouTube 可用性限制),
    # 再用 LLaVA **补足到目标**, 而不是三方各给一个写死的配额。
    # 这样即使某一路缺失, 总量也能守住 400K。
    rows_video = convert(select_video(args.video, stats, args.video_max_minutes,
                                      fraction=args.video_fraction),
                         "videoinstruct", stats, "video")
    rows_sg = convert(select_sharegpt4v(args.sharegpt4v, stats), "sharegpt4v", stats, "image")

    llava_quota = args.llava
    if llava_quota < 0:
        llava_quota = max(0, args.target - len(rows_video) - len(rows_sg))
        print(f"LLaVA 配额自动补足: {args.target:,} - {len(rows_video):,}(视频) "
              f"- {len(rows_sg):,}(ShareGPT4V) = {llava_quota:,}")
    rows_llava = convert(select_llava(llava_quota, stats), "llava665k", stats, "image")

    rows = rows_llava + rows_sg + rows_video
    random.Random(SEED + 9).shuffle(rows)
    n = write_jsonl(Path(args.out), rows)

    by = collections.Counter(r["source"] for r in rows)
    print(f"\n写出 {n} 条 -> {args.out}")
    for k, v in sorted(by.items()):
        print(f"   {k:<16}{v:>8,}")
    gap = args.target - n
    print(f"\n目标 {args.target:,}; 实际 {n:,} (缺口 {gap:,})")
    if gap > 0:
        print("  缺口主要来源: 视频尚未下载完 (见 logs/video_pipeline.log), "
              "或 LLaVA 可用图像池已耗尽")
    print("\n--- 丢弃统计 ---")
    for k in sorted(stats):
        print(f"   {k:<44}{stats[k]}")
    (PROCESSED / "sft_selection_stats.json").parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "sft_selection_stats.json", "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else v) for k, v in stats.items()},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

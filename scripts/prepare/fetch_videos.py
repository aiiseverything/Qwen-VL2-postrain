#!/usr/bin/env python3
"""抓取 VideoInstruct-100K 中**时长 < 3 分钟**的视频。

=== 为什么需要这一步 (问题 P2) =============================================
    VideoInstruct100K.json 的字段是 {q, a, video_id}, 而 video_id 形如
        v_k_ZXmr8pmrs
    这是 **ActivityNet 的命名约定 —— `v_` 后面是 YouTube 视频 ID**。
    也就是说视频**不在 HF 上**, 必须用 yt-dlp 从 YouTube 抓取。

    10 万条 turn 只对应 **13,303 个唯一视频**, 所以按视频去重后抓取即可覆盖
    全部 turn —— 实际工作量比「10 万条」的直觉小一个数量级。

=== 两阶段设计 ==============================================================
    阶段 1 (元数据): 只取每个视频的时长, 用于判定 <3 分钟。**不下载视频本体**。
    阶段 2 (本体)  : 仅下载通过时长筛选的视频。

    分两阶段的原因: 只查元数据比下载视频便宜几个数量级。若先下后筛,
    会白白下载大量超时长视频。产物 kept_videos.json 供 to_sharegpt.py 读取。

=== 诚实声明 ================================================================
    YouTube 抓取存在下架 / 地区封锁 / 限流等不确定性, **成功率无法预先保证**。
    本脚本会把失败原因分类计数并写入报告, 不隐藏失败。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW  # noqa: E402

ANN = RAW / "annotations" / "VideoInstruct100K.json"
VID_DIR = RAW / "videos"
CLIP_DIR = VID_DIR / "clips"
KEPT = VID_DIR / "kept_videos.json"
META = VID_DIR / "video_metadata.json"


def youtube_id(video_id: str) -> str:
    """ActivityNet 的 video_id 形如 `v_k_ZXmr8pmrs`, 其中 `v_` 只是标记,
    真正的 YouTube ID 是 `v_` 之后的部分。

    ⚠️ 直接把整个 video_id 拼进 URL (watch?v=v_k_ZXmr8pmrs) 会让**每一个**
       视频都返回 "This video is unavailable", 看起来像是数据全部失效,
       实际只是前缀没剥掉。实测剥离前后可用率 0% → 80%。见 docs/05 问题 P14。
    """
    return video_id[2:] if video_id.startswith("v_") else video_id


def yt_dlp() -> str:
    import shutil
    p = shutil.which("yt-dlp")
    if not p:
        here = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "yt-dlp"
        if here.exists():
            return str(here)
        sys.exit("找不到 yt-dlp: pip install yt-dlp 或 uv pip install yt-dlp")
    return p


# ⚠️ 关键: yt-dlp 默认的 web 客户端在本网络下对大量**仍然存在**的视频返回
#    "This video is not available" —— 实测这类视频 70% 其实可以取到。
#    改用 android 客户端后 7/10 恢复。必须带上这个参数, 否则会误判
#    「数据已失效」而白白丢掉约 5,000 个视频。见 docs/05 问题 P16。
YT_ARGS = ["--extractor-args", "youtube:player_client=android,web"]


def probe_duration(vid: str, timeout: int = 40) -> tuple[str, float | None, str]:
    """只取元数据, 不下载。返回 (vid, 秒数, 状态)。"""
    url = f"https://www.youtube.com/watch?v={youtube_id(vid)}"
    try:
        r = subprocess.run(
            [yt_dlp(), "--skip-download", "--no-warnings", "--print", "%(duration)s",
             "--socket-timeout", "20", *YT_ARGS, url],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return vid, None, "timeout"
    out = (r.stdout or "").strip().splitlines()
    for line in out:
        try:
            return vid, float(line), "ok"
        except ValueError:
            continue
    err = (r.stderr or "").lower()
    for tag, key in (("unavailable", "unavailable"), ("private", "private"),
                     ("removed", "removed"), ("blocked", "blocked"),
                     ("age", "age_restricted")):
        if tag in err:
            return vid, None, key
    return vid, None, "error"


def download(vid: str, max_height: int, timeout: int = 300) -> tuple[str, str]:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out = CLIP_DIR / f"{vid}.mp4"
    if out.exists() and out.stat().st_size > 1024:
        return vid, "cached"
    url = f"https://www.youtube.com/watch?v={youtube_id(vid)}"
    # ⚠️ 格式选择器必须**足够宽容** (见 docs/05 问题 P34)。
    #    早先写成 `bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/...`,
    #    硬性要求视频是 mp4、音频是 m4a —— 但大量视频**只提供 webm/opus**,
    #    于是直接报 `Requested format is not available`。
    #    实测这些视频其实完全可用(探测阶段能取到时长), 纯粹是选择器太窄。
    #    实测: 加宽前约 49% 失败, 其中绝大多数属于这一类。
    #    容器统一交给 --merge-output-format mp4 处理, 不必在选择器里限制。
    fmt = (f"bv*[height<={max_height}]+ba/"          # 首选: 分轨视频+音频
           f"b[height<={max_height}]/"               # 次选: 合轨且不超高度
           f"bv*+ba/b/"                              # 再退: 不限高度的分轨/合轨
           f"best")                                  # 兜底: 有啥用啥
    try:
        r = subprocess.run(
            [yt_dlp(), "-f", fmt, "--merge-output-format", "mp4",
             "--no-warnings", "--no-playlist", "--socket-timeout", "20",
             *YT_ARGS, "-o", str(out), url],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return vid, "timeout"
    if out.exists() and out.stat().st_size > 1024:
        return vid, "ok"
    err = (r.stderr or "").lower()
    if "requested format" in err:
        return vid, "no_format"
    if "private" in err:
        return vid, "private"
    if "unavailable" in err or "removed" in err:
        return vid, "unavailable"
    return vid, "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=3.0, help="时长上限(分钟)")
    ap.add_argument("--workers", type=int, default=4, help="并发数 (对训练 I/O 友好)")
    ap.add_argument("--max-height", type=int, default=480, help="下载分辨率上限")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个视频 (冒烟)")
    ap.add_argument("--stage", choices=["probe", "download", "all"], default="all")
    ap.add_argument("--reprobe-errors", action="store_true",
                    help="只重探已记录为 error/timeout 的视频 (换 android 客户端后值得重试)")
    args = ap.parse_args()

    VID_DIR.mkdir(parents=True, exist_ok=True)
    turns = json.load(open(ANN))
    vids = sorted({t["video_id"] for t in turns})
    if args.limit:
        vids = vids[: args.limit]
    print(f"唯一视频 {len(vids):,} 个 (覆盖 {len(turns):,} 条 turn)")
    print(f"时长筛选上限: {args.max_minutes} 分钟\n")

    # ---- 阶段 1: 元数据 --------------------------------------------------
    meta = json.load(open(META)) if META.exists() else {}
    if args.reprobe_errors:
        todo = [v for v in vids
                if meta.get(v, {}).get("status") in ("error", "timeout", "blocked")]
        print(f"重探模式: {len(todo):,} 个此前 error/timeout 的视频")
    else:
        todo = [v for v in vids if v not in meta]
    if args.stage in ("probe", "all") and todo:
        print(f"[1/2] 探测时长: {len(todo):,} 个待探测 (已有 {len(meta):,})")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(probe_duration, v): v for v in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                vid, dur, status = fut.result()
                meta[vid] = {"duration": dur, "status": status}
                if i % 100 == 0 or i == len(todo):
                    print(f"      {i}/{len(todo)}")
                    json.dump(meta, open(META, "w"))
        json.dump(meta, open(META, "w"))

    # ---- 筛选 -------------------------------------------------------------
    limit_s = args.max_minutes * 60
    kept, reasons = [], {}
    for v in vids:
        m = meta.get(v, {})
        st, dur = m.get("status"), m.get("duration")
        if st == "ok" and dur is not None and dur < limit_s:
            kept.append(v)
        else:
            reasons[st or "unknown"] = reasons.get(st or "unknown", 0) + 1
    json.dump(kept, open(KEPT, "w"))

    print(f"\n[筛选] 通过 (<{args.max_minutes}min): {len(kept):,} / {len(vids):,}")
    print("       未通过的分类统计:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"         {k:<16}{v:>7,}")
    turns_kept = sum(1 for t in turns if t["video_id"] in set(kept))
    print(f"       覆盖 turn 数: {turns_kept:,}")

    # ---- 阶段 2: 下载 -----------------------------------------------------
    if args.stage in ("download", "all") and kept:
        print(f"\n[2/2] 下载 {len(kept):,} 个视频 (≤{args.max_height}p, {args.workers} 并发)")
        done, dr = 0, {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download, v, args.max_height) for v in kept]
            for fut in as_completed(futs):
                vid, status = fut.result()
                dr[status] = dr.get(status, 0) + 1
                done += 1
                if done % 50 == 0 or done == len(kept):
                    print(f"      {done}/{len(kept)}  {dr}")
        print("\n       下载结果分类:")
        for k, v in sorted(dr.items(), key=lambda x: -x[1]):
            print(f"         {k:<16}{v:>7,}")
        ok = dr.get("ok", 0) + dr.get("cached", 0)
        print(f"\n       实际可用视频: {ok:,} / {len(kept):,} "
              f"({100*ok/max(len(kept),1):.1f}%)")

    print(f"\n产物: {KEPT}")
    print("下一步: python3 scripts/prepare/to_sharegpt.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只下载 Video-MME 中**时长符合要求**的视频（默认：只取 short 子集，即 <2 分钟）。

=== 为什么要这么做 ==========================================================
Video-MME 官方仓库把 900 个视频打包成 **20 个 ~5 GB 的 zip**（合计约 101 GB），
而且**分包不按时长**（按 YouTube ID 排），所以想按「3 分钟以内」筛选，
朴素做法只能把 101 GB 全下下来再筛。

本脚本绕开这一点：**利用 HTTP Range 请求读取远端 zip 的中央目录**，
定位到需要的成员，**只下载那几十个文件的字节区间**。
实测：补齐 short 子集只需约 1–2 GB，而不是 26 GB（补 5 个缺包）或 101 GB（全量）。

Video-MME 的时长分级（标注里的 `duration` 列）:
    short   < 2 分钟   ← 300 个视频 / 900 题   ← 本脚本默认目标
    medium  4–15 分钟
    long    > 15 分钟
「3 分钟以内」严格来说 = short 子集（<2min ⊂ <3min）。

用法:
    python scripts/download/05_videomme_short.py                 # 默认取 short
    python scripts/download/05_videomme_short.py --dry-run       # 只统计不下载
    python scripts/download/05_videomme_short.py --kinds short medium
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
BENCH = PROJECT_ROOT / "data/raw/bench/Video-MME"
ANN = BENCH / "videomme/test-00000-of-00001.parquet"
OUT = BENCH / "videos"

HF_REPO = "lmms-eval/Video-MME"
HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
N_CHUNKS = 20


def chunk_name(i: int) -> str:
    return f"videos_chunked_{i:02d}.zip"


def local_chunk_ids() -> set[str]:
    """本地已有的 chunk 里包含哪些 videoID。"""
    have: set[str] = set()
    for z in sorted(BENCH.glob("videos_chunked_*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                have |= {Path(n).stem for n in zf.namelist() if n.endswith(".mp4")}
        except zipfile.BadZipFile:
            print(f"  [warn] 本地 zip 损坏, 跳过: {z.name}")
    return have


def remote_chunk_ids(url: str, tries: int = 3) -> set[str]:
    """通过 Range 请求读远端 zip 的中央目录, 拿到成员列表 (不下载内容)。"""
    from remotezip import RemoteZip
    last = None
    for k in range(tries):
        try:
            with RemoteZip(url) as rz:
                return {Path(n).stem for n in rz.namelist() if n.endswith(".mp4")}
        except Exception as e:                           # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise last


def _read_with_retry(rz, name: str, tries: int = 4):
    """单文件读取 + 重试。

    ⚠️ 本机到 HF 的连接是**间歇性**的 (实测同一 URL 连续 3 次 000 之后又连续 4 次 200),
    典型错误是 `Errno 101 Network is unreachable` 与 `IncompleteRead`。
    没有重试的话, 一次抖动就会永久丢掉那个视频。
    """
    last = None
    for k in range(tries):
        try:
            return rz.read(name)
        except Exception as e:                           # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise last


def extract_from_remote(url: str, wanted: set[str], out_dir: Path) -> list[str]:
    """只下载 wanted 里的成员。"""
    from remotezip import RemoteZip
    got = []
    with RemoteZip(url) as rz:
        names = {Path(n).stem: n for n in rz.namelist() if n.endswith(".mp4")}
        todo = [v for v in wanted if v in names]
        for i, vid in enumerate(sorted(todo), 1):
            dest = out_dir / f"{vid}.mp4"
            if dest.exists() and dest.stat().st_size > 1024:
                got.append(vid); continue
            try:
                data = _read_with_retry(rz, names[vid])
                dest.write_bytes(data)
                got.append(vid)
            except Exception as e:                       # noqa: BLE001
                print(f"    [warn] {vid} 失败: {type(e).__name__}: {str(e)[:90]}")
            if i % 10 == 0:
                print(f"      {i}/{len(todo)}")
    return got


def extract_from_local(wanted: set[str], out_dir: Path) -> list[str]:
    """从已下载的本地 chunk 里抽取需要的成员。"""
    got = []
    for z in sorted(BENCH.glob("videos_chunked_*.zip")):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".mp4")}
        todo = [v for v in wanted if v in names]
        for vid in todo:
            dest = out_dir / f"{vid}.mp4"
            if dest.exists() and dest.stat().st_size > 1024:
                got.append(vid); continue
            try:
                dest.write_bytes(zf.read(names[vid])); got.append(vid)
            except Exception as e:                       # noqa: BLE001
                print(f"    [warn] {vid}: {e}")
        zf.close()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["short"],
                    help="要取的时长类别 (short/medium/long); 默认 short = <2min")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(ANN)
    print("Video-MME 时长分布 (唯一视频):")
    print("   " + str(df.groupby('duration')['videoID'].nunique().to_dict()))

    target = set(df[df["duration"].isin(args.kinds)]["videoID"])
    print(f"\n目标类别 {args.kinds} -> {len(target)} 个视频")

    OUT.mkdir(parents=True, exist_ok=True)
    already = {p.stem for p in OUT.glob("*.mp4")}
    have_local = local_chunk_ids()
    print(f"  已解压到 {OUT.name}/: {len(already)}")
    print(f"  本地 chunk 覆盖: {len(have_local & target)}")

    need = target - already
    if not need:
        print("\n✅ 目标视频已全部就绪"); return 0

    # 本地 chunk 能解决一部分
    from_local = need & have_local
    print(f"\n[1/2] 从本地 chunk 抽取 {len(from_local)} 个")
    if not args.dry_run and from_local:
        got = extract_from_local(from_local, OUT)
        print(f"      实得 {len(got)}")
    need -= from_local

    if not need:
        print("\n✅ 全部就绪"); return 0

    # 剩下的去远端按需取
    print(f"\n[2/2] 远端按需取 {len(need)} 个 (Range 请求, 不下载整包)")
    missing_chunks = []
    for i in range(1, N_CHUNKS + 1):
        url = HF_BASE.format(repo=HF_REPO, name=chunk_name(i))
        if args.dry_run:
            print(f"   chunk {i:02d}: (dry-run 不查远端)"); continue
        try:
            ids = remote_chunk_ids(url)
        except Exception as e:                           # noqa: BLE001
            print(f"   chunk {i:02d}: 读取失败 {type(e).__name__}"); continue
        hit = need & ids
        if not hit:
            continue
        print(f"   chunk {i:02d}: 命中 {len(hit)} 个 -> 下载")
        got = extract_from_remote(url, hit, OUT)
        print(f"             实得 {len(got)}")
        need -= set(got)
        if not need:
            break

    final = {p.stem for p in OUT.glob("*.mp4")}
    print(f"\n=== 汇总 ===")
    print(f"  目标 {len(target)} 个 | 已就绪 {len(target & final)} | 仍缺 {len(target - final)}")
    if target - final:
        print(f"  仍缺: {sorted(target - final)[:10]}{' ...' if len(target - final) > 10 else ''}")
    print(f"  视频目录: {OUT}")
    return 0 if not (target - final) else 1


if __name__ == "__main__":
    sys.exit(main())

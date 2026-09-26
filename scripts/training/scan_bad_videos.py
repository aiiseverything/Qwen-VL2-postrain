#!/usr/bin/env python3
"""扫描视频目录，找出**解码不出任何帧**的坏文件。

=== 为什么需要它 (docs/05 问题 P36) ==========================================
7,246 个 YouTube 下载里混有 yt-dlp 失败合并留下的残片 —— 文件有大小（几十 KB），
但**一帧都解不出来**。这些文件混进 SFT 数据后会在 collator 里炸：

    mm_plugin.py -> video_processor -> np.stack(video)
    ValueError: need at least one array to stack

而且报错发生在 DataLoader worker 内部，**看不到是哪条样本/哪个文件**，
只能靠全量扫描定位。

性能: 只解**首帧**（`next(decode(video=0), None)`）而不是全解 ——
      全解一个 5 MB 视频要几秒，7 千个要几小时；只解首帧约 10 分钟。

用法:
    python scripts/training/scan_bad_videos.py                 # 扫描
    python scripts/training/scan_bad_videos.py --delete        # 扫描并删除坏文件
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
CLIP = ROOT / "data/raw/videos/clips"
OUT = ROOT / "docs/training/bad_videos.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(CLIP))
    ap.add_argument("--delete", action="store_true", help="删除坏文件")
    args = ap.parse_args()

    import av
    d = Path(args.dir)
    files = sorted(d.glob("*.mp4"))
    print(f"扫描 {len(files):,} 个视频 (只解首帧)", flush=True)

    bad = []
    for i, p in enumerate(files, 1):
        try:
            c = av.open(str(p))
            frame = next(c.decode(video=0), None)
            c.close()
            if frame is None:
                bad.append((p, "0 帧"))
        except Exception as e:                       # noqa: BLE001
            bad.append((p, f"{type(e).__name__}"))
        if i % 1000 == 0:
            print(f"  {i:,}/{len(files):,}  坏={len(bad)}", flush=True)

    print(f"\n完成: {len(files):,} 个中坏文件 {len(bad)} 个 "
          f"({len(bad)/max(len(files),1)*100:.2f}%)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(f"{p.name}\t{r}" for p, r in bad))
    print(f"清单 -> {OUT.relative_to(ROOT)}")
    for p, r in bad[:10]:
        print(f"  {p.name:<28} {r}")

    if args.delete and bad:
        for p, _ in bad:
            p.unlink(missing_ok=True)
        print(f"\n已删除 {len(bad)} 个坏文件")
    elif bad:
        print("\n(未删除; 加 --delete 才删)")


if __name__ == "__main__":
    sys.exit(main())

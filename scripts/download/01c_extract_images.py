#!/usr/bin/env python3
"""解压上游图像归档到 data/raw/images/{coco,gqa,textvqa,vg}。

为什么用 Python 而不是 `unzip`(问题 P17):
  1. **ZIP64**: GQA 的 21.8 GB 归档是 ZIP64 格式。系统自带的老 `unzip`
     对大归档会报 "not enough memory for bomb detection" 并**误判为损坏**,
     而它其实完全正常。Python 的 zipfile 对 ZIP64 支持完整。
  2. **失败可见**: `unzip -t` 对损坏归档会持续尝试并最终只给一行总结;
     zipfile 能精确报出哪个成员、什么错误。

解压目标 (与标注里的 image 前缀对应, 见 scripts/prepare/common.py):
    coco/train2017/*.jpg      <- coco_train2017.zip      (顶层 train2017/)
    gqa/images/*.jpg          <- gqa_images.zip          (顶层 images/)
    textvqa/train_images/*.jpg<- textvqa_images.zip      (顶层 train_images/)
    vg/VG_100K/*.jpg          <- vg/images.zip           (顶层 VG_100K/)
    vg/VG_100K_2/*.jpg        <- vg/images2.zip          (顶层 VG_100K_2/)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
IMG = PROJECT_ROOT / "data/raw/images"

# (归档相对路径, 解压目标目录, 期望的顶层目录名, 期望文件数下限)
JOBS = [
    ("coco_train2017.zip",  IMG / "coco",    "train2017",   100_000),
    ("gqa_images.zip",      IMG / "gqa",     "images",      140_000),
    ("textvqa_images.zip",  IMG / "textvqa", "train_images", 20_000),
    ("vg/images.zip",       IMG / "vg",      "VG_100K",      60_000),
    ("vg/images2.zip",      IMG / "vg",      "VG_100K_2",    40_000),
]


def count_jpgs(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for f in os.scandir(d) if f.name.lower().endswith((".jpg", ".jpeg", ".png")))


def verify(z: zipfile.ZipFile, top: str, sample: int = 5) -> tuple[bool, str]:
    """抽查若干个成员的真实读取 —— 只有读得出来才算归档完好。"""
    names = [n for n in z.namelist() if n.startswith(top) and not n.endswith("/")]
    if not names:
        return False, f"归档内找不到 {top}/ 下的文件"
    import itertools
    for n in itertools.islice(names, sample):
        try:
            z.read(n)
        except Exception as e:                      # noqa: BLE001
            return False, f"{n}: {type(e).__name__}: {e}"
    return True, f"抽查 {min(sample,len(names))} 个成员读取正常 (共 {len(names):,})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="即使已解压也重来")
    args = ap.parse_args()

    t0 = time.time()
    failures = []
    for rel, target, top, expect in JOBS:
        zp = IMG / rel
        dest_top = target / top
        have = count_jpgs(dest_top)

        print(f"\n=== {rel}  ->  {target.relative_to(PROJECT_ROOT)}/{top}")
        if not zp.exists():
            print("   [skip] 归档不存在"); failures.append(rel); continue
        if have >= expect and not args.force:
            print(f"   [skip] 已解压 {have:,} 张 (>= {expect:,})"); continue

        try:
            z = zipfile.ZipFile(zp)
        except Exception as e:                      # noqa: BLE001
            print(f"   [FAIL] 无法打开: {e}"); failures.append(rel); continue

        ok, msg = verify(z, top)
        print(f"   校验: {'OK  ' if ok else 'FAIL'} {msg}")
        if not ok:
            failures.append(rel); continue

        print(f"   解压中 -> {dest_top} ...")
        target.mkdir(parents=True, exist_ok=True)
        n = 0
        try:
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                if not member.startswith(top + "/"):
                    continue
                z.extract(member, target)
                n += 1
                if n % 20000 == 0:
                    print(f"      {n:,} ...")
        except Exception as e:                      # noqa: BLE001
            print(f"   [FAIL] 解压中断于第 {n:,} 个: {e}"); failures.append(rel); continue
        got = count_jpgs(dest_top)
        print(f"   ✅ 解压 {n:,} 个成员, 目录内 {got:,} 张图")

    print(f"\n=== 汇总 (耗时 {time.time()-t0:.0f}s) ===")
    for d in ["coco/train2017", "gqa/images", "textvqa/train_images",
              "vg/VG_100K", "vg/VG_100K_2"]:
        print(f"  {d:<26}{count_jpgs(IMG/d):>9,} 张")
    if failures:
        print(f"\n❌ 仍有问题: {failures}")
        return 1
    print("\n✅ 全部解压完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())

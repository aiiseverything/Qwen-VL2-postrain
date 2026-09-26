#!/usr/bin/env python3
"""RLVR 数据构建 —— 产出 EasyR1/verl 可直接读取的 parquet。

EasyR1 期望的列:  problem (str) | answer (str) | images (list[PIL/bytes])

本机实测的源 schema (2026-09-21, pyarrow):
    hiyouga/geometry3k            images:list<struct<bytes,path>>, problem, answer   2101 train
    leonardPKU/GEOQA_R1V_Train_8K image:struct<bytes,path>,      problem, solution   8031 train
    FanqingM/MMK12                (下载后探测)

⚠️ geoqa 的答案列叫 `solution` 而**不是** `answer` —— 直接按 answer 取会得到
   全空的标签, 奖励函数会恒为 0, GRPO 白跑。这是必须显式处理的坑 (见 docs/05 问题 P6)。

图像以 bytes 内嵌在 parquet 里, 因此**无需**额外下载图像 —— 与 SFT 侧
(图像需外部下载)形成鲜明对比。
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, PROCESSED  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402


def _img_bytes(cell):
    """从 struct<bytes,path> 或 list/ndarray<struct> 中取出 PNG/JPEG bytes。

    实测: geometry3k 的 `images` 读出来是 **numpy.ndarray of dict**,
    既不是 list 也不是 tuple —— 只判断 list/tuple 会静默返回 None,
    导致 2101 条训练数据全部被当成「无图」丢弃。见 docs/05 问题 P7。
    """
    if cell is None:
        return None
    # ndarray / list / tuple 等一切非 dict 的可迭代序列 -> 取第一个元素
    if not isinstance(cell, dict):
        try:
            if len(cell) == 0:
                return None
            cell = cell[0]
        except TypeError:
            return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    return None


def build_geometry3k(out_dir: Path):
    f = RAW / "rlvr/geometry3k/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(f)
    rows = []
    for _, r in df.iterrows():
        b = _img_bytes(r["images"])
        if not b:
            continue
        rows.append({"problem": str(r["problem"]).strip(),
                     "answer": str(r["answer"]).strip(),
                     "images": [b]})
    return rows


def build_geoqa(out_dir: Path):
    f = RAW / "rlvr/geoqa_r1v/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(f)
    rows = []
    for _, r in df.iterrows():
        b = _img_bytes(r["image"])
        if not b:
            continue
        # ⚠️ 答案是 `solution` 列
        rows.append({"problem": str(r["problem"]).strip(),
                     "answer": str(r["solution"]).strip(),
                     "images": [b]})
    return rows


def build_mmk12(out_dir: Path):
    import glob
    files = sorted(glob.glob(str(RAW / "rlvr/mmk12/data/train-*.parquet")))
    if not files:
        print("   [skip] MMK12 parquet 未就绪"); return []
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        print(f"   MMK12 {Path(f).name}: cols={list(df.columns)}")
        img_col = next((c for c in ("image", "images") if c in df.columns), None)
        ans_col = next((c for c in ("answer", "solution", "label") if c in df.columns), None)
        q_col = next((c for c in ("problem", "question", "query") if c in df.columns), None)
        if not (img_col and ans_col and q_col):
            print(f"   [warn] 无法识别 MMK12 列, 跳过 {f}"); continue
        for _, r in df.iterrows():
            b = _img_bytes(r[img_col])
            if not b:
                continue
            rows.append({"problem": str(r[q_col]).strip(),
                         "answer": str(r[ans_col]).strip(),
                         "images": [b]})
    return rows


def save(rows, name: str, out_dir: Path):
    if not rows:
        print(f"   [skip] {name}: 0 行"); return None
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.parquet"
    import pyarrow.parquet as pq
    tbl = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(tbl, p)          # 注: pa.Table 没有 .to_parquet(), 须用 pq.write_table
    print(f"   {name:<12} {len(rows):>6,} 行 -> {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROCESSED / "rlvr"))
    args = ap.parse_args()
    out = Path(args.out_dir)

    print("构建 RLVR 数据 (EasyR1 格式: problem / answer / images)")
    save(build_geometry3k(out), "geometry3k", out)
    save(build_geoqa(out), "geoqa_r1v", out)
    save(build_mmk12(out), "mmk12", out)
    print("\n注意: 奖励函数 configs/reward_geo.py 默认做**答案精确匹配**, 不强制 CoT 格式 (PLAN K8)。")


if __name__ == "__main__":
    main()

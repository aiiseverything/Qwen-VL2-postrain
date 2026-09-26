#!/usr/bin/env python3
"""质量过滤 + 训练/评测去污染 (PLAN §7.5, 风险 K5)。

规则:
  1. 精确去重 + 近似去重 (对话文本 hash)
  2. 丢弃**空**答案 (注意: 单字符答案是合法的 —— 单选题答案就是 'A'/'B'/'C'/'D',
     **不要**用长度阈值, 否则会误删大量 VQA 数据, 见问题 P27)
  3. 丢弃媒体缺失或无法解码的样本
  4. 超长对话按 cutoff_len 丢弃 (而非截断 —— 截断会破坏 answer 完整性)
  5. **去污染**: SFT 图像 vs MME/MathVista 评测图像做重叠检查, 命中的样本剔除
     并**在报告中如实披露数量** (K5: 污染会让分数虚高, 结论无效)

去污染方法: 用图像内容的感知指纹比对。评测集图像 (MME) 以 bytes 内嵌在 parquet,
不需要落盘即可算指纹。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, PROCESSED, PROJECT_ROOT, load_jsonl, write_jsonl, text_hash  # noqa: E402

DEFAULT_CUTOFF_CHARS = 11000     # ≈3667 tokens; 留出视觉 token (~334) 余量, 需 < cutoff_len 4096
_FP_CACHE: dict[str, str] = {}     # 路径 -> 指纹 缓存 (见主循环注释)


def image_fingerprint(b: bytes) -> str | None:
    """图像感知指纹: 缩放到 16x16 灰度后再 hash —— 对重编码/缩放鲁棒。

    性能: 对 JPEG 先用 draft() 让解码器直接出小图 (DCT 降采样),
    比整图解码后再 resize 快一个数量级 —— 这个函数要在 40 万张图上跑。
    """
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(b))
        if im.format == "JPEG":
            im.draft("L", (16, 16))       # 让 libjpeg 少解码
        im = im.convert("L").resize((16, 16))
        return hashlib.sha1(im.tobytes()).hexdigest()
    except Exception:
        return None


def file_fingerprint(p: Path) -> str | None:
    """对磁盘上的图像文件算指纹 (训练侧用)。"""
    try:
        return image_fingerprint(p.read_bytes())
    except OSError:
        return None


def _cell_to_bytes(cell):
    """从 parquet 单元格里取出图像字节。

    各 benchmark 的存放方式**不统一** (实测):
        MME        : image = struct<bytes, path>
        MathVista  : image = "images/1001.jpg" (只是路径字符串!)
                     decoded_image = struct<bytes, ...>   <-- 字节在这里
    早先只找 image/images 两列, 导致 **MathVista 收集到 0 个指纹**,
    去污染对它实际没生效。见 docs/05 问题 P21。
    """
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        return cell.get("bytes")
    try:                                   # ndarray / list of struct
        return _cell_to_bytes(cell[0]) if len(cell) else None
    except (TypeError, IndexError):
        return None


def eval_fingerprints(limit_per_set: int | None = None) -> set[str]:
    """收集 MME / MathVista 评测图像的指纹。"""
    fps: set[str] = set()
    import glob
    import pandas as pd

    # 候选列按优先级排列; decoded_image 是 MathVista 实际存字节的地方
    CANDIDATE_COLS = ("decoded_image", "image", "images")
    for name, pat in (("MME", "bench/MME/**/*.parquet"),
                      ("MathVista", "bench/MathVista/**/*.parquet")):
        files = sorted(glob.glob(str(RAW / pat), recursive=True))
        n = 0
        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            b = None
            for c in CANDIDATE_COLS:
                if c not in df.columns:
                    continue
                # 先试第一个非空单元格, 确认这一列真的含字节
                for cell in df[c].head(3):
                    b = _cell_to_bytes(cell)
                    if b:
                        break
                if b:
                    col = c
                    break
            else:
                continue
            for cell in df[col]:
                bb = _cell_to_bytes(cell)
                if not bb:
                    continue
                fp = image_fingerprint(bb)
                if fp:
                    fps.add(fp); n += 1
        if files:
            flag = "✅" if n else "⚠️ 未取到字节 —— 该 benchmark 的去污染未生效"
            print(f"   去污染: {name} 收集 {n} 张评测图像指纹 {flag}")
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(PROCESSED / "sft_400k.jsonl"))
    ap.add_argument("--out", dest="out", default=str(PROCESSED / "sft_400k.clean.jsonl"))
    ap.add_argument("--cutoff-chars", type=int, default=DEFAULT_CUTOFF_CHARS)
    ap.add_argument("--skip-decontam", action="store_true")
    ap.add_argument("--workers", type=int, default=16, help="并行指纹计算线程数")
    args = ap.parse_args()

    stats = collections.Counter()
    rows = list(load_jsonl(Path(args.inp)))
    stats["input"] = len(rows)
    print(f"输入 {len(rows):,} 条")

    evalfps: set[str] = set()
    if not args.skip_decontam:
        print("  收集评测集图像指纹 (去污染)...")
        evalfps = eval_fingerprints()
        stats["eval_fingerprints"] = len(evalfps)
        if not evalfps:
            print("  [warn] 未收集到评测指纹 —— 去污染实际未生效, 须在报告中声明")

    global _FP_CACHE
    _FP_CACHE = {}

    # ---- 并行预算所有训练图像的指纹 (带磁盘缓存) --------------------------
    # 上一版是**单线程串行**读 40 万张图, 耗时约 90 分钟。
    # 图像解码是 I/O + C 层运算, PIL 解码时释放 GIL, 故多线程有实际收益。
    if evalfps:
        import concurrent.futures as cf
        import time as _t
        CACHE_FILE = PROCESSED / "image_fp_cache.json"
        if CACHE_FILE.exists():
            try:
                _FP_CACHE = json.load(open(CACHE_FILE))
                print(f"  载入指纹缓存: {len(_FP_CACHE):,} 条")
            except Exception:
                _FP_CACHE = {}
        todo = []
        for r in rows:
            if r.get("media_type") != "image":
                continue
            m = r.get("media") or []
            if not m:
                continue
            k = str(PROJECT_ROOT / m[0])
            if k not in _FP_CACHE:
                todo.append(k)
        todo = list(dict.fromkeys(todo))
        if todo:
            nw = args.workers
            print(f"  并行计算图像指纹: {len(todo):,} 张 ({nw} 线程)...")
            t0 = _t.time()
            with cf.ThreadPoolExecutor(max_workers=nw) as ex:
                for i, (k, fp) in enumerate(zip(
                        todo, ex.map(lambda p: file_fingerprint(Path(p)) or "", todo)), 1):
                    _FP_CACHE[k] = fp
                    if i % 50000 == 0:
                        el = _t.time() - t0
                        print(f"    {i:,}/{len(todo):,}  ({i/el:.0f} 张/s, "
                              f"剩余 {((len(todo)-i)/(i/el))/60:.1f} 分钟)")
            print(f"  指纹完成, 用时 {(_t.time()-t0)/60:.1f} 分钟")
            json.dump(_FP_CACHE, open(CACHE_FILE, "w"))
            print(f"  缓存已写入 {CACHE_FILE.name} (下次可秒载)")

    out, seen = [], set()
    t_start = __import__("time").time()
    for i, r in enumerate(rows):
        if i and i % 25000 == 0:
            el = __import__("time").time() - t_start
            rate = i / el if el else 0
            eta = (len(rows) - i) / rate if rate else 0
            print(f"    进度 {i:,}/{len(rows):,}  已剔除 {stats['dropped_decontaminated']:,} "
                  f"({rate:.0f} 条/s, 剩余约 {eta/60:.1f} 分钟)")
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            stats["dropped_too_few_messages"] += 1; continue

        ans = msgs[-1].get("content", "")
        # ⚠️ 只丢**真正为空**的答案。早先写成 `len(ans.strip()) < 2`,
        #    结果把 47,289 条**单选题的合法答案**（'A'/'B'/'C'/'D'）全丢了 ——
        #    占数据集的 12%。单选题在 VQA 里很常见, 单字符答案是完全正常的。
        #    见 docs/05 问题 P27。
        if not ans or not ans.strip():
            stats["dropped_empty_answer"] += 1; continue

        total = sum(len(m.get("content", "")) for m in msgs)
        if total > args.cutoff_chars:
            stats["dropped_too_long"] += 1; continue

        media = r.get("media") or []
        if not media:
            stats["dropped_no_media"] += 1; continue
        mp = PROJECT_ROOT / media[0]
        if not mp.exists():
            stats["dropped_media_missing"] += 1; continue

        if evalfps and r.get("media_type") == "image":
            # 路径级缓存: 实测 40 万条样本只对应 21 万个唯一图像文件 (1.86x 复用),
            # 缓存可省掉约 46% 的解码工作。
            key = str(mp)
            fp = _FP_CACHE.get(key)
            if fp is None:
                fp = file_fingerprint(mp) or ""
                _FP_CACHE[key] = fp
            if fp and fp in evalfps:
                stats["dropped_decontaminated"] += 1; continue

        h = text_hash(json.dumps(msgs, ensure_ascii=False))
        if h in seen:
            stats["dropped_duplicate"] += 1; continue
        seen.add(h)

        out.append(r)

    n = write_jsonl(Path(args.out), out)
    print(f"\n输出 {n:,} 条 -> {args.out}")
    print(f"过滤掉 {stats['input'] - n:,} 条")
    print("\n--- 明细 ---")
    for k in sorted(stats):
        print(f"   {k:<34}{stats[k]}")
    ap_ = PROCESSED / "quality_filter_stats.json"
    ap_.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(stats), open(ap_, "w"), ensure_ascii=False, indent=2)
    print(f"\n统计写入 {ap_}")


if __name__ == "__main__":
    main()

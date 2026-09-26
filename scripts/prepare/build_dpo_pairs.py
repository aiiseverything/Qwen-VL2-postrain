#!/usr/bin/env python3
"""构建 DPO 偏好对 (目标 10,000 条)。

来源与目标配额:
    openbmb/RLHF-V-Dataset      5,733 对  (细粒度人工修正, 幻觉偏好)   ← 全部取用
    zhiqings/LLaVA-RLHF-Data    补足到 10,000 对

本机实测 schema:
    RLHF-V-Dataset:  ds_name | image(struct<bytes,path>) | text | origin_dataset
                     | origin_split | idx | image_path
                     `text` 字段内含 dict {question, chosen, rejected}
                     (dataset card 的 dtype 标为 string, 故按 str 解析并容错)

输出格式 (LLaMA-Factory DPO / sharegpt):
    {"messages":[{"role":"user","content":"<image>..."}],
     "chosen":  {"role":"assistant","content":"..."},
     "rejected":{"role":"assistant","content":"..."},
     "images":["data/processed/dpo_images/xxx.png"]}

图像以 bytes 内嵌在 parquet 中, 需落盘后再交给训练框架 (见 docs/05 问题 P8:
LLaMA-Factory 读取 DPO sharegpt 时 images 必须指向真实文件)。
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, PROCESSED, write_jsonl, text_hash  # noqa: E402

import pandas as pd  # noqa: E402

TARGET = 10_000
SEED = 42
IMG_OUT = PROCESSED / "dpo_images"
ALLOW_HEURISTIC = False      # 由 --allow-llava-rlhf-heuristic 置为 True


def _cell_bytes(cell):
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    if not isinstance(cell, (bytes, bytearray)):
        try:
            return _cell_bytes(cell[0]) if len(cell) else None
        except TypeError:
            return None
    return cell


def _parse_pref(text):
    """从 RLHF-V 的 text 字段取出 question/chosen/rejected。"""
    if text is None:
        return None
    if isinstance(text, str):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(text, dict):
        return None
    q, c, r = text.get("question"), text.get("chosen"), text.get("rejected")
    if not (q and c and r):
        return None
    return str(q).strip(), str(c).strip(), str(r).strip()


def _save_image(b, stem: str, ext=".png"):
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    p = IMG_OUT / f"{stem}{ext}"
    if not p.exists():
        p.write_bytes(b)
    # ⚠️ 必须返回**绝对路径**。LLaMA-Factory 解析 sharegpt 的 images 字段时是相对
    #    **数据集文件所在目录** 的。原先返回相对项目根的路径, 会被解析成
    #    data/processed/data/processed/dpo_images/... —— 文件不存在。
    #    (SFT 侧已修, DPO 侧漏了; 见 docs/05 问题 P28)
    return str(p.resolve())


def dedupe(rows):
    out, seen = [], set()
    for r in rows:
        h = text_hash(r["chosen"]["content"] + "||" + r["rejected"]["content"])
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def build_rlhfv(stats):
    f = RAW / "dpo/RLHF-V-Dataset/RLHF-V-Dataset.parquet"
    if not f.exists():
        stats["rlhfv_missing"] += 1; return []
    df = pd.read_parquet(f)
    rows = []
    for i, r in df.iterrows():
        parsed = _parse_pref(r.get("text"))
        if not parsed:
            stats["rlhfv_bad_text"] += 1; continue
        q, c, rj = parsed
        b = _cell_bytes(r.get("image"))
        if not b:
            stats["rlhfv_no_image"] += 1; continue
        # 用内容 hash 做稳定文件名, 避免跨源重名
        stem = text_hash(q)[:20]
        rows.append({"messages": [{"role": "user", "content": "<image>\n" + q}],
                     "chosen": {"role": "assistant", "content": c},
                     "rejected": {"role": "assistant", "content": rj},
                     "images": [_save_image(b, stem)],
                     "source": "rlhf-v"})
    stats["rlhfv_kept"] = len(rows)
    return rows


def build_llava_rlhf(stats, need: int):
    """从 LLaVA-RLHF-Data 构建偏好对。

    ⚠️ 实测该数据集**不含显式的 chosen/rejected 字段** (问题 P15):
        llava_ppo50k-aokvqa12k-vqa10k.json : {id, image, conversations,
                                              captions[], caption_type, length_bonus}
        llava_reward10k-aokvqa5k.json      : {id, image, conversations,
                                              captions[], caption_type}
       `captions` 是**多个候选回答的并列列表**, 没有标注哪个更好。
       README 也没有给出排序约定。

    **因此默认不从这里造对** —— 凭空假设「第一条最好」会污染 DPO 的偏好信号,
    而 DPO 对错误偏好标签非常敏感。改用两个明确的来源:
      1. RLHF-V 的 5,733 对 (有明确 chosen/rejected)
      2. 若仍不足, 应换用其它有显式标注的偏好数据集, 而不是猜

    只有显式传 --allow-llava-rlhf-heuristic 时才用 caption_type/length_bonus
    推导, 且会在输出与统计中标注 assumption 字段, 便于日后审计。
    """
    import glob as _glob
    import json as _json
    files = sorted(_glob.glob(str(RAW / "dpo/LLaVA-RLHF-Data/*.json")))
    if not files:
        stats["llava_rlhf_missing"] += 1; return []
    stats["llava_rlhf_files"] = [Path(f).name for f in files]

    if not ALLOW_HEURISTIC:
        stats["llava_rlhf_skipped_no_labels"] = 1
        print("   [skip] LLaVA-RLHF-Data 无显式 chosen/rejected (问题 P15)")
        print("          -> 若需强行推导, 加 --allow-llava-rlhf-heuristic (会标注 assumption)")
        return []

    # —— 仅在显式允许时才走的推导路径 (启发式, 有假设) ——
    rows = []
    for f in files:
        try:
            data = _json.load(open(f))
        except Exception:
            stats["llava_rlhf_unreadable"] += 1; continue
        if not isinstance(data, list):
            continue
        for d in data:
            convs = d.get("conversations") or []
            caps = d.get("captions") or []
            if not convs or len(caps) < 2:
                stats["llava_rlhf_bad_row"] += 1; continue
            q = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
            if not q:
                stats["llava_rlhf_bad_row"] += 1; continue
            # 假设: length_bonus 越大表示越受偏好; 缺失时退化为「较长者更优」
            key = d.get("length_bonus")
            if key is not None:
                ranked = sorted(caps, key=lambda c: -abs(len(c) - d.get("length_bonus", 0)))
            else:
                ranked = sorted(caps, key=len, reverse=True)
            rows.append({"messages": [{"role": "user", "content": "<image>\n" + q.strip()}],
                         "chosen":   {"role": "assistant", "content": ranked[0].strip()},
                         "rejected": {"role": "assistant", "content": ranked[-1].strip()},
                         "images": [],
                         "source": "llava-rlhf",
                         "assumption": "heuristic: length_bonus/length ranking — 未经人工验证"})
            if len(rows) >= need:
                break
        if len(rows) >= need:
            break
    stats["llava_rlhf_kept_heuristic"] = len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED / "dpo_10k.jsonl"))
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--allow-llava-rlhf-heuristic", action="store_true",
                    help="⚠️ 允许从无标注的 captions 列表推导偏好对 (会加 assumption 标记)")
    args = ap.parse_args()

    global ALLOW_HEURISTIC
    ALLOW_HEURISTIC = args.allow_llava_rlhf_heuristic

    stats = collections.Counter()
    rows = dedupe(build_rlhfv(stats))
    if len(rows) < args.target:
        rows += dedupe(build_llava_rlhf(stats, args.target - len(rows)))
    rows = dedupe(rows)[: args.target]

    n = write_jsonl(Path(args.out), rows)
    by = collections.Counter(r["source"] for r in rows)
    print(f"\n写出 {n:,} 条偏好对 -> {args.out}")
    for k, v in sorted(by.items()):
        print(f"   {k:<16}{v:>8,}")
    print(f"\n目标 {args.target:,}; 实际 {n:,} (缺口 {args.target - n:,})")
    print("\n--- 统计 ---")
    for k in sorted(stats):
        print(f"   {k:<34}{stats[k]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""生成数据统计报告 data/manifests/stats_report.md —— PLAN Phase 2.6 验收物。

报告内容 (PLAN §6 Phase 2 验收标准):
  * 每个来源的条数、平均长度
  * 被过滤条数及原因
  * 与评测集的图像重叠数 (去污染结果)
  * 许可与风险等级
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, DATA_ROOT, PROJECT_ROOT, load_jsonl  # noqa: E402

MANIFEST = DATA_ROOT / "manifests"
REPORT = MANIFEST / "stats_report.md"


def summarize(path: Path, label: str):
    if not path.exists():
        return None
    rows = list(load_jsonl(path))
    if not rows:
        return None
    by_src = collections.Counter(r.get("source", "?") for r in rows)
    by_kind = collections.Counter(r.get("media_type", "?") for r in rows)

    turns, chars, ans_chars = [], [], []
    for r in rows:
        msgs = r.get("messages") or []
        turns.append(len(msgs))
        chars.append(sum(len(m.get("content", "")) for m in msgs))
        if msgs:
            ans_chars.append(len(msgs[-1].get("content", "")))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0

    return {
        "label": label, "path": str(path.relative_to(PROJECT_ROOT)), "n": len(rows),
        "by_source": dict(by_src), "by_kind": dict(by_kind),
        "avg_turns": mean(turns), "avg_chars": mean(chars), "avg_answer_chars": mean(ans_chars),
        "max_chars": max(chars) if chars else 0,
    }


def load_json_if(path: Path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    MANIFEST.mkdir(parents=True, exist_ok=True)
    sections = []

    sft = summarize(PROCESSED / "sft_400k.clean.jsonl", "SFT (已过滤)")
    if sft is None:
        sft = summarize(PROCESSED / "sft_400k.jsonl", "SFT (未过滤)")
    dpo = summarize(PROCESSED / "dpo_10k.jsonl", "DPO 偏好对")

    qf = load_json_if(PROCESSED / "quality_filter_stats.json")
    sel = load_json_if(PROCESSED / "sft_selection_stats.json")

    L = []
    L.append("# Postrain 数据统计报告\n")
    L.append("> 自动生成 —— `scripts/prepare/stats.py`\n")

    # ---- SFT ----
    L.append("\n## 1. SFT 数据集\n")
    if sft:
        L.append(f"- 文件: `{sft['path']}`")
        L.append(f"- 总条数: **{sft['n']:,}**")
        L.append(f"- 平均轮数: {sft['avg_turns']:.2f} | 平均字符数: {sft['avg_chars']:.0f} "
                 f"| 平均答案字符数: {sft['avg_answer_chars']:.0f} | 最长: {sft['max_chars']:,}")
        L.append("\n| 来源 | 条数 |\n|---|---|")
        for k, v in sorted(sft["by_source"].items()):
            L.append(f"| {k} | {v:,} |")
        L.append("\n| 模态 | 条数 |\n|---|---|")
        for k, v in sorted(sft["by_kind"].items()):
            L.append(f"| {k} | {v:,} |")
        target = 400_000
        L.append(f"\n**配额达成率: {sft['n']:,} / {target:,} = "
                 f"{100*sft['n']/target:.1f}%**")
    else:
        L.append("_尚未生成 SFT 数据_")

    # ---- DPO ----
    L.append("\n## 2. DPO 偏好对\n")
    if dpo:
        L.append(f"- 文件: `{dpo['path']}`")
        L.append(f"- 总条数: **{dpo['n']:,}** (目标 10,000)")
        L.append(f"- 平均字符数: {dpo['avg_chars']:.0f}")
        L.append("\n| 来源 | 条数 |\n|---|---|")
        for k, v in sorted(dpo["by_source"].items()):
            L.append(f"| {k} | {v:,} |")
    else:
        L.append("_尚未生成 DPO 数据_")

    # ---- RLVR ----
    L.append("\n## 3. RLVR 数据\n")
    rlvr_dir = PROCESSED / "rlvr"
    if rlvr_dir.exists():
        L.append("| 数据集 | 文件 |\n|---|---|")
        for p in sorted(rlvr_dir.glob("*.parquet")):
            try:
                import pyarrow.parquet as pq
                n = pq.ParquetFile(p).metadata.num_rows
                L.append(f"| {p.stem} | `{p.name}` ({n:,} 行) |")
            except Exception:
                L.append(f"| {p.stem} | `{p.name}` |")
    else:
        L.append("_尚未生成 RLVR 数据_")

    # ---- 过滤原因 ----
    L.append("\n## 4. 过滤与丢弃原因\n")
    if qf:
        L.append("### 4.1 质量过滤 / 去污染 (`quality_filter.py`)\n")
        L.append("| 原因 | 条数 |\n|---|---|")
        for k, v in sorted(qf.items()):
            L.append(f"| {k} | {v:,} |")
        dec = qf.get("dropped_decontaminated", 0)
        L.append(f"\n> **去污染披露 (PLAN K5)**: 训练集中与 MME/MathVista 评测图像重合、"
                 f"已被剔除的样本共 **{dec:,}** 条。")
        if not qf.get("eval_fingerprints"):
            L.append("> ⚠️ 本次运行**未收集到评测图像指纹**, 去污染实际未生效 —— "
                     "结论中不得声称已去污染。")
    else:
        L.append("_质量过滤尚未运行_")
    if sel:
        L.append("\n### 4.2 抽取阶段丢弃 (`to_sharegpt.py`)\n")
        L.append("| 原因 | 数量 |\n|---|---|")
        for k, v in sorted(sel.items()):
            if isinstance(v, (int, float)):
                L.append(f"| {k} | {v:,} |")
            else:
                L.append(f"| {k} | `{json.dumps(v, ensure_ascii=False)}` |")

    # ---- 许可 ----
    L.append("\n## 5. 许可与风险\n")
    L.append("| 数据集 | 许可 | 风险 |\n|---|---|---|")
    L.append("| liuhaotian/LLaVA-Instruct-150K | CC-BY-4.0 (受 OpenAI 条款约束) | medium |")
    L.append("| Lin-Chen/ShareGPT4V | **CC-BY-NC-4.0** | **HIGH (禁商用)** |")
    L.append("| MBZUAI/VideoInstruct-100K | CC-BY-SA-4.0 | medium |")
    L.append("| openbmb/RLHF-V-Dataset | **CC-BY-NC-4.0** | **HIGH (禁商用)** |")
    L.append("| zhiqings/LLaVA-RLHF-Data | **CC-BY-NC-4.0** | **HIGH (禁商用)** |")
    L.append("| hiyouga/geometry3k | MIT | low |")
    L.append("| leonardPKU/GEOQA_R1V_Train_8K | apache-2.0 | low |")
    L.append("| FanqingM/MMK12 | apache-2.0 | low |")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"报告已写入 {REPORT}")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()

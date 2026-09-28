#!/usr/bin/env python3
"""汇总评测结果 -> results/report.md + results/report.csv

PLAN §9.3 要求报告必须包含:
  * 模型 × benchmark 对比表
  * 相对基线的增减
  * **判分方式标注**          ← K6: MathVista 无 key 时是规则判分, 不可与论文比较
  * **数据污染检查结果**       ← K5: 未去污染就必须声明, 不能默认干净
  * 未达预期项的诚实说明

⚠️ 这个脚本刻意**不隐藏**任何不利信息。一份"好看但没标注判分方式和污染情况"的
   报告, 在方法学上是无效的。
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
EVAL_DIR = PROJECT_ROOT / "results/eval"
OUT_MD = PROJECT_ROOT / "results/report.md"
OUT_CSV = PROJECT_ROOT / "results/report.csv"

MODELS = ["pretrained", "sft", "dpo", "grpo"]
BENCH = ["MME", "MathVista", "Video-MME"]

# 判分方式: 由环境决定, 不猜测
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"))


def _latest_run_dir(bench: str, model: str):
    """VLMEvalKit 的实际落盘结构是 results/eval/<bench>/<model>/T<时间戳>/

    ★ 2026-09-28 修: 原来是 EVAL_DIR/<model>/<bench> —— **路径写反了**,
    d.is_dir() 恒为 False, 于是所有分数都收集不到, 报告里全是「—」。
    （MME 其实三个模型都跑成功了, 见 results/eval/MME/<model>/T*/ 里的 score.csv）
    """
    base = EVAL_DIR / bench / model
    if not base.is_dir():
        return None
    runs = sorted([p for p in base.glob("T*") if p.is_dir()], reverse=True)
    return runs[0] if runs else base


def read_score(model: str, bench: str):
    """返回 (主分数, 来源文件, 明细说明)。找不到返回 (None, "", "")

    ★ MME 的 VLMEvalKit 产物 <model>_MME_score.csv 是**纯数字表**
    （表头是类别名, 数据行是对应分数, 没有任何 acc/score 字样）,
    原来那条靠关键词匹配的通用解析认不出来 —— 现在按类别名精确取。
    """
    d = _latest_run_dir(bench, model)
    if d is None:
        return None, "", ""

    if bench == "MME":
        for f in sorted(d.glob("*_MME_score.csv")):
            try:
                with open(f, newline="") as fh:
                    rdr = csv.reader(fh)
                    head = next(rdr, [])
                    vals = next(rdr, [])
            except (OSError, StopIteration, csv.Error):
                continue
            got = {}
            for k, v in zip(head, vals):
                try:
                    got[k.strip().strip('"').lower()] = float(v)
                except (TypeError, ValueError):
                    pass
            # 官方 MME 报两个聚合分: perception(10 类, 满分 2000) / reasoning(4 类, 满分 800)
            if "perception" in got:
                detail = (f"perception {got['perception']:.2f}/2000"
                          + (f"; reasoning {got['reasoning']:.2f}/800" if "reasoning" in got else ""))
                return got["perception"], f.name, detail
        return None, "", ""

    # 其余 benchmark: 沿用关键词解析 (带 acc/accuracy/score 字样的产物)
    for pat in ("*.csv", "*.json", "*.txt"):
        for f in sorted(d.glob(pat)):
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            m = re.search(r"(?:acc|accuracy|score)[^0-9]*([0-9]+\.[0-9]+)", txt, re.I)
            if m:
                return float(m.group(1)), f.name, ""
    return None, "", ""


def contamination_note() -> str:
    p = PROJECT_ROOT / "data/processed/quality_filter_stats.json"
    if not p.exists():
        return ("⚠️ **未运行去污染**：`quality_filter_stats.json` 不存在。"
                "**本报告的分数不得声称已去污染。**")
    s = json.load(open(p))
    n = s.get("dropped_decontaminated", 0)
    fp = s.get("eval_fingerprints", 0)
    if not fp:
        return ("⚠️ **去污染未生效**：收集到的评测集图像指纹为 0，"
                "去污染实际没有执行。**不得声称已去污染。**")
    return (f"✅ 已执行去污染：收集评测集图像指纹 {fp:,} 张，"
            f"从训练集剔除重叠样本 **{n:,}** 条。")


def judging_note(bench: str) -> str:
    if bench == "MathVista":
        if HAS_OPENAI_KEY:
            return "官方协议（GPT 抽取判分）"
        return ("⚠️ **规则抽取判分（非官方）** —— 无 OpenAI API key，"
                "**分数不可与论文数字直接比较**（PLAN K6）")
    if bench == "MME":
        return "官方协议（是/否二元题，accuracy + accuracy+）"
    if bench == "Video-MME":
        return "VLMEvalKit 默认抽帧；⚠️ 四个模型必须用**同一抽帧策略**"
    return "见 VLMEvalKit 默认协议"


def main():
    rows = []
    for m in MODELS:
        for b in BENCH:
            score, src, detail = read_score(m, b)
            rows.append({"model": m, "bench": b, "score": score,
                         "source_file": src, "detail": detail,
                         "judging": judging_note(b)})

    # 相对基线
    base = {r["bench"]: r["score"] for r in rows if r["model"] == "pretrained"}
    for r in rows:
        bs = base.get(r["bench"])
        r["delta_vs_base"] = (None if (r["score"] is None or bs is None)
                              else round(r["score"] - bs, 3))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "bench", "score", "delta_vs_base",
                                          "detail", "judging", "source_file"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    L = ["# Postrain 评测报告\n",
         "> 自动生成：`scripts/eval/collect_results.py`\n",
         "\n## 1. 模型 × Benchmark 对比\n",
         "| 模型 | " + " | ".join(BENCH) + " |",
         "|---|" + "---|" * len(BENCH)]
    for m in MODELS:
        cells = []
        for b in BENCH:
            r = next(x for x in rows if x["model"] == m and x["bench"] == b)
            if r["score"] is None:
                cells.append("—")
            else:
                d = r["delta_vs_base"]
                cells.append(f"{r['score']:.3f}" + (f" ({d:+.3f})" if d is not None else ""))
        L.append(f"| **{m}** | " + " | ".join(cells) + " |")

    L += ["\n> 括号内为相对 `pretrained` 基线的增减。\n"]
    # 有明细的 (如 MME 的 perception/reasoning 双聚合分) 单独列出来, 免得丢信息
    details = [f"- **{r['model']} · {r['bench']}**：{r['detail']}"
               for r in rows if r["detail"]]
    if details:
        L += ["", "实测明细（聚合分口径）：", *details, ""]

    L += ["\n## 2. 判分方式（必须逐项标注）\n",
          "| Benchmark | 判分方式 |", "|---|---|"]
    for b in BENCH:
        L.append(f"| {b} | {judging_note(b)} |")

    L += ["\n## 3. 数据污染检查\n", contamination_note(),

          "\n## 4. 未达预期项的说明\n",
          "_（本节省略 = 未如实评估。请逐条填写：哪些指标没涨、为什么、"
          "是数据问题还是方法问题）_\n",
          "- 例：**Video-MME 未提升** —— SFT 训练数据中视频占比低（受 YouTube "
          "可用性限制），预期视频理解提升有限。这是数据决定的，不是训练失败。\n",

          "\n## 5. 复现信息\n",
          f"- 训练数据: `data/processed/sft_400k.clean.jsonl`",
          f"- 评测工具: VLMEvalKit（四个模型统一）",
          f"- 原始输出: `results/eval/<bench>/<model>/T<时间戳>/`"]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"报告 -> {OUT_MD}")
    print(f"CSV  -> {OUT_CSV}")
    for r in rows:
        print(f"  {r['model']:<11}{r['bench']:<12}{r['score']}")


if __name__ == "__main__":
    main()

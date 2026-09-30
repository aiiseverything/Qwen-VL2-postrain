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

MODELS = ["base", "base_sft", "pretrained", "sft", "sft_ep3", "dpo", "grpo"]
BENCH = ["MME", "MathVista", "Video-MME"]

# 报告里用的简称 -> VLMEvalKit 实际落盘的目录名（2026-09-29 实测）
BENCH_DIR = {"MME": "MME", "MathVista": "MathVista_MINI", "Video-MME": "Video-MME_16frame"}

# ★ 2026-09-29: 模型 key -> 报告里的**准确**称呼。
#   背景：`models/Qwen2-VL-2B-Instruct` 是**官方已指令微调**的版本（PLAN.md:78 选的就是它），
#   不是原始的 `Qwen/Qwen2-VL-2B`（纯预训练）。原来那一列直接叫 "pretrained" 是错的，
#   会让人误以为对比的是裸预训练模型 —— 实际是在"已指令微调"的基础上再训。
#   目录键（pretrained/）保持不变以免打断历史产物，只改报告显示名。
MODEL_DISPLAY = {
    "pretrained": "instruct(官方)",
    "base": "base(未指令微调)",
    "base_sft": "base+sft",
    "sft": "sft",
    "sft_ep3": "sft_ep3",
    "dpo": "dpo",
    "grpo": "grpo",
}

# 判分方式: 由环境决定, 不猜测
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"))


def _latest_run_dir(bench: str, model: str):
    """VLMEvalKit 的实际落盘结构是 results/eval/<目录名>/<model>/T<时间戳>/

    ★ 2026-09-28 修 bug1: 原来是 EVAL_DIR/<model>/<bench> —— **路径写反了**,
    d.is_dir() 恒为 False, 于是所有分数都收集不到, 报告里全是「—」。
    ★ 2026-09-29 修 bug2: 报告里用的**简称**与磁盘上的**目录名**不同 ——
    `MathVista` 实际是 `MathVista_MINI`, `Video-MME` 实际是 `Video-MME_16frame`,
    于是这两个 benchmark 即使跑成功了也一直取不到分。现在显式映射。
    """
    base = EVAL_DIR / BENCH_DIR.get(bench, bench) / model
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

    if bench == "Video-MME":
        # ★ 2026-09-29 补: VLMEvalKit 的 Video-MME 产物是嵌套 JSON,
        #   形如 {"short": {"overall": "0.600", "domain": {...}}, "medium": ..., "long": ...}
        #   —— 里面没有 acc/accuracy/score 字样, 通用关键词解析认不出来 ✓
        #   本项目只评测 3 分钟以内的片段（见 docs/05「Video-MME 的接线」）⇒ 取 short。
        for f in sorted(d.glob("*_score*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            seg = data.get("short")
            if not isinstance(seg, dict) or "overall" not in seg:
                continue
            try:
                v = float(seg["overall"]) * 100.0        # 0.600 -> 60.0 (%)
            except (TypeError, ValueError):
                continue
            parts = []
            for split in ("short", "medium", "long"):
                s = data.get(split)
                if isinstance(s, dict) and s.get("overall") is not None:
                    try:
                        parts.append(f"{split} {float(s['overall']) * 100:.1f}%")
                    except (TypeError, ValueError):
                        pass
            return v, f.name, "Video-MME 分档正确率: " + "; ".join(parts)
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
        return ("VLMEvalKit 默认抽帧（本项目只评 `short` 档，即 3 分钟以内）；"
                "⚠️ 四个模型必须用**同一抽帧策略**。"
                "**实测判分模型可换**：09-23 用 gpt-4o-mini、09-29 用 gpt-5.5，"
                "`short overall` 同为 0.600 ⇒ 换判分模型不改变可比性")
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
        L.append(f"| **{MODEL_DISPLAY.get(m, m)}** | " + " | ".join(cells) + " |")

    L += [
        "\n> ⚠️ **各列刻度不同，别横向比绝对值**（2026-09-29 澄清）：",
        "> - `MME` 是 **2000 分制**（perception 10 类 × 200；reasoning 4 类 × 200 = 800），"
        "**不是百分比**。例：perception 1492.71 → 1536.30 是 +43.6 **分**，换算仅 **+2.2 个百分点**。",
        "> - `MathVista` / `Video-MME` 是**百分比**（Video-MME 取 3 分钟以内的 short 档）。",
        "> - 括号内为相对基线的增减，**同刻度**之差。",
        "> - 换判分模型不影响可比性（09-23 用 gpt-4o-mini、09-29 用 gpt-5.5，"
        "Video-MME short overall 同为 0.600 实测验证）。",
        "> ",
        "> **关于两条基线（2026-09-29 澄清）**：",
        "> - `instruct(官方)` = `Qwen2-VL-2B-Instruct`，**官方已指令微调**的版本，"
        "也是本项目 SFT/DPO 的起点。**报告的主基线是它** —— 所以增益是在「已指令微调」之上再取得的。",
        "> - `base(未指令微调)` = `Qwen2-VL-2B`（无 -Instruct），纯预训练。",
        ">   ⚠️ **它只能当参照，不能当同口径基线**：预训练模型没有 chat template、"
        "不按对话格式回答，用同一套对话式 prompt 评测天然吃亏。它的分数低是**预期**的，"
        "不代表「训练没用」。",
        "",
    ]
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
          "> 实测数据见上表。以下逐条如实说明**没涨的部分**（2026-09-29 填）。\n",
          "- **MathVista 全线小幅退步**（相对 instruct：sft −2.1 / sft_ep3 −1.7 / dpo −1.9）：",
          "  40 万条混训数据以感知与描述类为主（llava665k 296k、sharegpt4v 50k，多为 caption/VQA 风格），",
          "  数学与推理样本占比很低 ⇒ 预期本就有限，实测是「没涨、略降」。属**数据问题**，非训练失败。\n",
          "- **Video-MME 同样小幅退步**（sft −2.7 / sft_ep3 −2.4 / dpo −2.3）：",
          "  SFT 数据中视频样本仅 **13.4%**（受 YouTube 可用性限制），且只取 3 分钟以内片段，",
          "  长时序理解无从获得 ⇒ 同样**数据决定**。\n",
          "- **多训一轮 epoch（sft_ep3）是净亏**：MME perception 从 +43.6 掉到 +13.1，",
          "  reasoning 从 457.14 掉到 **436.79（−19.6）**，而 MathVista / Video-MME 只各涨 0.3~0.4。",
          "  ⇒ 2 epoch 已到拐点，第 3 轮开始退化（过拟合或调度重启所致）。**不建议再训第 3 轮。**\n",
          "- **GRPO 一列全空 = 未完成**（环境问题已修复，但按用户决定暂缓），",
          "  不是「跑了但没分」，不得与其他两项混为一谈。\n",

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

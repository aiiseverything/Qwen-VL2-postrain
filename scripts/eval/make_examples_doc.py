#!/usr/bin/env python
"""把 MME 的逐题预测做成图文对照文档（PDF + Markdown）。

为什么单独做这个: 汇总分数（MME 1492.71 → 1536.30）看不出"模型到底怎么变的"。
把**图片 + 问题 + 标准答案 + 四个模型的回答**摆在一起，才能判断：
是感知真的变强了，还是只是学会了评测集的作答风格。

选例策略（自动 + 少量人工指定）:
  * 进步: instruct 错、sft+sft_ep3+dpo 都对
  * 退步: instruct 对、三个训练后模型都错
  * 仅格式: 归一化后答案相同，但原始字符串不同（带句点/大小写）
  * 人工指定: 标准答案存疑的两条（见 DUBIOUS）

用法:
    .venv-eval/bin/python scripts/eval/make_examples_doc.py   # 需要 openpyxl（.venv-eval 有） \
        --out-dir docs/exp --name MME-例子-20260929
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = ["pretrained", "sft", "sft_ep3", "dpo"]
DISPLAY = {"pretrained": "instruct(官方)", "sft": "sft(2ep)", "sft_ep3": "sft_ep3", "dpo": "dpo"}
IMG_ROOT = PROJECT_ROOT / "data/raw/lmudata/images/MME"
FONT_CANDIDATES = [
    Path("/home/ml-user/workdir/tools/fonts/NotoSansSC-Regular.otf"),
    Path("/tmp/NotoSansSC-Regular.otf"),
]
FONT_URL = ("https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/SubsetOTF/SC/"
            "NotoSansSC-Regular.otf")

# 标准答案人工存疑的两条（判断理由见正文注释）—— 按问题片段匹配
DUBIOUS = ["directed by john hillcoat", "If I eat an apple every day"]


def ensure_font() -> Path | None:
    for p in FONT_CANDIDATES:
        if p.exists():
            return p
    dst = FONT_CANDIDATES[0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(FONT_URL, dst)
        return dst if dst.exists() else None
    except Exception as e:
        print(f"⚠️ 中文字体下载失败({e})，PDF 将退化为英文渲染")
        return None


def norm(s) -> str:
    return re.sub(r"[\s\.\!\。\，\,]+$", "", str(s).strip().lower())


def load() -> pd.DataFrame:
    d = {}
    for m in MODELS:
        f = sorted(glob.glob(str(PROJECT_ROOT / f"results/eval/MME/{m}/T*/{m}_MME.xlsx")))
        if not f:
            raise SystemExit(f"缺 {m} 的 MME 预测文件")
        d[m] = pd.read_excel(f[-1]).set_index("index")
    df = d["pretrained"][["category", "image_path", "question", "answer"]].copy()
    for m in MODELS:
        df[DISPLAY[m]] = d[m]["prediction"].astype(str)
    df["img"] = df["image_path"].map(lambda x: IMG_ROOT / x)
    df["ok_img"] = df["img"].map(os.path.exists)
    df["ans_n"] = df["answer"].map(norm)
    for m in MODELS:
        df[DISPLAY[m] + "_n"] = df[DISPLAY[m]].map(norm)
        df[DISPLAY[m] + "_ok"] = df[DISPLAY[m] + "_n"] == df["ans_n"]
    return df


def pick(df: pd.DataFrame, n_each: int = 4) -> list[tuple[str, str]]:
    """返回 [(标题, index)]"""
    I, S, E, D = [DISPLAY[m] for m in MODELS]
    q = df[df.ok_img]
    out: list[tuple[str, str]] = []
    up = q[(~q[I + "_ok"]) & q[S + "_ok"] & q[E + "_ok"] & q[D + "_ok"]]
    dn = q[(q[I + "_ok"]) & (~q[S + "_ok"]) & (~q[E + "_ok"]) & (~q[D + "_ok"])]
    fmt = q[(q[I + "_n"] == q[S + "_n"]) & (q[I] != q[S])]        # 内容同、格式不同
    for title, sub in (("进步", up), ("退步", dn), ("仅格式差异", fmt)):
        for i in sub.sample(min(n_each, len(sub)), random_state=11).index:
            out.append((title, i))
    for frag in DUBIOUS:
        hit = q[q["question"].str.contains(re.escape(frag), case=False, na=False)]
        for i in hit.index[:1]:
            out.append(("标准答案存疑", i))
    return out


def page(pdf: PdfPages, row, title: str, idx, cjk: bool) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 纵向
    fig.suptitle(f"[{title}] {row['category']}  ·  index={idx}", fontsize=13, y=0.975)

    # ---- 图片 ----
    ax = fig.add_axes([0.08, 0.53, 0.84, 0.40])
    ax.imshow(plt.imread(str(row["img"])))
    ax.axis("off")

    # ---- 文字 ----
    txt = fig.add_axes([0.08, 0.05, 0.84, 0.45]); txt.axis("off")
    y = 1.0
    def line(s, color="black", size=10, bold=False, indent=0.0):
        nonlocal y
        wrapped = textwrap.wrap(str(s), width=78) or [""]
        for k, w in enumerate(wrapped):
            txt.text(indent, y, w, fontsize=size, color=color, va="top",
                     fontweight="bold" if bold else "normal")
            y -= 0.043 if size >= 11 else 0.038
        y -= 0.012

    line(f"Q: {row['question']}", size=10.5)
    line(f"标准答案 (ground truth): {row['answer']}", color="#1a7f37", bold=True)
    y -= 0.01
    for m in MODELS:
        key = DISPLAY[m]
        ok = bool(row[key + "_ok"])
        line(f"{key:14s} → {str(row[key])[:40]:40s} {'[对]' if ok else '[错]'}",
             color="#1a7f37" if ok else "#b3261e", size=10)
    if title == "标准答案存疑":
        y -= 0.02
        line("⚠️ 人工判断：该题的标准答案本身可疑（见文档说明）。", color="#b26a00", size=9)
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf: PdfPages, df: pd.DataFrame, picks) -> None:
    I, S, E, D = [DISPLAY[m] for m in MODELS]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("MME 逐题分析摘要（2026-09-29）", fontsize=15, y=0.96)
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.85]); ax.axis("off")
    y = 1.0
    def L(s="", size=10.5, color="black", bold=False, dy=None):
        nonlocal y
        for w in (textwrap.wrap(str(s), width=76) or [""]):
            ax.text(0, y, w, fontsize=size, color=color, va="top",
                    fontweight="bold" if bold else "normal")
            y -= 0.030 if size <= 10.5 else 0.038
        y -= 0.010 if dy is None else dy

    up = ((~df[I + "_ok"]) & df[S + "_ok"] & df[D + "_ok"]).sum()
    dn = ((df[I + "_ok"]) & (~df[S + "_ok"]) & (~df[D + "_ok"])).sum()
    L("一、分数（同刻度对照）", bold=True)
    L("MME perception 是 2000 分制，不是百分比：1492.71 → 1536.30 = +43.6 分 = 仅 +2.2 个百分点")
    L("另两项是百分比：MathVista 49.2% → 47.1%；Video-MME(short) 60.0% → 57.3%")
    L()
    L("二、逐题变化（统一归一化口径）", bold=True)
    L(f"instruct 错 → 训练后全对：{up} 条")
    L(f"instruct 对 → 训练后全错：{dn} 条")
    L(f"净变化：{up - dn:+d} 条 ⇒ MME 上的\"提升\"并不稳健")
    L()
    L("三、作答风格的变化（这可能是分数变化的主因）", bold=True)
    L(f"标准答案 Yes/No = {int((df['ans_n']=='yes').sum())} / {int((df['ans_n']=='no').sum())}（完美平衡）")
    for m in MODELS:
        v = df.loc[df["ans_n"].isin(["yes", "no"]), DISPLAY[m] + "_n"]
        L(f"  {DISPLAY[m]:14s} 答 Yes {100*(v=='yes').mean():5.1f}%   答 No {100*(v=='no').mean():5.1f}%")
    L("⇒ instruct 偏 Yes(54%)且 2.6% 是带解释的长答；训练后模型几乎严格 50/50、只答 yes/no")
    L()
    L("四、结论", bold=True)
    L("MME 的 +2.2pp 很可能主要来自\"学会了评测集的作答风格\"，而非感知能力提升；")
    L("这与 MathVista/Video-MME（LLM 判分、对格式不敏感）一致小幅退步的现象吻合。")
    L("报告里应写成：后训练在这份配方下 MME 基本持平、另两项小幅下降。")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/exp")
    ap.add_argument("--name", default="MME-例子")
    args = ap.parse_args()

    fp = ensure_font()
    if fp:
        matplotlib.font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.sans-serif"] = [
            matplotlib.font_manager.FontProperties(fname=str(fp)).get_name(),
            "DejaVu Sans",   # 回退: Noto Sans SC 缺 ✓/✗/→ 这些符号字形
        ]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"✓ 中文字体: {fp}")

    df = load()
    picks = pick(df)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.name}.pdf"
    md_path = out_dir / f"{args.name}.md"

    with PdfPages(pdf_path) as pdf:
        summary_page(pdf, df, picks)
        for title, idx in picks:
            page(pdf, df.loc[idx], title, idx, cjk=bool(fp))
    print(f"✓ PDF  -> {pdf_path}  ({pdf_path.stat().st_size//1024} KB, {len(picks)+1} 页)")

    # ---- markdown 版（图用相对路径，便于在 IDE 里直接看/改）----
    rel = os.path.relpath(IMG_ROOT, out_dir)
    md = [f"# MME 逐题对照（{args.name}）\n",
          "> 由 `scripts/eval/make_examples_doc.py` 生成。图片相对路径指向 "
          f"`{rel}/<category>/<file>`。\n",
          "## 摘要\n",
          f"- 训练后 **instruct 错→全对**: {int(((~df['instruct(官方)_ok']) & df['sft(2ep)_ok'] & df['dpo_ok']).sum())} 条；"
          f"**instruct 对→全错**: {int((df['instruct(官方)_ok'] & (~df['sft(2ep)_ok']) & (~df['dpo_ok'])).sum())} 条\n",
          "- ⚠️ MME perception 是 **2000 分制**：1492.71→1536.30 是 +43.6 分 = **仅 +2.2 个百分点**\n",
          "- 标准答案 Yes/No = 1187/1187（完美平衡）；instruct 答 Yes 54.3%，训练后 ~50%\n",
          "\n---\n"]
    for k, (title, idx) in enumerate(picks, 1):
        row = df.loc[idx]
        md += [f"\n## {k}. [{title}] {row['category']}（index={idx}）\n",
               f"\n![{idx}]({rel}/{row['image_path']})\n",
               f"\n**问**：{row['question']}\n",
               f"\n**标准答案**：{row['answer']}\n",
               "\n| 模型 | 回答 | 对错 |", "|---|---|---|"]
        for m in MODELS:
            key = DISPLAY[m]
            md.append(f"| {key} | `{str(row[key])[:70]}` | {'✓' if row[key+'_ok'] else '✗'} |")
        if title == "标准答案存疑":
            md.append("\n> ⚠️ 人工判断：本题标准答案可疑，需人工复核。")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ MD   -> {md_path}")
    print("\n选中示例:")
    for title, idx in picks:
        print(f"  [{title}] {df.loc[idx, 'category']} index={idx}")


if __name__ == "__main__":
    main()

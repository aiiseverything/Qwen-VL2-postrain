"""RLVR 奖励函数 —— 供 EasyR1/verl GRPO 调用。

设计要点
--------
1. **答案精确匹配为主** (PLAN §7.3)。先把模型输出与参考答案都规整到可比较的
   规范形式, 再做精确匹配; 匹配不上再退化为数值容差比较。

2. ⚠️ 答案格式在源数据里**不统一** (本机实测):
       hiyouga/geometry3k            answer = "3"                (裸答案)
       leonardPKU/GEOQA_R1V_Train_8K answer = "<answer> 145° </answer>"  (带标签)
   若只做朴素字符串相等, geoqa 的 8,031 条会**全部判错**, 奖励恒为 0,
   GRPO 完全学不到东西却看不出报错 —— 这是最危险的静默失败 (docs/05 问题 P6)。
   故本函数先剥离 <answer>/<solution> 等标签再比较。

3. **格式奖励默认关闭** (PLAN K8): 在 Qwen2-VL-2B 这种小模型上强制
   <think></think> CoT 格式会掉分 (R1-V 作者自述)。保留为消融开关,
   通过 reward_function_kwargs.format_reward 打开。
"""
from __future__ import annotations

import math
import re

# <answer> ... </answer> / <solution> ... </solution> 等标签
_TAG = re.compile(r"<\s*(answer|solution|final)\s*>(.*?)<\s*/\s*\1\s*>",
                  re.IGNORECASE | re.DOTALL)
_THINK = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")
# 提取数值 (允许负号、小数、千分位、百分号)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _strip_tags(s: str) -> str:
    s = _THINK.sub(" ", s or "")
    m = _TAG.search(s)
    if m:
        return m.group(2)
    return s


def normalize(s: str) -> str:
    """规整到规范形式, 用于精确匹配。"""
    s = _strip_tags(s)
    s = s.strip().lower()
    s = s.rstrip(".。")                      # 去掉句末标点
    s = s.replace("×", "*").replace("÷", "/")
    s = _WS.sub(" ", s)
    return s.strip()


def extract_number(s: str) -> float | None:
    s = _strip_tags(s).replace(",", "")
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def answer_reward(pred: str, gold: str, tol: float = 1e-4) -> float:
    """参考答案与预测的匹配奖励。"""
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    if p == g:                                     # 1) 规整后精确匹配
        return 1.0
    # 2) 数值容差匹配 (处理 "145°" vs "145", "1/2" vs "0.5" 之外的情形)
    pn, gn = extract_number(pred), extract_number(gold)
    if pn is not None and gn is not None:
        if math.isclose(pn, gn, rel_tol=tol, abs_tol=tol):
            return 1.0
    # 3) 预测中包含标准答案 (模型多说了话但答案正确)
    if g and g in p:
        return 1.0
    return 0.0


def format_reward(pred: str) -> float:
    """可选格式奖励: 要求 思考过程 + <answer> 包裹。默认权重为 0 (K8)。"""
    if not pred:
        return 0.0
    has_think = bool(re.search(r"<\s*think\s*>", pred, re.IGNORECASE))
    has_answer = bool(_TAG.search(pred))
    return 0.5 * has_think + 0.5 * has_answer


def compute_score(predictions, ground_truths=None, **kwargs):
    """EasyR1 奖励入口。

    兼容两种调用约定:
      * compute_score(solution_str, ground_truth) -> float      (单条)
      * compute_score(list_of_str, list_of_gt)     -> list[float]
    """
    w_fmt = float(kwargs.get("format_reward", 0.0))     # ★ K8 默认 0
    w_ans = float(kwargs.get("answer_reward", 1.0))

    single = isinstance(predictions, str)
    preds = [predictions] if single else list(predictions)

    if ground_truths is None:
        gts = [None] * len(preds)
    elif isinstance(ground_truths, str):
        gts = [ground_truths] * len(preds)
    else:
        gts = list(ground_truths)
        if len(gts) == 1 and len(preds) > 1:
            gts = gts * len(preds)

    scores = []
    for p, g in zip(preds, gts):
        s = w_ans * answer_reward(p, g if g is not None else "")
        if w_fmt:
            s += w_fmt * format_reward(p)
        scores.append(s)

    return scores[0] if single else scores

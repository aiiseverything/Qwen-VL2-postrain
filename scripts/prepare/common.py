"""Postrain 数据准备的公共工具。

设计约束 (用户要求):
  * 不加载 torch / 不碰 CUDA —— GRPO 的数据准备完全是 CPU 文本工作。
  * 不写入 project-1/ —— 所有路径经 resolve_root() 限制在 PROJECT_ROOT 内。

统一格式 (LLaMA-Factory "sharegpt" 多模态格式):
    {
      "messages": [{"role": "user", "content": "<image>..."},
                   {"role": "assistant", "content": "..."}],
      "images":   ["coco/train2017/000000033471.jpg"]
    }
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ml-user/workdir/project-2"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"
SAMPLES = DATA_ROOT / "samples"
IMG_ROOT = RAW / "images"

# LLaVA / ShareGPT4V 标注里的 image 前缀 -> 解压后图像所在的子目录
# 实测前缀分布见 docs/05。仅保留**本机确实能拿到图像**的前缀。
IMAGE_PREFIX_TO_DIR = {
    "coco": "coco/train2017",
    "gqa": "gqa/images",
    "textvqa": "textvqa/train_images",
    "vg": "vg",           # vg/VG_100K 与 vg/VG_100K_2
    "sam": "sam/images",  # 需 ShareGPT4V 自建图像包; 未下载时样本会被过滤
}

# 被排除的 #前缀 及原因 (进入 stats 报告, 让过滤可解释)
EXCLUDED_PREFIX_REASON = {
    "ocr_vqa": "上游 OCR-VQA 约 250GB 且需逐图匹配, 成本过高 —— 本机 400K 配方不含",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows, limit: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    return n


def resolve_image(rel: str) -> Path | None:
    """把标注里的相对图像路径解析为本机绝对路径; 无法映射则返回 None。"""
    if not rel:
        return None
    rel = rel.replace("\\", "/").lstrip("./")
    prefix = rel.split("/", 1)[0]
    if prefix in EXCLUDED_PREFIX_REASON:
        return None
    sub = IMAGE_PREFIX_TO_DIR.get(prefix)
    if sub is None:
        return None
    # vg 需要区分 VG_100K / VG_100K_2 —— 保留原始子路径
    if prefix == "vg":
        tail = rel.split("/", 1)[1] if "/" in rel else ""
        return IMG_ROOT / "vg" / tail
    fname = rel.rsplit("/", 1)[-1]
    return IMG_ROOT / sub / fname


def to_sharegpt(convs: list[dict]) -> list[dict] | None:
    """LLaVA 风格 conversations -> LLaMA-Factory messages。首个 human 前补 <image>。"""
    if not convs:
        return None
    msgs, first = [], True
    for c in convs:
        role = {"human": "user", "gpt": "assistant"}.get(c.get("from"))
        val = (c.get("value") or "").strip()
        if role is None or not val:
            return None
        if first and role == "user" and "<image>" not in val:
            val = "<image>\n" + val
        first = False
        msgs.append({"role": role, "content": val})
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    return msgs


_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).lower()


def text_hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()


def media_hash(path: Path) -> str | None:
    """按文件名+大小做轻量指纹 (不读全图, 避免 I/O 抢占训练)。"""
    try:
        st = path.stat()
    except OSError:
        return None
    import hashlib
    return hashlib.sha1(f"{path.name}:{st.st_size}".encode()).hexdigest()[:16]

#!/usr/bin/env python
"""直接调 LLaMA-Factory 的**真实** Qwen2VLPlugin, 分别测视频/图像两条路径是否泄漏。

用法: leak_test2.py <video|image> [条数]
"""
import sys, glob, json, os

sys.path.insert(0, os.path.join(os.getcwd(), "third_party/LlamaFactory/src"))
from llamafactory.data.mm_plugin import Qwen2VLPlugin  # noqa: E402

mode = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


plugin = Qwen2VLPlugin(
    image_token="<|vision_start|><|image_pad|><|vision_end|>",
    video_token="<|vision_start|><|video_pad|><|vision_end|>",
    audio_token=None,
)
kwargs = dict(
    video_fps=2.0, video_maxlen=32, video_max_pixels=16384,
    image_max_pixels=262144, image_min_pixels=1024,
)

if mode == "video":
    items = sorted(glob.glob("data/raw/videos/**/*.mp4", recursive=True))[:n]
else:
    # 从真实训练数据里取图片路径
    items = []
    with open("data/processed/sft_400k.clean.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("media_type") == "image" and d.get("images"):
                p = d["images"][0]
                if os.path.exists(p):
                    items.append(p)
            if len(items) >= n:
                break

print(f"模式={mode}  测试 {len(items)} 条")
base = rss_mb()
marks = []
for i, it in enumerate(items, 1):
    if mode == "video":
        out = plugin._regularize_videos([it], **kwargs)
    else:
        out = plugin._regularize_images([it], **kwargs)
    del out
    if i % 10 == 0:
        cur = rss_mb()
        marks.append((i, cur))
        print(f"  {i:3d} 条  RSS={cur:8.1f} MB  (Δ{cur-base:+.1f} MB)")
if len(marks) >= 2:
    slope = (marks[-1][1] - marks[0][1]) / (marks[-1][0] - marks[0][0])
    print(f"\n结果: {mode}  每条件增长 {slope:+.3f} MB  → 168 样本/步 × {0.134 if mode=='video' else 0.866:.3f} 占比 ≈ {slope*168*(0.134 if mode=='video' else 0.866):+.1f} MB/优化步")

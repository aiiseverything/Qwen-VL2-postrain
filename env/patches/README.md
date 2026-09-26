# 本地补丁（针对 `third_party/`）

> ⚠️ `third_party/` 在 `.gitignore` 里 —— 对它的改动**不会进主仓库**，
> 一旦重克隆 / 重装就会**静默丢失**。所以每个补丁在这里存一份 `.patch`，
> 并写清楚**为什么**要打。

应用方式：

```bash
cd third_party/LlamaFactory
git apply /home/ml-user/workdir/project-2/env/patches/<名字>.patch
```

---

## `llamafactory-video-decode-leak.patch`（2026-09-27）

**改哪**：`src/llamafactory/data/mm_plugin.py` → `Qwen2VLPlugin._regularize_videos`
（`qwen2_vl` 模板的视频解码路径，训练时的 collator 每遇到一个视频样本就走这里）

**为什么**：原代码 `av.open(video, "r")` 解码完视频**从不调用 `container.close()`**，
每个视频泄漏约 **2.6 MB** 的 PyAV 解码缓冲。训练时每个 dataloader worker
以 ~12 MB/min 线性增长，几小时就撑爆 64 GiB 容器。

关键在于**只有真正解码视频才会涨** —— 所以 4 小时的 tokenize 阶段
（只读 jsonl）和纯图像数据都看不出异常，只有训练跑起来才暴露。

**实测**（40 个真实视频，复刻 `mm_plugin` 的解码模式，`/tmp/leak_test.py`）：

| 变体 | 40 个视频后 RSS | 每视频增长 | 推算每优化步 |
|---|---|---|---|
| 原代码（无 close） | 219.4 MB（+183.4） | **+2.61 MB** | ~+59 MB |
| 加 `container.close()` | 124.2 MB（第 20 个之后**完全走平**） | **+0.04 MB** | ~+1 MB |

**改法**：把解码块包进 `try/finally`，在 `finally` 里 `container.close()`。
用 `try/finally` 是因为坏视频会抛 `ValueError`，那条路径也必须释放
（同类兄弟函数 `_get_qwen_video_stream_metadata` 本来就调了 `close()`）。

**影响面**：`Qwen3VLPlugin` / `GLM4VPlugin` / `Qwen2OmniPlugin` 都继承
`Qwen2VLPlugin`，一并受益。

**背景**：这是 2026-09-26/27 三次训练 OOM 的**第二个**机制（第一个是
保存 checkpoint 时的物化）。完整复盘见 `docs/training/OOM-复盘-20260926-27.md`，
问题编号条目见 `docs/05-下载准备搭建记录.md` P41。

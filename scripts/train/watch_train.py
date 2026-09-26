#!/usr/bin/env python
# ============================================================================
# 训练监视进程 —— 盯 SFT/DPO/GRPO 全链的健康度, 并定时出图
# ----------------------------------------------------------------------------
# 为什么需要它:
#   2026-09-26 的 SFT 死在 step 27, 是**容器 host 内存打满被内核 SIGKILL**,
#   而不是显存 OOM (`exit -9` + cgroup `oom_kill 1`, 日志里 0 次 CUDA OOM)。
#   这种死法在日志里只留下一行 SIGKILL, 事先毫无征兆 —— 唯一能提前发现的办法
#   就是**持续采样 host 内存**。本脚本因此把内存当第一指标。
#
# 它盯什么 (每轮都判, 结果写 results/watch/):
#   1. host 内存 / cgroup        —— warn 50GB / crit 58GB (上限 64GB)
#   2. GPU 显存与利用率 (7 卡)    —— 显存 >42GB warn; 某卡连续掉零 warn
#   3. 训练进程是否还在           —— 用 nvidia-smi compute-apps 判定 (框架无关)
#   4. 进度是否在推进             —— trainer_log.jsonl 的 step 不涨就 warn
#   5. loss / grad_norm 是否正常  —— NaN/inf crit, 突刺或长期 0 warn
#   6. checkpoint 是否按期落盘
#   7. 日志尾有没有 Traceback / CUDA OOM / SIGKILL
#
# 出图 (默认每 5 分钟重画一次, 覆盖同名文件, 直接可看):
#   results/watch/01_loss.png     loss + grad_norm + lr vs step
#   results/watch/02_mem.png      host 内存 + 7 卡显存/利用率 vs 时间
#   results/watch/03_progress.png step vs 时间 + 吞吐/ETA
#
# 用法:
#   setsid nohup .venv-lf/bin/python scripts/train/watch_train.py \
#       > logs/watch/watch.log 2>&1 < /dev/null &
#
#   # 可选: 进程死了且有 checkpoint 时自动续训 (默认关)
#   setsid nohup .venv-lf/bin/python scripts/train/watch_train.py --auto-resume &
# ============================================================================
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # 无显示器环境, 必须在 pyplot 之前设
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CGROUP = Path("/sys/fs/cgroup")
OUT_DIR = PROJECT_ROOT / "results" / "watch"
METRICS_CSV = OUT_DIR / "metrics.csv"
ALERTS = OUT_DIR / "alerts.log"
STATUS = OUT_DIR / "STATUS.txt"

# trainer_log.jsonl 可能出现在任一阶段; 每轮挑最新的那个
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
# 进度/梯度范数从 stdout 日志里抓 (chain 的 tee 输出 + 诊断跑的输出)
LOG_GLOBS = ["logs/chain_*.log", "logs/diag/sft_memtest*.log", "logs/grpo*.log"]

LOSS_RE = re.compile(r"\{'loss':\s*([0-9.eE+-]+).*?'grad_norm':\s*([0-9.eE+-]+).*?'learning_rate':\s*([0-9.eE+-]+)")
BAD_RE = re.compile(r"Traceback|CUDA out of memory|SIGKILL|ChildFailedError|RuntimeError|Killed")

CSV_HEADER = "ts,iso,total_mb,anon_mb,file_mb,slab_mb,gpu_mem_max_mb,gpu_util_avg,gpu_idle_count,step,loss,grad_norm,lr,epoch,total_steps"


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def read_cgroup() -> dict:
    """容器内存: 总量 + anon/file/slab 拆分 (单位 MB)。"""
    mb = 1048576
    out = {"total": 0.0, "anon": 0.0, "file": 0.0, "slab": 0.0}
    try:
        out["total"] = int(CGROUP.joinpath("memory.current").read_text()) / mb
        stat = CGROUP.joinpath("memory.stat").read_text()
        for line in stat.splitlines():
            k, v = line.split()[:2]
            if k in out:
                out[k] = int(v) / mb
    except Exception:
        pass
    return out


def read_gpus() -> tuple[list[dict], list[dict]]:
    """返回 (每卡状态, 占用显存的进程)。框架无关 —— GRPO 走 vLLM 也算得准。"""
    gpus, apps = [], []
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in raw.strip().splitlines():
            idx, used, util, total = [x.strip() for x in line.split(",")]
            gpus.append({"index": int(idx), "mem": float(used), "util": float(util), "total": float(total)})
    except Exception:
        pass
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in raw.strip().splitlines():
            if not line.strip():
                continue
            pid, mem = [x.strip() for x in line.split(",")]
            apps.append({"pid": int(pid), "mem": float(mem)})
    except Exception:
        pass
    return gpus, apps


def count_procs(pattern: str) -> int:
    try:
        out = subprocess.run(["pgrep", "-c", "-f", pattern], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip() or 0)
    except Exception:
        return 0


def newest(pattern_list: list[str], max_age: float | None = None) -> Path | None:
    """最新的匹配文件; 给 max_age 时只认这段时间内写过的 (避免读到上次跑批的残留)。"""
    cands: list[Path] = []
    for pat in pattern_list:
        cands += [Path(p) for p in glob.glob(str(PROJECT_ROOT / pat))]
    cands = [p for p in cands if p.is_file()]
    if max_age is not None:
        now = time.time()
        cands = [p for p in cands if now - p.stat().st_mtime <= max_age]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def read_progress(max_log_age: float = 900) -> dict:
    """从最新的 trainer_log.jsonl 读进度; 从 stdout 日志读 grad_norm。

    ⚠️ 必须过滤"陈旧"的 trainer_log.jsonl —— 上次崩溃/校准跑留下的旧日志
    (results/calib/sft_b8/、上一次崩掉的 results/checkpoints/sft/) 会让监视
    误报一个几小时前的 step 和假告警。只有 max_log_age 秒内被写过的才算数。
    """
    prog: dict = {}
    now = time.time()
    logs = []
    for pat in ("checkpoints/*/trainer_log.jsonl", "diag/*/trainer_log.jsonl", "calib/*/trainer_log.jsonl"):
        logs += list((PROJECT_ROOT / "results").glob(pat))
    logs = [p for p in logs if now - p.stat().st_mtime <= max_log_age]
    if logs:
        newest_log = max(logs, key=lambda p: p.stat().st_mtime)
        prog["log"] = str(newest_log.relative_to(PROJECT_ROOT))
        prog["mtime"] = newest_log.stat().st_mtime
        last = None
        try:
            for line in newest_log.read_text(errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        if last:
            # LLaMA-Factory 用 current_steps/total_steps; transformers 是 global_step
            prog["step"] = last.get("current_steps", last.get("global_step"))
            prog["total_steps"] = last.get("total_steps")
            prog["loss"] = last.get("loss")
            prog["lr"] = last.get("lr", last.get("learning_rate"))
            prog["epoch"] = last.get("epoch")
            prog["remaining"] = last.get("remaining_time")
    # grad_norm 只在 stdout 里
    log = newest(LOG_GLOBS, max_age=max_log_age)
    if log:
        prog["stdout_log"] = str(log.relative_to(PROJECT_ROOT))
        prog["stdout_mtime"] = log.stat().st_mtime
        try:
            tail = log.read_text(errors="ignore")[-40000:]
            hits = LOSS_RE.findall(tail.replace("\r", "\n"))
            if hits:
                loss, gn, lr = hits[-1]
                prog["grad_norm"] = float(gn)
                prog["stdout_loss"] = float(loss)
                prog["stdout_lr"] = float(lr)
                prog["grad_series"] = [float(h[1]) for h in hits[-40:]]
        except Exception:
            pass
    return prog


def scan_log_tail() -> str | None:
    log = newest(LOG_GLOBS, max_age=3600)
    if not log:
        return None
    try:
        tail = log.read_text(errors="ignore")[-20000:]
    except Exception:
        return None
    for line in tail.replace("\r", "\n").splitlines():
        if BAD_RE.search(line) and "Traceback (most recent call last)" not in line:
            return line.strip()[:200]
    return None


def ckpt_info() -> tuple[int, float]:
    """(checkpoint 个数, 最新 checkpoint 的 mtime)"""
    dirs = []
    if CKPT_DIR.exists():
        for pattern in ("sft/checkpoint-*", "dpo/checkpoint-*", "grpo/*/checkpoint-*", "grpo/checkpoint-*"):
            dirs += [Path(p) for p in glob.glob(str(CKPT_DIR / pattern))]
    if not dirs:
        return 0, 0.0
    return len(dirs), max(d.stat().st_mtime for d in dirs)


# ---------------------------------------------------------------------------
# 健康判定
# ---------------------------------------------------------------------------
def evaluate(args, cg, gpus, apps, prog, hist) -> list[tuple[str, str, str]]:
    """返回 [(级别, 项目, 说明)] —— 级别 OK/WARN/CRIT"""
    res: list[tuple[str, str, str]] = []

    # 1. host 内存 —— 判据是 **anon(匿名内存)**, 不是总量。
    #    原因: 存 checkpoint 时要在主机内存里物化 ~19GB (docs/05 P37),
    #    而 file (页缓存) 会被内核自动回收腾地方 (实测保存时 file 从 6.2G 被挤到 1.9G)。
    #    ⇒ 生死线: anon + 19GB < 64GB, 即 anon 超过 ~45GB 就危险。
    agb = cg["anon"] / 1024
    gb = cg["total"] / 1024
    if agb >= args.anon_crit:
        res.append(("CRIT", "host内存", f"anon {agb:.1f}GB ≥ {args.anon_crit}GB —— 收尾保存要 +19GB, 必定撞 64GB 上限"))
    elif agb >= args.anon_warn:
        res.append(("WARN", "host内存", f"anon {agb:.1f}GB ≥ {args.anon_warn}GB —— 离危险线(45GB)不远了"))
    else:
        res.append(("OK", "host内存", f"anon {agb:.1f}GB (收尾保存余量 ~{64-agb-19:.0f}GB) / 总量 {gb:.1f}GB (file {cg['file']/1024:.1f}G, slab {cg['slab']/1024:.1f}G)"))

    # 2. GPU
    if gpus:
        used = [g["mem"] for g in gpus]
        util = [g["util"] for g in gpus]
        hot = [g["index"] for g in gpus if g["mem"] > args.gpu_mem_warn]
        if hot:
            res.append(("WARN", "显存", f"卡 {hot} 已用 >{args.gpu_mem_warn}MB (单卡 {gpus[0]['total']:.0f}MB)"))
        else:
            res.append(("OK", "显存", f"峰值 {max(used):.0f}MB / {gpus[0]['total']:.0f}MB"))
        idle = [g["index"] for g in gpus if g["util"] == 0]
        hist.setdefault("idle_streak", {})
        for g in gpus:
            hist["idle_streak"][g["index"]] = hist["idle_streak"].get(g["index"], 0) + 1 if g["util"] == 0 else 0
        chronic = [i for i, s in hist["idle_streak"].items() if s >= args.idle_ticks]
        if chronic:
            res.append(("WARN", "GPU利用率", f"卡 {chronic} 连续 {args.idle_ticks} 次采样 0% —— 可能掉队/挂住"))
        else:
            res.append(("OK", "GPU利用率", f"均值 {sum(util)/len(util):.0f}%，{len(idle)} 张空闲(瞬时)"))
    else:
        res.append(("WARN", "GPU", "nvidia-smi 读不到卡"))

    # 3. 进程在不在
    trainer_alive = count_procs("llamafactory/launcher.py|verl.trainer.main|run_chain.sh|launch_4xl40.sh")
    # ⚠️ 不要用 pgrep -f pt_data_worker 数 worker —— 它认的是 cmdline, 而 torch 只改了
    #    comm, 实测会数成 0。直接数容器里的 python 进程总数, 它才是 host 内存的真正来源。
    npy = count_procs("python")
    if apps:
        res.append(("OK", "训练进程", f"{len(apps)} 个进程占用显存，容器 python 进程共 {npy} 个"))
    elif trainer_alive:
        res.append(("OK", "训练进程", f"进程在（{trainer_alive} 个，未占用显存 —— 多半在 tokenize/加载阶段），python 共 {npy} 个"))
    else:
        res.append(("CRIT", "训练进程", "训练进程不存在，且没有进程占用显存 —— 训练已退出"))

    # 4. 进度在推进吗
    step = prog.get("step")
    if step is None:
        res.append(("WARN", "进度", "还没有 trainer_log.jsonl —— 尚未进入训练循环"))
    else:
        total = prog.get("total_steps") or 0
        age = time.time() - prog.get("mtime", time.time())
        pct = f"{100*step/total:.1f}%" if total else "?"
        if age > args.stall_min * 60 and trainer_alive:
            res.append(("WARN", "进度", f"step {step}/{total} ({pct}) 已 {age/60:.0f} 分钟没有新日志 —— 可能卡住/在 thrash"))
        else:
            res.append(("OK", "进度", f"step {step}/{total} ({pct})，epoch {prog.get('epoch')}，日志 {age:.0f}s 前"))

    # 5. loss / grad_norm
    loss = prog.get("loss", prog.get("stdout_loss"))
    if loss is not None:
        if not math.isfinite(loss):
            res.append(("CRIT", "loss", f"loss = {loss} —— 非有限值，训练已经坏了"))
        else:
            hist.setdefault("loss", []).append(float(loss))
            hist["loss"] = hist["loss"][-50:]
            med = sorted(hist["loss"])[len(hist["loss"]) // 2]
            if med > 0 and loss > 3 * med:
                res.append(("WARN", "loss", f"loss {loss:.4f} 是近期中位数 {med:.4f} 的 {loss/med:.1f} 倍 —— 突刺"))
            else:
                res.append(("OK", "loss", f"{loss:.4f} (近期中位 {med:.4f})"))
    gn = prog.get("grad_norm")
    if gn is not None:
        if not math.isfinite(gn):
            res.append(("CRIT", "grad_norm", f"{gn} —— 非有限值"))
        else:
            hist.setdefault("gn", []).append(float(gn))
            hist["gn"] = hist["gn"][-40:]
            zero_streak = 0
            for v in reversed(hist["gn"]):
                if v == 0:
                    zero_streak += 1
                else:
                    break
            if zero_streak >= 8:
                res.append(("WARN", "grad_norm", f"连续 {zero_streak} 次为 0 —— 梯度可能是空的 (loss mask 有问题?)"))
            elif gn > 100:
                res.append(("WARN", "grad_norm", f"{gn:.2f} > 100 —— 梯度偏大"))
            else:
                res.append(("OK", "grad_norm", f"{gn:.3f}"))

    # 6. checkpoint
    n_ckpt, ckpt_mtime = ckpt_info()
    if step and n_ckpt == 0 and step >= args.save_steps:
        res.append(("WARN", "checkpoint", f"已到 step {step} (>={args.save_steps}) 却还没有 checkpoint"))
    else:
        age = f"{(time.time()-ckpt_mtime)/60:.0f} 分钟前" if ckpt_mtime else "还没有"
        res.append(("OK", "checkpoint", f"{n_ckpt} 个，最新 {age}"))

    # 7. 日志里的错误
    bad = scan_log_tail()
    if bad:
        res.append(("CRIT", "日志", f"日志尾部出现异常: {bad}"))
    return res


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def read_log_history() -> tuple[list[int], list, list]:
    """从 trainer_log.jsonl 读**完整**的 (step, loss, lr )。

    每个 logging step 一条, 比"每 60 秒采样一次"完整得多 (采样会漏掉中间的 logging step)。
    同一个文件里若混了多次跑批 (chain 的 tee -a 会追加), 按 step 去重、保留最后一次。
    """
    logs: list[Path] = []
    for pat in ("checkpoints/*/trainer_log.jsonl", "diag/*/trainer_log.jsonl", "calib/*/trainer_log.jsonl"):
        logs += list((PROJECT_ROOT / "results").glob(pat))
    if not logs:
        return [], [], []
    newest_log = max(logs, key=lambda p: p.stat().st_mtime)
    by_step: dict[int, tuple] = {}
    try:
        for line in newest_log.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = d.get("current_steps", d.get("global_step"))
            if s is None:
                continue
            by_step[int(s)] = (d.get("loss"), d.get("lr", d.get("learning_rate")))
    except Exception:
        return [], [], []
    xs = sorted(by_step)
    return xs, [by_step[s][0] for s in xs], [by_step[s][1] for s in xs]


def plot_loss(rows, prog, out: Path) -> None:
    """三个面板: loss(带滑动平均) / grad_norm / learning_rate。

    * **loss / lr** 取自 `trainer_log.jsonl` —— **每个 logging step 一条, 完整**
      (每 60 秒采样会漏掉中间的 logging step)。
    * **grad_norm** 取自 metrics.csv 的 60 秒采样 —— 它只出现在 stdout 日志里,
      那行没有 step 号, 无法可靠对齐到 step。
    早先版本画的是"上次出图以来的增量"(内存累积、画完清空), 每次只有 5 分钟窗口,
    完全看不出趋势 —— 2026-09-27 修。
    """

    def series_from_rows(col: str):
        pts: list[tuple[int, float]] = []
        for r in rows:
            v = r.get(col, "")
            if r.get("step") and v not in ("", None):
                try:
                    pts.append((int(r["step"]), float(v)))
                except (TypeError, ValueError):
                    pass
        if not pts:
            return [], []
        xs, ys = zip(*pts)
        return list(xs), list(ys)

    xs_loss: list[int] = []
    ys_loss: list[float] = []
    xs_lr: list[int] = []
    ys_lr: list[float] = []
    for s, lo, lr in zip(*read_log_history()):
        if lo is not None:
            xs_loss.append(s)
            ys_loss.append(float(lo))
        if lr is not None:
            xs_lr.append(s)
            ys_lr.append(float(lr))
    xs_gn, ys_gn = series_from_rows("grad_norm")
    if not xs_loss and not xs_gn:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if xs_loss:
        axes[0].plot(xs_loss, ys_loss, lw=0.8, alpha=0.5, label="loss")
        if len(ys_loss) >= 10:
            w = 10
            ma = [sum(ys_loss[max(0, i - w):i + 1]) / len(ys_loss[max(0, i - w):i + 1]) for i in range(len(ys_loss))]
            axes[0].plot(xs_loss, ma, lw=1.8, label=f"loss (ma{w})")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if xs_gn:
        axes[1].plot(xs_gn, ys_gn, lw=0.9, color="tab:orange")
    axes[1].axhline(1.0, ls="--", c="grey", lw=0.8, label="clip=1.0")
    axes[1].set_ylabel("grad_norm")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    if xs_lr:
        axes[2].plot(xs_lr, ys_lr, lw=0.9, color="tab:green")
    axes[2].set_ylabel("learning_rate")
    axes[2].set_xlabel("optimizer step")
    axes[2].grid(alpha=0.3)
    last = (xs_loss or xs_gn)[-1]
    total = prog.get("total_steps")
    # ⚠️ 本机没有中日韩字体 (fc-list :lang=zh 为空), 图里一律用英文, 免得渲染成方块
    fig.suptitle(f"Training curves (step {last}/{total})" if total else "Training curves")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_mem(rows, out: Path) -> None:
    if len(rows) < 2:
        return
    ts = [datetime.fromisoformat(r["iso"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ts, [r["total_mb"] / 1024 for r in rows], lw=1.6, label="container total")
    axes[0].plot(ts, [r["anon_mb"] / 1024 for r in rows], lw=1.2, label="anon (process heap)")
    axes[0].plot(ts, [r["file_mb"] / 1024 for r in rows], lw=1.2, label="file (page cache)")
    axes[0].axhline(64, ls="--", c="red", lw=1.2, label="cgroup max 64GB")
    axes[0].axhline(58, ls=":", c="orange", lw=1.0, label="crit 58GB")
    axes[0].set_ylabel("GB")
    axes[0].set_title("Container host memory  (this is what killed the first SFT run)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(ts, [r["gpu_mem_max_mb"] / 1024 for r in rows], lw=1.5, color="tab:purple", label="max GPU mem (GB)")
    axes[1].plot(ts, [r["gpu_util_avg"] for r in rows], lw=1.2, color="tab:blue", label="avg util (%)")
    axes[1].axhline(46, ls="--", c="red", lw=1.0, label="46GB per card")
    axes[1].set_ylabel("GB / %")
    axes[1].set_xlabel("time")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_progress(rows, prog, out: Path) -> None:
    pts = [(datetime.fromisoformat(r["iso"]), r["step"]) for r in rows if r.get("step") is not None]
    if len(pts) < 2:
        return
    ts, steps = zip(*pts)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts, steps, lw=1.8, marker=".", ms=3)
    ax.set_ylabel("optimizer step")
    ax.set_xlabel("time")
    total = prog.get("total_steps")
    if total:
        ax.axhline(total, ls="--", c="grey", lw=1.0)
        ax.text(0.99, 0.95, f"target {total}", transform=ax.transAxes, ha="right", fontsize=9, color="grey")
    # 用最近 30 个采样点估吞吐 + ETA
    if len(steps) >= 3:
        span_t = (ts[-1] - ts[0]).total_seconds()
        span_s = steps[-1] - steps[0]
        if span_t > 0 and span_s > 0:
            rate = span_s / span_t * 3600        # step/小时
            txt = f"last {len(steps)} samples: {rate:.0f} step/h"
            if total and rate > 0:
                eta = datetime.now() + timedelta(hours=(total - steps[-1]) / rate)
                txt += f", ETA {eta:%m-%d %H:%M}"
            if prog.get("remaining"):
                txt += f"  (framework ETA {prog['remaining']})"
            ax.text(0.01, 0.95, txt, transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Progress")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="训练健康监视 + 定时出图")
    ap.add_argument("--interval", type=float, default=60, help="采样间隔秒 (默认 60)")
    ap.add_argument("--plot-every", type=float, default=300, help="出图间隔秒 (默认 300)")
    ap.add_argument("--stall-min", type=float, default=12, help="多久没有新 step 判为卡住 (分钟)")
    ap.add_argument("--anon-warn", type=float, default=42, help="anon 内存 warn 阈值 GB (危险线 45)")
    ap.add_argument("--anon-crit", type=float, default=47, help="anon 内存 crit 阈值 GB")
    ap.add_argument("--gpu-mem-warn", type=float, default=42000, help="单卡显存 warn 阈值 MB")
    ap.add_argument("--idle-ticks", type=int, default=3, help="连续几次采样 0% 判该卡掉队")
    ap.add_argument("--save-steps", type=int, default=100, help="配置里的 save_steps, 用于校验 checkpoint")
    ap.add_argument("--auto-resume", action="store_true", help="进程死亡且有 checkpoint 时自动续训 (默认关)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs" / "watch").mkdir(parents=True, exist_ok=True)
    hist: dict = {}
    last_plot = 0.0
    last_alert: dict[str, float] = {}

    def log(msg: str) -> None:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    def alert(msg: str) -> None:
        now = time.time()
        if now - last_alert.get(msg, 0) > 1800:      # 同一个告警 30 分钟只写一次, 免刷屏
            last_alert[msg] = now
            with ALERTS.open("a") as f:
                f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")

    log(f"开始监视 | 间隔 {args.interval}s | 出图 {args.plot_every}s | 输出 {OUT_DIR}")
    if not METRICS_CSV.exists():
        METRICS_CSV.write_text(CSV_HEADER + "\n")

    while True:
        t0 = time.time()
        try:
            cg = read_cgroup()
            gpus, apps = read_gpus()
            prog = read_progress()
            checks = evaluate(args, cg, gpus, apps, prog, hist)

            gpu_mem_max = max((g["mem"] for g in gpus), default=0.0)
            gpu_util_avg = sum(g["util"] for g in gpus) / len(gpus) if gpus else 0.0
            gpu_idle = sum(1 for g in gpus if g["util"] == 0)

            iso = datetime.now().isoformat(timespec="seconds")
            # 按进程归因: 内存涨的时候到底是谁在涨 (上次 OOM 复盘缺的就是这个)
            try:
                ps = subprocess.run(["ps", "-eo", "rss,pid,comm", "--sort=-rss", "--no-headers"],
                                    capture_output=True, text=True, timeout=15).stdout
                top = []
                for line in ps.strip().splitlines()[:6]:
                    parts = line.split()
                    if len(parts) >= 3:
                        top.append(f"{parts[1]}:{parts[2]}:{float(parts[0])/1024:.0f}MB")
                with (OUT_DIR / "procs.log").open("a") as f:
                    f.write(f"{iso} " + "  ".join(top) + "\n")
            except Exception:
                pass

            with METRICS_CSV.open("a") as f:
                f.write(",".join(str(x) for x in [
                    time.time(), iso, round(cg["total"]), round(cg["anon"]), round(cg["file"]), round(cg["slab"]),
                    round(gpu_mem_max), round(gpu_util_avg), gpu_idle,
                    prog.get("step", ""), prog.get("loss", ""), prog.get("grad_norm", ""),
                    prog.get("lr", ""), prog.get("epoch", ""), prog.get("total_steps", ""),
                ]) + "\n")

            # 累历史, 供出图
            rows = []
            try:
                import csv
                with METRICS_CSV.open() as f:
                    rows = list(csv.DictReader(f))
                rows = [{**r, "total_mb": float(r["total_mb"]), "anon_mb": float(r["anon_mb"]),
                         "file_mb": float(r["file_mb"]), "gpu_mem_max_mb": float(r["gpu_mem_max_mb"]),
                         "gpu_util_avg": float(r["gpu_util_avg"]),
                         "step": int(r["step"]) if r["step"] else None} for r in rows]
            except Exception:
                rows = []

            step = prog.get("step")

            # 状态摘要
            lines = [f"# 训练监视状态  {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
            for lvl, name, msg in checks:
                mark = {"OK": "✅", "WARN": "⚠️ ", "CRIT": "❌"}[lvl]
                lines.append(f"{mark} {name:10s} {msg}")
                if lvl != "OK":
                    alert(f"{lvl} {name}: {msg}")
            if prog.get("log"):
                lines += ["", f"进度来源: {prog['log']}"]
            if prog.get("stdout_log"):
                lines += [f"日志来源: {prog['stdout_log']}"]
            STATUS.write_text("\n".join(lines) + "\n")
            for lvl, name, msg in checks:
                if lvl != "OK":
                    log(f"{lvl} {name}: {msg}")

            # 出图
            if time.time() - last_plot >= args.plot_every:
                last_plot = time.time()
                try:
                    plot_loss(rows, prog, OUT_DIR / "01_loss.png")
                    plot_mem(rows, OUT_DIR / "02_mem.png")
                    plot_progress(rows, prog, OUT_DIR / "03_progress.png")
                    log(f"已出图 -> {OUT_DIR}/0[123]_*.png (累计 {len(rows)} 个采样点)")
                except Exception as e:
                    log(f"出图失败(不影响监视): {e!r}")

            # 可选自动续训
            if args.auto_resume:
                dead = not apps and count_procs("llamafactory/launcher.py|verl.trainer.main") == 0
                n_ckpt, _ = ckpt_info()
                if dead and count_procs("run_chain.sh") == 0 and n_ckpt > 0 and step is not None:
                    alert("CRIT 训练进程已死但存在 checkpoint —— 自动续训 (run_chain.sh)")
                    subprocess.Popen(
                        ["bash", str(PROJECT_ROOT / "scripts/train/run_chain.sh")],
                        cwd=PROJECT_ROOT, env={**os.environ, "STAGES": "sft dpo"},
                        stdout=open(PROJECT_ROOT / "logs/chain_resume.log", "a"), stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
        except Exception as e:
            log(f"监视循环异常(继续): {e!r}")

        time.sleep(max(5, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()

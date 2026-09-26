#!/usr/bin/env python3
"""环境自检 —— CUDA / 显存 / NCCL / 版本一致性。

⚠️ 设计约束 (用户要求: 不要占用显存):
   本脚本默认**只用 nvidia-smi 查询 GPU**, 不 import torch、不创建 CUDA context。
   在 4×L40 上 `torch.cuda.is_available()` 会为每张卡建立一个约 300MB 的
   context, 如果机器上还有别的训练在跑, 这会白白吃掉显存。
   需要真实的 torch/NCCL 检查时, 显式加 --with-cuda。

用法:
    python env/check_env.py --stage sft|rl|eval [--with-cuda]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED = {
    "sft":  ["torch", "transformers", "peft", "trl", "datasets", "deepspeed", "accelerate"],
    "rl":   ["torch", "transformers", "vllm", "ray", "datasets", "accelerate"],
    "eval": ["torch", "transformers", "datasets", "accelerate"],
}
VENV = {"sft": ".venv-lf", "rl": ".venv-rl", "eval": ".venv-eval"}
# 已知会互相打架的组合 (PLAN K2)
CONFLICT = [("llamafactory", "vllm"), ("deepspeed", "vllm")]


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        return f"<{e}>"


def check_gpus_nvml() -> int:
    """仅用 nvidia-smi, 不碰 CUDA context。"""
    if not shutil.which("nvidia-smi"):
        print("  [!!] 找不到 nvidia-smi —— 无 NVIDIA 驱动或非 GPU 机器")
        return 1
    out = sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
              "--format=csv,noheader"])
    if not out or out.startswith("<"):
        print(f"  [!!] nvidia-smi 调用失败: {out}")
        return 1
    free = 0
    print("  GPU 状态 (只读, 未创建 CUDA context):")
    for line in out.splitlines():
        idx, name, used, total, util = [x.strip() for x in line.split(",")]
        mu = int(used.split()[0]); mt = int(total.split()[0])
        avail = mt - mu
        if avail > 8000:
            free += 1
        print(f"    [{idx}] {name:<16} {used:>9} / {total:<9} util={util:>4}  可用 {avail} MiB")
    print(f"  -> 显存充足 (>8GB 可用) 的卡: {free} 张")
    print(f"  driver: {sh(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']).splitlines()[0] if out else '?'}")
    return 0


def check_torch(stage: str) -> int:
    py = PROJECT_ROOT / VENV[stage] / "bin/python"
    if not py.exists():
        print(f"  [!!] venv 不存在: {py} —— 先跑 env/server_l40_{stage}.sh")
        return 1
    code = (
        "import importlib, json, sys\n"
        f"mods={REQUIRED[stage]!r}\n"
        "r={}\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod=importlib.import_module(m); r[m]=getattr(mod,'__version__','?')\n"
        "    except Exception as e: r[m]='MISSING: %s' % type(e).__name__\n"
        "print(json.dumps(r))\n"
    )
    out = sh([str(py), "-c", code])
    try:
        vers = json.loads(out)
    except Exception:                                    # noqa: BLE001
        print(f"  [!!] 无法读取 venv 版本: {out[:200]}")
        return 1
    bad = 0
    for m in REQUIRED[stage]:
        v = vers.get(m, "?")
        ok = not str(v).startswith("MISSING")
        bad += 0 if ok else 1
        print(f"    {'OK ' if ok else '!! '} {m:<14} {v}")
    for a, b in CONFLICT:
        if a in vers and b in vers and not str(vers[a]).startswith("MISSING") \
           and not str(vers[b]).startswith("MISSING"):
            print(f"  [!!] 版本冲突风险 (PLAN K2): 同一 venv 内同时存在 {a} 与 {b}")
            bad += 1
    return 1 if bad else 0


def check_cuda_real(stage: str) -> int:
    """可选的真实 CUDA/NCCL 检查 —— 会创建 CUDA context, 占用显存。"""
    py = PROJECT_ROOT / VENV[stage] / "bin/python"
    code = ("import torch;print(torch.cuda.is_available(), torch.cuda.device_count(),"
            "torch.version.cuda, torch.cuda.nccl.version() if torch.cuda.is_available() else '')")
    print(f"  [--with-cuda] {sh([str(py), '-c', code])}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "rl", "eval"], required=True)
    ap.add_argument("--with-cuda", action="store_true",
                    help="执行真实 CUDA 检查 (⚠️ 会创建 context 并占用显存)")
    args = ap.parse_args()

    print(f"=== Postrain 环境自检 | stage={args.stage} ===")
    print("\n[1] GPU (nvidia-smi 只读)")
    rc_gpu = check_gpus_nvml()
    print("\n[2] venv 依赖版本")
    rc_venv = check_torch(args.stage)
    print("\n[3] 磁盘")
    print(f"  {sh(['df','-h',str(PROJECT_ROOT)]).splitlines()[-1] if sh(['df','-h',str(PROJECT_ROOT)]) else '?'}")
    print("\n[4] NCCL 环境变量")
    for k in ("NCCL_IB_DISABLE", "NCCL_DEBUG", "CUDA_VISIBLE_DEVICES"):
        import os
        print(f"    {k}={os.environ.get(k,'<未设置>')}")
    if args.with_cuda:
        print("\n[5] CUDA (真实检查)")
        check_cuda_real(args.stage)
    else:
        print("\n[5] 跳过真实 CUDA 检查 (加 --with-cuda 启用; 默认不占显存)")

    sys.exit(1 if (rc_gpu or rc_venv) else 0)


if __name__ == "__main__":
    main()

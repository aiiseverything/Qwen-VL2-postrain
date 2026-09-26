#!/usr/bin/env bash
# ============================================================================
# 全流程训练链 —— 一次挂上去, 无人值守跑完 SFT → DPO → GRPO → 评测
# ----------------------------------------------------------------------------
# 为什么要有这个: 每个阶段都依赖上一阶段的 checkpoint
#     SFT 产出  ->  DPO 的 model_name_or_path
#     DPO 产出  ->  GRPO 的 model_path
#     三者产出  ->  评测的 4 个模型
#   分开手动跑要盯 4 次; 串成一条链只需挂一次。
#
# 设计原则:
#   * **每一步都有门槛**: 上一步没产出 checkpoint 就绝不进下一步
#     (否则 DPO 会在空目录上报错, 白白浪费几小时)
#   * **失败不静默**: 任何一步失败立刻停止并写清楚, 不做"继续尝试"
#   * **全程脱离终端**: setsid + nohup, 关终端不影响
#
# 用法:
#   setsid nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC=7 \
#     bash scripts/train/run_chain.sh > logs/chain.log 2>&1 < /dev/null &
#
# 单独跑某一步: STAGES="sft" bash scripts/train/run_chain.sh
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export NPROC="${NPROC:-7}"
export FORCE_TORCHRUN=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export DEEPSPEED_TIMEOUT=180000000

LOG="$PROJECT_ROOT/logs"
CKPT="$PROJECT_ROOT/results/checkpoints"
mkdir -p "$LOG" "$CKPT"

STAGES="${STAGES:-calib sft dpo grpo eval}"

ts()   { date +"%Y-%m-%d %H:%M:%S"; }
banner() { printf '\n\033[1;36m%s\n  %s\n%s\033[0m\n' "════════════════════════════════════════════════════════" "$*" "════════════════════════════════════════════════════════"; }
ok()   { printf '\033[1;32m[%s] ✅ %s\033[0m\n' "$(ts)" "$*"; }
bad()  { printf '\033[1;31m[%s] ❌ %s\033[0m\n' "$(ts)" "$*"; }
info() { printf '\033[1;34m[%s]\033[0m %s\n' "$(ts)" "$*"; }

FAILED=0
abort() { bad "$*"; FAILED=1; }

# ---------------------------------------------------------------------------
# 前置检查 —— 宁可现在失败, 不要跑 10 小时才发现基础不对
# ---------------------------------------------------------------------------
banner "训练链启动 | 卡=$CUDA_VISIBLE_DEVICES | NPROC=$NPROC | 阶段=[$STAGES]"

for f in configs/sft_qwen2vl_2b_7gpu.yaml configs/dpo_qwen2vl_2b_7gpu.yaml \
         configs/grpo_qwen2vl_2b.yaml configs/ds_zero2.json \
         data/processed/sft_400k.clean.jsonl data/processed/dpo_10k.jsonl; do
  [[ -f "$f" ]] || abort "缺失必需文件: $f"
done
[[ -d models/Qwen2-VL-2B-Instruct ]] || abort "缺失基座模型"
(( FAILED )) && { bad "前置检查未通过, 退出"; exit 1; }
ok "前置检查通过"

info "GPU 状态 (确认没有被别人占用):"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    /'

# 记录环境快照 (训练笔记需要)
{
  echo "=== 训练链启动快照 $(ts) ==="
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NPROC=$NPROC"
  .venv-lf/bin/python -c "
import torch,transformers,deepspeed,peft,trl,datasets
print(f'torch={torch.__version__} transformers={transformers.__version__} deepspeed={deepspeed.__version__}')
print(f'peft={peft.__version__} trl={trl.__version__} datasets={datasets.__version__}')
" 2>/dev/null
  .venv-rl/bin/python -c "
import importlib.metadata as m
for p in ('vllm','verl','flash-attn'):
    try: print(f'{p}={m.version(p)}')
    except Exception: print(f'{p}=MISSING')
" 2>/dev/null
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "$LOG/chain_env_snapshot.txt" 2>&1
ok "环境快照 -> logs/chain_env_snapshot.txt"

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
have_ckpt() {   # have_ckpt <目录> : 目录里有 model.safetensors 或 checkpoint-*
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/model.safetensors" ]] && return 0
  compgen -G "$d/checkpoint-*" >/dev/null && return 0
  return 1
}

run_lf() {      # run_lf <阶段名> <配置> <日志>
  local name="$1" cfg="$2" logfile="$3"
  banner "$name"
  info "配置: $cfg"
  info "日志: $logfile"
  set -o pipefail
  if env PATH="$PROJECT_ROOT/.venv-lf/bin:$PATH" \
         PYTHONPATH="$PROJECT_ROOT/third_party/LlamaFactory/src" \
         bash scripts/train/launch_4xl40.sh "$name" 2>&1 | tee -a "$logfile"; then
    return 0
  else
    return 1
  fi
}

# ---------------------------------------------------------------------------
# 阶段 1: 吞吐校准 (20 步) —— 先确认 7 卡能跑, 再放全量
# ---------------------------------------------------------------------------
if [[ "$STAGES" == *calib* ]]; then
  banner "阶段 0/4: 吞吐校准 (20 步)"
  if env PATH="$PROJECT_ROOT/.venv-lf/bin:$PATH" \
         PYTHONPATH="$PROJECT_ROOT/third_party/LlamaFactory/src" \
         .venv-lf/bin/llamafactory-cli train configs/calib_sft_b8.yaml \
         > "$LOG/chain_calib.log" 2>&1; then
    ok "校准通过 —— 7 卡可跑"
    grep -aoE "'loss': [0-9.]+" "$LOG/chain_calib.log" | tail -3 | sed 's/^/    /'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    显存 /'
  else
    abort "校准失败 —— 7 卡配置有问题, 不放全量。见 logs/chain_calib.log"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 阶段 2: 全量 SFT
# ---------------------------------------------------------------------------
if [[ "$STAGES" == *sft* ]]; then
  if have_ckpt "$CKPT/sft"; then
    ok "SFT checkpoint 已存在, 跳过"
  else
    run_lf sft configs/sft_qwen2vl_2b_7gpu.yaml "$LOG/chain_sft.log" \
      || { abort "SFT 失败"; exit 1; }
    have_ckpt "$CKPT/sft" || { abort "SFT 跑完但没有 checkpoint"; exit 1; }
    ok "SFT 完成 -> $CKPT/sft"
  fi
fi

# ---------------------------------------------------------------------------
# 阶段 3: DPO (依赖 SFT checkpoint)
# ---------------------------------------------------------------------------
if [[ "$STAGES" == *dpo* ]]; then
  if ! have_ckpt "$CKPT/sft"; then
    abort "跳过 DPO: 缺 SFT checkpoint"
  elif have_ckpt "$CKPT/dpo"; then
    ok "DPO checkpoint 已存在, 跳过"
  else
    run_lf dpo configs/dpo_qwen2vl_2b_7gpu.yaml "$LOG/chain_dpo.log" \
      || { abort "DPO 失败"; exit 1; }
    have_ckpt "$CKPT/dpo" || { abort "DPO 跑完但没有 checkpoint"; exit 1; }
    ok "DPO 完成 -> $CKPT/dpo"
    # ★ DPO 健康检查: rewards/accuracies 应在 0.5-0.8
    info "DPO 健康指标 (rewards/accuracies 健康区间 0.5-0.8):"
    grep -aoE "rewards/accuracies['\"]?[: ]+[0-9.]+" "$LOG/chain_dpo.log" | tail -3 | sed 's/^/    /' || true
  fi
fi

# ---------------------------------------------------------------------------
# 阶段 4: GRPO (依赖 DPO checkpoint)
# ---------------------------------------------------------------------------
if [[ "$STAGES" == *grpo* ]]; then
  if ! have_ckpt "$CKPT/dpo"; then
    abort "跳过 GRPO: 缺 DPO checkpoint"
  else
    banner "阶段 3/4: GRPO (EasyR1)"
    if env PATH="$PROJECT_ROOT/.venv-rl/bin:$PATH" \
           PYTHONPATH="$PROJECT_ROOT/third_party/EasyR1" \
           CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
           .venv-rl/bin/python -m verl.trainer.main \
           config=configs/grpo_qwen2vl_2b.yaml \
           > "$LOG/chain_grpo.log" 2>&1; then
      ok "GRPO 完成 -> $CKPT/grpo"
      # ★ 最关键的健康检查: reward 不能恒为 0
      info "GRPO reward 统计 (均值必须 > 0, 方差必须 > 0):"
      grep -aoE "(reward|score)[^ ]*[: ]+[0-9.]+" "$LOG/chain_grpo.log" | tail -5 | sed 's/^/    /' || true
    else
      abort "GRPO 失败 —— 见 logs/chain_grpo.log"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 阶段 5: 评测 (4 模型 × 3 benchmark)
# ---------------------------------------------------------------------------
if [[ "$STAGES" == *eval* ]]; then
  banner "阶段 4/4: 评测"
  if env PATH="$PROJECT_ROOT/.venv-eval/bin:$PATH" \
         PROJECT_ROOT="$PROJECT_ROOT" \
         MODELS="pretrained sft dpo grpo" \
         DATA="MME MathVista_MINI Video-MME_16frame" \
         bash scripts/eval/run_all.sh > "$LOG/chain_eval.log" 2>&1; then
    ok "评测完成 -> results/report.md"
  else
    abort "评测失败 —— 见 logs/chain_eval.log"
  fi
fi

# ---------------------------------------------------------------------------
banner "训练链结束"
if (( FAILED )); then
  bad "有阶段失败, 见上文日志"
  exit 1
fi
ok "全部阶段完成"
echo
echo "  产物:"
for d in sft dpo grpo; do
  [[ -d "$CKPT/$d" ]] && printf "    ✅ %-40s %s\n" "$CKPT/$d" "$(du -sh "$CKPT/$d" 2>/dev/null | cut -f1)"
done
[[ -f results/report.md ]] && printf "    ✅ %-40s\n" "results/report.md"
echo
echo "  下一步: 把实测数据填进 docs/training/训练笔记.md"

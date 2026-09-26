#!/usr/bin/env bash
# ============================================================================
# 4×L40 多卡启动器 (R5) —— 带断点续训
# ----------------------------------------------------------------------------
# 用法:
#   bash scripts/train/launch_4xl40.sh sft            # 全量 SFT
#   bash scripts/train/launch_4xl40.sh sft --max_steps 100   # ★ Phase 7.4 吞吐校准
#   bash scripts/train/launch_4xl40.sh dpo
#   bash scripts/train/launch_4xl40.sh grpo
#
# 续训: 脚本自动探测 output_dir 下最新 checkpoint, 追加 --resume_from_checkpoint
# ============================================================================
set -euo pipefail

STAGE="${1:?用法: launch_4xl40.sh <sft|dpo|grpo> [额外参数...]}"; shift || true

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"

NPROC="${NPROC:-3}"                       # 本机实际可用 3 张 (4,5,6)
MASTER_PORT="${MASTER_PORT:-29500}"
VENV="${VENV:-$PROJECT_ROOT/.venv-lf}"    # LLaMA-Factory 独立 venv (见 docs/03, 规避 K2)

# 卡选择: PLAN 原定 4×L40; 本机 project-1 的训练常驻占用 0-3, 故默认用 4,5,6。
# ⚠️ 千万不要改回 0,1,2,3 —— 那会和别人正在跑的训练抢卡。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"

# ⚠️ 必须把 venv 的 bin 放到 PATH 最前面 (见 docs/05 问题 P20)。
#    LLaMA-Factory 的 launcher 是用 `torchrun` (靠 PATH 解析) 拉起子进程的。
#    若 PATH 上先命中别的环境 (本机是 /opt/venv/main/bin/torchrun, python 3.10,
#    没装 transformers), 子进程会以 **ModuleNotFoundError: No module named
#    'transformers'** 失败 —— 报错完全指不到真正的原因。
export PATH="${VENV}/bin:${PATH}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE=1                  # 无 IB 时禁用, 避免握手超时
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1                 # 否则重定向到文件时日志成块缓冲, 像卡死

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

case "$STAGE" in
  sft)  CFG=configs/sft_qwen2vl_2b.yaml ; OUT=results/checkpoints/sft ;;
  dpo)  CFG=configs/dpo_qwen2vl_2b.yaml ; OUT=results/checkpoints/dpo ;;
  grpo) CFG=configs/grpo_qwen2vl_2b.yaml; OUT=results/checkpoints/grpo ;;
  *) echo "未知阶段: $STAGE (可选 sft|dpo|grpo)" >&2; exit 2 ;;
esac

# SFT: 按实际卡数自动选配置, 保证 global batch 与 PLAN 的 4 卡方案一致。
# (4 卡 per_device4 × accum8 × 4 = 128; 3 卡 per_device4 × accum11 × 3 = 132)
if [[ "$STAGE" == "sft" && "$NPROC" != "4" && -f "configs/sft_qwen2vl_2b_${NPROC}gpu.yaml" ]]; then
  CFG="configs/sft_qwen2vl_2b_${NPROC}gpu.yaml"
  log "卡数为 $NPROC, 自动选用 $CFG"
fi
CFG="${CFG_OVERRIDE:-$CFG}"

# --- GRPO 走 EasyR1, 不走 LLaMA-Factory -------------------------------------
if [[ "$STAGE" == "grpo" ]]; then
  log "GRPO 阶段使用 EasyR1 (venv: .venv-rl)"
  log "⚠️ 首次运行请先冒烟: 用 geometry3k 跑通管线再上全量 (风险 K3)"
  # EasyR1/verl 用 Hydra 启动, 配置传参形式是 `config=<path>`（不是 --config）。
  # 并且 `verl` 包来自 EasyR1 仓库本身（vendored），必须让它可以被 import:
  #   要么 `uv pip install -e third_party/EasyR1`，要么设 PYTHONPATH。
  # 见 docs/05 问题 P31。
  export PYTHONPATH="$PROJECT_ROOT/third_party/EasyR1:${PYTHONPATH:-}"
  export PATH="$PROJECT_ROOT/.venv-rl/bin:$PATH"
  exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$PROJECT_ROOT/.venv-rl/bin/python" -m verl.trainer.main \
    "config=$CFG" "$@"
fi

# --- SFT / DPO 走 LLaMA-Factory ---------------------------------------------
#
# ⚠️ 参数形式很重要 (见 docs/05 问题 P30):
#   LLaMA-Factory 在 config 是 .yaml 时, 用 `OmegaConf.from_cli(sys.argv[2:])`
#   解析额外参数 —— 这是 **key=value 点列表语法**, 不是 argparse 的 `--key value`。
#   传 `--max_steps 100` 会直接报:
#     ValueError: Some keys are not used by the HfArgumentParser: ['--max_steps', '100']
#   传 `max_steps=100` 才正确。
#   下面统一把 `--key value` / `--key=value` 都翻译成 `key=value`, 免得调用方踩坑。
normalize_overrides() {
  local out=() key=()
  while (( $# )); do
    case "$1" in
      --*=*) out+=("${1#--}") ;;                       # --k=v  -> k=v
      --*)
        key="${1#--}"
        if [[ $# -ge 2 && "$2" != --* && "$2" != *=* ]]; then
          out+=("${key}=$2"); shift                    # --k v   -> k=v
        else
          out+=("${key}=true")                         # --flag  -> flag=true
        fi ;;
      *)     out+=("$1") ;;                            # 已经是 k=v
    esac
    shift
  done
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

OVERRIDES=()
while IFS= read -r line; do [[ -n "$line" ]] && OVERRIDES+=("$line"); done < <(normalize_overrides "$@")

# 断点续训: 用 key=value 形式, 不是 --resume_from_checkpoint
if [[ -d "$OUT" ]]; then
  LATEST=$(ls -1d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    log "检测到 checkpoint: $LATEST —— 自动续训"
    OVERRIDES+=("resume_from_checkpoint=$LATEST")
  fi
fi

log "启动 $STAGE | 配置 $CFG | $NPROC 卡 | 输出 $OUT"
log "覆盖参数: ${OVERRIDES[*]:-<无>}"

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$VENV/bin/llamafactory-cli" train "$CFG" ${OVERRIDES[@]+"${OVERRIDES[@]}"}

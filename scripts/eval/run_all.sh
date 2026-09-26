#!/usr/bin/env bash
# ============================================================================
# 一键评测 —— 4 个模型 × 3 个 benchmark
# ----------------------------------------------------------------------------
# 修正了初版的三处错误（见 docs/05 问题 P29）:
#   1. 入口写成 `python -m vlmeval.run` —— **该模块不存在**。
#      真实入口是 third_party/VLMEvalKit/run.py
#   2. `--model-path` **不是 run.py 的参数**。本地权重必须走 config JSON
#      （由 scripts/eval/make_vlmeval_config.py 生成）
#   3. `--data MathVista` 未注册 —— 正确键是 **MathVista_MINI**
#
# 评测纪律 (PLAN D5 / §9.2):
#   四个模型必须用同一工具链、同一 prompt 模板、同一抽帧策略, 否则分数不可比。
#
# 用法:
#   bash scripts/eval/run_all.sh                    # 全部已就绪的模型
#   MODELS="pretrained sft" DATA="MME" bash scripts/eval/run_all.sh
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6}"
export PYTHONUNBUFFERED=1
VENV="$PROJECT_ROOT/.venv-eval"
export PATH="$VENV/bin:$PATH"                 # 见问题 P20

VLMEVAL_ROOT="$PROJECT_ROOT/third_party/VLMEvalKit"
# VLMEvalKit 的数据根（标注 TSV + video/ 目录都在这里）
export LMUData="${LMUData:-$PROJECT_ROOT/data/raw/lmudata}"

# VideoMME 从 HF 缓存里找 lmms-lab/Video-MME 仓库。
# 本机用独立缓存目录 + 符号链接指向已下载的数据（避免与训练共用缓存互相干扰）。
export HF_HOME="${VLMEVAL_HF_HOME:-$PROJECT_ROOT/.cache/hf-eval}"
# ★ 只评测 3 分钟以内的视频（用户要求）。short 档 = <2min ⊂ <3min。
#   不设此变量则评测全部 900 个视频（需 ~101 GB 视频，本机只有 short 子集）。
export VLMEVALKIT_VMME_DURATION="${VLMEVALKIT_VMME_DURATION:-short}"

MODELS="${MODELS:-pretrained sft dpo grpo}"
DATA="${DATA:-MME MathVista_MINI Video-MME_16frame}"
OUT="$PROJECT_ROOT/results/eval"
CFG="$PROJECT_ROOT/configs/vlmeval_config.json"

ts()  { date +%H:%M:%S; }
log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(ts)" "$*"; }
warn(){ printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }

mkdir -p "$OUT" "$LMUData"

# ---- 前置: 导入自检 ---------------------------------------------------------
log "环境自检"
if ! "$VENV/bin/python" -c "import sys; sys.path.insert(0,'$VLMEVAL_ROOT'); import vlmeval" 2>/dev/null; then
  warn "vlmeval 无法导入 —— 先修 .venv-eval（缺 libGL/依赖）。见 docs/05 问题 P29"
  exit 1
fi
if [[ ! -f "$VLMEVAL_ROOT/run.py" ]]; then
  warn "找不到 VLMEvalKit 入口: $VLMEVAL_ROOT/run.py"; exit 1
fi

# ---- 逐数据集评测 -----------------------------------------------------------
# ⚠️ VLMEvalKit 的 run.py 里 `--config` 与 `--data` **互斥**:
#     assert args.data is None and args.model is None
#     => 用 --config 时, 要跑什么都由 config 里的 data 段决定。
#     所以这里**每个数据集生成一份单独的 config**, 而不是传 --data。
FAILED=()
for d in $DATA; do
  work="$OUT/$d"
  if [[ -f "$work/DONE" ]]; then log "已完成, 跳过: $d"; continue; fi
  mkdir -p "$work"
  cfg_d="$PROJECT_ROOT/configs/vlmeval_${d}.json"
  if ! "$VENV/bin/python" scripts/eval/make_vlmeval_config.py \
        --datasets "$d" --out "$cfg_d" >/dev/null; then
    warn "跳过 $d: 生成配置失败（可能没有可用模型目录）"; FAILED+=("$d/无模型"); continue
  fi
  log "评测数据集: $d"
  if "$VENV/bin/python" "$VLMEVAL_ROOT/run.py" \
        --config "$cfg_d" \
        --work-dir "$work" \
        --mode all 2>&1 | tee "$work/run.log"; then
    touch "$work/DONE"; log "✅ $d 完成"
  else
    warn "❌ $d 失败 (日志: $work/run.log)"; FAILED+=("$d")
  fi
done

# ---- 汇总 -------------------------------------------------------------------
log "生成报告"
"$VENV/bin/python" scripts/eval/collect_results.py || warn "汇总失败"

if (( ${#FAILED[@]} )); then
  echo; warn "以下数据集未成功: ${FAILED[*]}"
fi

echo
log "注意: 若 MathVista 无 OpenAI key, 则为规则判分 —— 报告中必须标注不可与论文比较 (K6)"

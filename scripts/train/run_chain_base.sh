#!/usr/bin/env bash
# ============================================================================
# base 线训练链：在**真·预训练基座**（Qwen2-VL-2B，无 -Instruct）上重跑 SFT → DPO
# ----------------------------------------------------------------------------
# 为什么要有它:
#   现有产物全是 "instruct 线"（官方已指令微调 → SFT → DPO → GRPO）。
#   用户要求**对照着再跑一条 base 线**，用于分析「官方指令微调带来了什么、
#   我们的后训练在两种起点上分别带来什么」。
#
# 与 instruct 线的差异（**只差这些**，其余超参逐行相同）:
#   SFT : model_name_or_path = models/Qwen2-VL-2B       (不是 -Instruct)
#         output_dir         = results/checkpoints/base_sft
#   DPO : model_name_or_path = results/checkpoints/base_sft
#         output_dir         = results/checkpoints/base_dpo
#
# ✅ 训练数据与 instruct 线**完全一致**：base 与 instruct 的 tokenizer 逐字节相同
#    （vocab.json / merges.txt / tokenizer.json 三个 md5 全一致，实测），
#    所以两条线共用同一份 tokenized 数据（data/cache/sft_tok）—— 受控对照。
#    ⇒ 也不会付那 4 小时的 tokenize。
#
# ⚠️ 显存/内存沿用 instruct 线的保命配置（batch 2 / accum 12 / worker 1 / 不存周期
#    checkpoint）—— 见 docs/05 P36/P37/P42，别改。
#
# 用法（会先等 GPU 上的评测跑完）:
#   setsid nohup bash scripts/train/run_chain_base.sh > logs/chain_base.log 2>&1 < /dev/null &
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
LOG="$PROJECT_ROOT/logs"
CKPT="$PROJECT_ROOT/results/checkpoints"

ts()   { date +"%Y-%m-%d %H:%M:%S"; }
ok()   { printf '\033[1;32m[%s] ✅ %s\033[0m\n' "$(ts)" "$*"; }
bad()  { printf '\033[1;31m[%s] ❌ %s\033[0m\n' "$(ts)" "$*"; }
info() { printf '\033[1;34m[%s]\033[0m %s\n' "$(ts)" "$*"; }

run_stage() {   # run_stage <名字> <配置> <输出目录> <日志>
  local name="$1" cfg="$2" out="$3" logfile="$4"
  info "启动 $name | 配置 $cfg | 输出 $out"
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC=7 FORCE_TORCHRUN=1 OMP_NUM_THREADS=8 \
      TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=1 \
      PATH="$PROJECT_ROOT/.venv-lf/bin:$PATH" \
      PYTHONPATH="$PROJECT_ROOT/third_party/LlamaFactory/src" \
      "$PROJECT_ROOT/.venv-lf/bin/llamafactory-cli" train "$cfg" \
      > "$logfile" 2>&1
}

have_model() { [[ -f "$1/model.safetensors" ]]; }

# ---------------------------------------------------------------------------
# 0. 路径互斥校验 —— 硬门槛, 宁可不跑也不能覆盖已有结果
# ---------------------------------------------------------------------------
# 用户明确要求: base 线绝不能与 instruct 线路径重叠（否则之前跑的结果被覆盖）。
# 这里不只是"检查一下", 而是**不等就退出**。
assert_disjoint() {
  local new="$1" old="$2"
  if [[ "$new" == "$old" || "$new" == "$old"/* ]]; then
    bad "路径冲突: base 线要写 '$new'，而它落在 instruct 线 '$old' 之内！拒绝启动。"
    exit 1
  fi
  [[ -e "$new" && "$new" == *sft* && -f "$new/model.safetensors" ]] && \
    info "注意: $new 已存在模型产物，将跳过该阶段（不会覆盖）"
}
assert_disjoint "$CKPT/base_sft"  "$CKPT/sft"
assert_disjoint "$CKPT/base_dpo"  "$CKPT/dpo"
assert_disjoint "$CKPT/base_grpo" "$CKPT/grpo"
assert_disjoint "$CKPT/base_sft"  "$CKPT/sft_ep3"     # ep3 那条也要护住
ok "路径互斥校验通过（base_sft / base_dpo / base_grpo 与已有产物完全隔离）"

# ---------------------------------------------------------------------------
# 0b. 等 GPU **真正**空出来（评测/别的训练在跑就等）
# ---------------------------------------------------------------------------
# ★ 2026-09-29 事故与修法:
#   原来只查"这一刻有没有进程"。实测被抢跑 —— vlmeval 重启后有 ~1 分钟启动间隙
#   (03:04:11 重启, 03:04:31 这里就判定"空闲"), base SFT 与评测同时占了 7 张卡,
#   显存/CPU 互相挤, 有 CUDA OOM 风险。
#   ⇒ 现在要求**连续 3 次(6 分钟)都查不到进程**才认为空闲, 才不会被切换/启动间隙骗到。
info "等待 GPU 空闲（评测或别的训练跑完）..."
idle_rounds=0
while (( idle_rounds < 3 )); do
  if pgrep -f "VLMEvalKit/run.p[y]" >/dev/null || pgrep -f "llamafactory/launcher.p[y]" >/dev/null; then
    idle_rounds=0
  else
    idle_rounds=$((idle_rounds + 1))
    info "GPU 空闲确认 $idle_rounds/3（需连续 3 次 = 6 分钟，防启动间隙误判）"
  fi
  sleep 120
done
ok "GPU 确实空闲，开始 base 线"

# ---------------------------------------------------------------------------
# 0c. base 模型的评测（**排在 SFT 之前**）
# ---------------------------------------------------------------------------
# 为什么放最前: 只要 1 个模型 × 3 个 benchmark（约 1–2 小时），而 SFT 要 11 小时。
# 先把"真·预训练基座"那一行补齐，报告才是完整对照表。
# MODELS 只有 base ⇒ 生成的 config 与别的跑批不同 ⇒ DONE 标记名不同（按模型集哈希），
# 既不会命中、也不会干扰 instruct 线那批已有结果。
if compgen -G "$PROJECT_ROOT/results/eval/*/base/*/*_score*" >/dev/null; then
  ok "base 评测已有分数，跳过"
else
  info "评测 base 模型（MME / MathVista / Video-MME，约 1–2 小时）"
  env MODELS="base" DATA="MME MathVista_MINI Video-MME_16frame" \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}" \
      bash scripts/eval/run_all.sh > "$LOG/chain_base_eval.log" 2>&1 \
    || info "base 评测有失败项（见 logs/chain_base_eval.log），继续往下跑 SFT"
  ok "base 评测结束 -> results/eval/*/base/"
fi

# ---------------------------------------------------------------------------
# 1. base SFT（2 epoch，与 instruct 线同配置）
# ---------------------------------------------------------------------------
if have_model "$CKPT/base_sft"; then
  ok "base_sft 已存在，跳过"
else
  run_stage "base SFT" configs/sft_qwen2vl_2b_base_7gpu.yaml "$CKPT/base_sft" "$LOG/chain_base_sft.log" \
    || { bad "base SFT 失败 —— 见 logs/chain_base_sft.log"; exit 1; }
  have_model "$CKPT/base_sft" || { bad "base SFT 跑完但没有 model.safetensors"; exit 1; }
  ok "base SFT 完成 -> $CKPT/base_sft"
fi

# ---------------------------------------------------------------------------
# 2. base DPO（1 epoch）
# ---------------------------------------------------------------------------
if have_model "$CKPT/base_dpo"; then
  ok "base_dpo 已存在，跳过"
else
  run_stage "base DPO" configs/dpo_qwen2vl_2b_base_7gpu.yaml "$CKPT/base_dpo" "$LOG/chain_base_dpo.log" \
    || { bad "base DPO 失败 —— 见 logs/chain_base_dpo.log"; exit 1; }
  have_model "$CKPT/base_dpo" || { bad "base DPO 跑完但没有 model.safetensors"; exit 1; }
  ok "base DPO 完成 -> $CKPT/base_dpo"
  info "DPO 健康指标 (rewards/accuracies 健康区间 0.5-0.8):"
  grep -aoE "rewards/accuracies['\"]?[: ]+[0-9.]+" "$LOG/chain_base_dpo.log" | tail -3 | sed 's/^/    /' || true
fi

ok "base 线（SFT + DPO）完成。GRPO 待单独启动（先校准步时）。"

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
  mkdir -p "$work"
  cfg_d="$PROJECT_ROOT/configs/vlmeval_${d}.json"
  # ★ 2026-09-29 修: 原来**没传 --models** ⇒ make_vlmeval_config.py 默认"所有存在的模型目录"，
  #   于是 run_all.sh 顶部的 MODELS 变量形同虚设 —— 想只评测 base 一个模型也会把 5 个全跑一遍
  #   （8+ 小时），而且模型集一变 DONE 标记名也变。
  if ! "$VENV/bin/python" scripts/eval/make_vlmeval_config.py \
        --datasets "$d" --models $MODELS --out "$cfg_d" >/dev/null; then
    warn "跳过 $d: 生成配置失败（可能没有可用模型目录）"; FAILED+=("$d/无模型"); continue
  fi

  # ★ 2026-09-29 修（两个 bug，都会让评测**静默地不产出**）：
  #   bug1 原来的 DONE 只看数据集名（$work/$d）⇒ **换模型集也会整块跳过**。
  #        实测事故：09-23 的 pretrained-only Video-MME 留了 DONE，
  #        导致 09-28 的 sft/dpo/grpo Video-MME 一个都没跑（日志只有一行"已完成, 跳过"）。
  #        ⇒ 现在把标记名绑到**模型配置的内容哈希**：模型集一变，标记名就变，会重跑。
  #   bug2 原来只要 run.py 退 0 就打 DONE，而 MathVista 判分 100% 失败时它照样退 0
  #        ⇒ 修好判分后也会被跳过。现在**必须真的产出 *_score* 文件**才算完成。
  SLUG=$(md5sum "$cfg_d" | cut -c1-8)
  DONE_FILE="$work/DONE.$SLUG"
  if [[ -f "$DONE_FILE" ]]; then log "已完成(同模型集 $SLUG), 跳过: $d"; continue; fi

  log "评测数据集: $d  (模型集 ${SLUG})"
  # ★ 2026-09-29 加: 判分模型的显式指定。
  #   背景: vlmeval 的默认判分模型是 gpt-4o-mini（videomme.py:61 DEFAULT_JUDGE_MODEL），
  #   而本项目可用的 relay（xmapi）上**没有 gpt-4 系列**，只有 gpt-5.x / gpt-6
  #   ⇒ 必须显式指定，否则 judge_fail_rate 100%（实测 MathVista 1000/1000 全失败）。
  #   密钥走 third_party/VLMEvalKit/.env（third_party/ 在 .gitignore 里，不会进仓库）。
  #   JUDGE_MODEL 可用环境变量覆盖；传 exact_matching 则用纯规则判分（不调 API）。
  # ★ 只传 --judge（模型名）。**不要**传 --judge-base-url：
  #   run.py:396 是 `f"{judge_base_url.rstrip('/')}/chat/completions"` —— 它会自己拼后缀，
  #   而 OPENAI_API_BASE 约定的是**完整端点**（见 vlmeval/api/gpt.py:16 的 APIBASES['OFFICIAL']）。
  #   两者同时用会拼成 .../chat/completions/chat/completions（实测报 JSONDecodeError）。
  #   ⇒ 端点与密钥统一走 third_party/VLMEvalKit/.env，这里只覆盖模型名。
  JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.5}"
  JUDGE_ARGS=(--judge "$JUDGE_MODEL")
  log "判分模型: $JUDGE_MODEL（端点/密钥取自 VLMEvalKit/.env）"
  if "$VENV/bin/python" "$VLMEVAL_ROOT/run.py" \
        --config "$cfg_d" \
        --work-dir "$work" \
        "${JUDGE_ARGS[@]}" \
        --mode all 2>&1 | tee "$work/run.log"; then
    # 产物形如 $work/<model>/T<时间戳>/<model>_<bench>_score*.csv|json
    # ★ 2026-09-29 修: 原来用 "$work/*/*_score*" —— 会被**别的模型的旧分数文件**骗过
    #   （扁平化的历史产物就在 <bench>/<model>/ 下），于是跑崩了也照样打 ✅。
    #   现在只看"这一轮真正写出的最新运行目录"。
    # ★ 2026-09-30 修: 必须带尾斜杠只匹配**目录** —— 否则会匹配到 vlmeval 的
    #   日志文件 logs/T<时间戳>_....log（它也是 T 开头），于是永远找不到分数 ✗
    newest_run=$(ls -1dt "$work"/*/T*/ 2>/dev/null | head -1)
    [[ -n "$newest_run" ]] && newest_run="${newest_run%/}"
    if [[ -n "$newest_run" ]] && compgen -G "$newest_run/*_score*" >/dev/null; then
      touch "$DONE_FILE"; log "✅ $d 完成"
    else
      warn "❌ $d 跑完了但没有任何 *_score* 产物（判分可能全失败）—— 不打 DONE，可重跑"
      FAILED+=("$d/无分数")
    fi
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

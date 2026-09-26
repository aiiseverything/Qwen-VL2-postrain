#!/usr/bin/env bash
# ============================================================================
# 视频轨 —— 与图像轨**互不阻塞**
# ----------------------------------------------------------------------------
# 步骤: 探测时长 → 筛选 <3min → 下载视频 → 回填 SFT 数据 → 重新去污染+统计
#
# 为什么要独立成一条轨 (问题 P18):
#   初版把视频下载放在 SFT 抽取之前, 2,864 个视频要下 18 小时, 把关键路径堵死。
#   视频只影响 SFT 数据里 10 万条中的一部分, 不该阻塞图像侧的数据准备。
#
# 用法:
#   setsid nohup env CUDA_VISIBLE_DEVICES="" bash scripts/run_video_pipeline.sh \
#     > logs/video_pipeline.log 2>&1 < /dev/null &
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
PY="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"; mkdir -p "$LOG_DIR"

MAX_MIN="${MAX_MIN:-3}"        # 用户要求: 低于 3 分钟
WORKERS="${WORKERS:-6}"

ts()   { date +%H:%M:%S; }
step() { printf '\n\033[1;36m[%s] ===== %s =====\033[0m\n' "$(ts)" "$*"; }
ok()   { printf '\033[1;32m[%s] ✅ %s\033[0m\n' "$(ts)" "$*"; }
bad()  { printf '\033[1;31m[%s] ❌ %s\033[0m\n' "$(ts)" "$*"; }

FAILED=()
run_step() { local n="$1"; shift; step "$n"; if "$@"; then ok "$n"; else bad "$n"; FAILED+=("$n"); fi; }

step "视频轨启动 (仅下载 <${MAX_MIN} 分钟的视频, ${WORKERS} 并发)"
printf '\033[1;34m[%s]\033[0m CUDA_VISIBLE_DEVICES=%s\n' "$(ts)" "'$CUDA_VISIBLE_DEVICES'"

# 阶段 1: 探测 (--reprobe-errors 会用 android 客户端重试此前 error/timeout 的)
# SKIP_PROBE=1 可跳过探测直接下载 —— 适用于已经重探过、剩下的失败者
# 确实是用 android 客户端也拿不到的情况, 再探一次收益很低。
if [[ "${SKIP_PROBE:-0}" == "1" ]]; then
  ok "阶段 1/4: 跳过探测 (SKIP_PROBE=1), 直接使用已有 kept_videos.json"
else
  run_step "阶段 1/4: 探测视频时长" \
    "$PY" scripts/prepare/fetch_videos.py \
      --max-minutes "$MAX_MIN" --workers "$WORKERS" --stage probe

  run_step "阶段 1b/4: 用 android 客户端重探失败的" \
    "$PY" scripts/prepare/fetch_videos.py \
      --max-minutes "$MAX_MIN" --workers "$WORKERS" --stage probe --reprobe-errors
fi

# 阶段 2: 下载 (已存在的文件会被跳过, 可反复跑)
run_step "阶段 2/4: 下载 <${MAX_MIN}min 视频" \
  "$PY" scripts/prepare/fetch_videos.py \
    --max-minutes "$MAX_MIN" --workers "$WORKERS" --stage download

# 阶段 3: 回填视频样本进 SFT 数据
step "阶段 3/4: 回填视频样本到 SFT"
if [[ -f data/processed/sft_400k.jsonl ]]; then
  mv -f data/processed/sft_400k.jsonl data/processed/sft_400k.imgonly.jsonl
  "$PY" scripts/prepare/to_sharegpt.py --out data/processed/sft_400k.jsonl \
    2>&1 | tail -20 && ok "回填完成" || { bad "回填失败"; FAILED+=("回填"); }
else
  bad "缺少 sft_400k.jsonl —— 请先跑图像轨 scripts/run_pipeline.sh"
  FAILED+=("回填(前置缺失)")
fi

# 阶段 4: 重新去污染 + 统计
run_step "阶段 4/4: 重新去污染" \
  "$PY" scripts/prepare/quality_filter.py \
    --in data/processed/sft_400k.jsonl --out data/processed/sft_400k.clean.jsonl

run_step "收尾: 统计报告" "$PY" scripts/prepare/stats.py

step "视频轨结束 —— 汇总"
echo "  已下载片段: $(ls data/raw/videos/clips/*.mp4 2>/dev/null | wc -l)"
if [[ -f data/raw/videos/kept_videos.json ]]; then
  echo "  通过时长筛选: $("$PY" -c "import json;print(len(json.load(open('data/raw/videos/kept_videos.json'))))")"
fi
if (( ${#FAILED[@]} )); then
  echo; bad "以下步骤失败:"; for f in "${FAILED[@]}"; do echo "    - $f"; done
else
  echo; ok "视频轨全部成功"
fi

#!/usr/bin/env bash
# ============================================================================
# 数据流水线 (图像轨) —— 无人值守
# ----------------------------------------------------------------------------
# 历史教训: 初版把「视频下载」排在 SFT 抽取**之前**, 结果 2,864 个视频要下
# 18 小时, 把后面的数据库准备全堵死 (见 docs/05 问题 P18)。
# 现在拆成两条**互不阻塞**的轨:
#   * 本脚本 = 图像轨 (关键路径): 等图像 → 解压 → 抽取 400K → 去污染 → 统计
#   * scripts/run_video_pipeline.sh = 视频轨 (独立): 探测 → 下载 → 回填视频样本
#
# 用法:
#   setsid nohup env CUDA_VISIBLE_DEVICES="" bash scripts/run_pipeline.sh \
#     > logs/pipeline.log 2>&1 < /dev/null &
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""          # 绝不占用显存
export PYTHONUNBUFFERED=1               # 否则重定向到文件时日志会成块缓冲, 看着像卡死
LOG_DIR="$PROJECT_ROOT/logs"; mkdir -p "$LOG_DIR"
PY="$PROJECT_ROOT/.venv/bin/python"

ts()   { date +%H:%M:%S; }
step() { printf '\n\033[1;36m[%s] ===== %s =====\033[0m\n' "$(ts)" "$*"; }
ok()   { printf '\033[1;32m[%s] ✅ %s\033[0m\n' "$(ts)" "$*"; }
bad()  { printf '\033[1;31m[%s] ❌ %s\033[0m\n' "$(ts)" "$*"; }
info() { printf '\033[1;34m[%s]\033[0m %s\n' "$(ts)" "$*"; }

FAILED=()
run_step() { local n="$1"; shift; step "$n"; if "$@"; then ok "$n"; else bad "$n"; FAILED+=("$n"); fi; }
count_jpg() { [[ -d "$1" ]] && find "$1" -maxdepth 1 -name '*.jpg' | wc -l || echo 0; }

step "图像轨启动"
info "CUDA_VISIBLE_DEVICES='' —— 不占显存, 不影响 project-1 的训练"

# ---------------------------------------------------------------------------
# 阶段 1: 补齐并解压图像 (含 ZIP64 安全的完整性校验)
# ---------------------------------------------------------------------------
step "阶段 1/4: 补齐 + 解压图像归档"
bash scripts/download/resume_all.sh 2>&1 | sed 's/^/    /' || true
run_step "解压图像" "$PY" scripts/download/01c_extract_images.py

for d in coco/train2017 gqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2; do
  printf "    %-26s %8s 张\n" "$d" "$(count_jpg "data/raw/images/$d")"
done

# ---------------------------------------------------------------------------
# 阶段 2: SFT 抽取 (图像部分; 视频样本由视频轨回填)
# ---------------------------------------------------------------------------
if [[ ! -f data/processed/sft_400k.jsonl ]]; then
  run_step "阶段 2/4: SFT 抽取" \
    "$PY" scripts/prepare/to_sharegpt.py --out data/processed/sft_400k.jsonl
else
  ok "阶段 2/4: sft_400k.jsonl 已存在, 跳过"
fi

# ---------------------------------------------------------------------------
# 阶段 3: 质量过滤 + 去污染
# ---------------------------------------------------------------------------
if [[ ! -f data/processed/sft_400k.clean.jsonl ]]; then
  run_step "阶段 3/4: 质量过滤 + 去污染" \
    "$PY" scripts/prepare/quality_filter.py \
      --in data/processed/sft_400k.jsonl --out data/processed/sft_400k.clean.jsonl
else
  ok "阶段 3/4: sft_400k.clean.jsonl 已存在, 跳过"
fi

# ---------------------------------------------------------------------------
# 阶段 4: 统计报告
# ---------------------------------------------------------------------------
run_step "阶段 4/4: 统计报告" "$PY" scripts/prepare/stats.py

# ---------------------------------------------------------------------------
step "图像轨结束 —— 汇总"
for f in data/processed/sft_400k.jsonl data/processed/sft_400k.clean.jsonl \
         data/processed/dpo_10k.jsonl data/manifests/stats_report.md; do
  if [[ -f "$f" ]]; then printf "    ✅ %-46s %s\n" "$f" "$(du -h "$f" | cut -f1)"
  else printf "    ❌ %-46s 缺失\n" "$f"; fi
done
if (( ${#FAILED[@]} )); then
  echo; bad "以下步骤失败:"; for f in "${FAILED[@]}"; do echo "    - $f"; done
else
  echo; ok "图像轨全部成功"
fi
echo
echo "  ★ 必看: data/manifests/stats_report.md"
echo "  ★ 视频轨独立运行: bash scripts/run_video_pipeline.sh"

#!/usr/bin/env bash
# ============================================================================
# 断点续传总控 —— 幂等, 可反复执行
# ----------------------------------------------------------------------------
# 用途: 终端断开 / 机器重启 / 下载中断后, 一条命令把**尚未完成**的下载
#       重新挂到后台(完全脱离终端)继续跑。
#
# 用法:
#   bash scripts/download/resume_all.sh          # 只补未完成的, 已完成的跳过
#   bash scripts/download/resume_all.sh --force  # 强制重启全部脚本
#   bash scripts/download/resume_all.sh --status # 只看状态, 不启动任何东西
#
# 设计: 每个下载脚本自身都已幂等 (fetch_url 比对期望字节数后跳过,
#       hf download 自带断点续传), 所以重复执行是安全的。
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs"; mkdir -p "$LOG_DIR"

FORCE=0; STATUS_ONLY=0
for a in "$@"; do
  [[ "$a" == "--force" ]] && FORCE=1
  [[ "$a" == "--status" ]] && STATUS_ONLY=1
done

gb() { awk -v s="$1" 'BEGIN{printf "%.2f GB", s/1e9}'; }
size() { stat -c%s "$1" 2>/dev/null || echo 0; }

# 每个脚本对应的"完成判据"函数
done_model()  { [[ -f models/Qwen2-VL-2B-Instruct/model-00001-of-00002.safetensors ]]; }
done_images() {
  [[ -f data/raw/images/coco_train2017.zip ]] || return 1
  # COCO 期望 19,336,861,798 字节
  [[ "$(size data/raw/images/coco_train2017.zip)" -ge 19336861798 ]] || return 1
  [[ -d data/raw/images/vg ]] || return 1
  return 0
}
# 注: RLHF-V 是 parquet; LLaVA-RLHF-Data 提供的是 **JSON**(非 parquet),
#     所以这里两类文件都要认, 否则会误判为未完成而反复重下。
done_dpo()   { [[ -n "$(find data/raw/dpo/RLHF-V-Dataset  -name '*.parquet' -print -quit 2>/dev/null)" ]] \
                && [[ -n "$(find data/raw/dpo/LLaVA-RLHF-Data -name '*.json'    -print -quit 2>/dev/null)" ]]; }
done_rlvr()  { [[ -n "$(find data/raw/rlvr/geometry3k -name 'train-*.parquet' -print -quit 2>/dev/null)" ]] \
                && [[ -n "$(find data/raw/rlvr/geoqa_r1v   -name 'train-*.parquet' -print -quit 2>/dev/null)" ]] \
                && [[ -n "$(find data/raw/rlvr/mmk12      -name 'train-*.parquet' -print -quit 2>/dev/null)" ]]; }
done_bench() { [[ -n "$(find data/raw/bench/MME       -name '*.parquet' -print -quit 2>/dev/null)" ]] \
                && [[ -n "$(find data/raw/bench/MathVista -name '*.parquet' -print -quit 2>/dev/null)" ]] \
                && [[ -n "$(find data/raw/bench/Video-MME -name '*.parquet' -print -quit 2>/dev/null)" ]]; }

show_status() {
  echo "=== Postrain 下载状态 ==="
  for pair in "00_model:模型 Qwen2-VL-2B" "01b_images:图像归档(COCO/GQA/TextVQA/VG)" \
              "02_dpo:DPO 偏好数据" "03_rlvr:RLVR 数据" "04_bench:评测数据"; do
    s="${pair%%:*}"; label="${pair#*:}"
    printf "  %-34s " "$label"
    case "$s" in
      00_model)  done_model  && echo "✅ 完成" || echo "⏳ 未完成" ;;
      01b_images) done_images && echo "✅ 完成" || echo "⏳ 未完成" ;;
      02_dpo)    done_dpo    && echo "✅ 完成" || echo "⏳ 未完成" ;;
      03_rlvr)   done_rlvr   && echo "✅ 完成" || echo "⏳ 未完成" ;;
      04_bench)  done_bench  && echo "✅ 完成" || echo "⏳ 未完成" ;;
    esac
  done
  echo
  echo "--- 磁盘 ---"; df -h "$PROJECT_ROOT" | tail -1
  echo "--- 正在运行的下载脚本 ---"
  # ps -eo pid,etimes,args -> $1=pid $2=etimes $3=bash $4=script
  ps -eo pid,etimes,args --no-headers \
    | awk '$3=="bash" && $4 ~ /^scripts\/download\//{printf "  pid=%-8s %ss  %s\n",$1,$2,$4}' || true
  echo "--- 图像归档字节数 (期望: coco 19.34G / gqa 21.82G / textvqa 7.07G) ---"
  for f in data/raw/images/*.zip; do [[ -f "$f" ]] && printf "  %-30s %s\n" "$(basename "$f")" "$(gb "$(size "$f")")"; done
  echo "--- 模型 ---"
  for f in models/Qwen2-VL-2B-Instruct/model-*.safetensors; do [[ -f "$f" ]] && printf "  %-40s %s\n" "$(basename "$f")" "$(gb "$(size "$f")")"; done
}

if [[ "$STATUS_ONLY" == "1" ]]; then show_status; exit 0; fi

launch() {   # launch <script> <done_fn>
  local script="$1" fn="$2"
  if [[ "$FORCE" != "1" ]] && "$fn"; then
    echo "  [skip] $script  (已完成)"; return 0
  fi
  # ⚠️ 必须比对**完整路径**。`ps -eo args` 下 $1=bash, $2=scripts/download/<x>.sh。
  #    早先误写成 `$3==$script`(只比文件名), 永远为假, 导致重复实例同时写
  #    同一个文件 —— 两个 curl 争抢同一 .partial 会静默损坏数据。
  if ps -eo args --no-headers | awk -v s="scripts/download/$script" \
       '$1=="bash" && $2==s{f=1} END{exit !f}'; then
    echo "  [run ] $script  (已在运行中, 不重复启动)"; return 0
  fi
  echo "  [start] $script"
  setsid nohup env CUDA_VISIBLE_DEVICES="" PROJECT_ROOT="$PROJECT_ROOT" \
    bash "scripts/download/$script" > "$LOG_DIR/dl_${script%.sh}.log" 2>&1 < /dev/null &
  sleep 1
}

echo "=== 补齐未完成的下载 (完全脱离终端) ==="
launch 00_model.sh   done_model
launch 01b_images.sh done_images
launch 02_dpo.sh     done_dpo
launch 03_rlvr.sh    done_rlvr
launch 04_bench.sh   done_bench
echo
show_status
echo
echo "日志: logs/dl_*.log     状态: bash scripts/download/resume_all.sh --status"

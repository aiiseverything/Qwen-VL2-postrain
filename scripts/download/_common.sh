#!/usr/bin/env bash
# ============================================================================
# 公共下载函数库 —— 所有 download/*.sh 均 source 本文件
# ----------------------------------------------------------------------------
# 设计约束 (来自用户要求 + PLAN):
#   1. **绝不占用显存、绝不调用 CUDA** —— 本文件不加载任何 torch/CUDA。
#   2. **绝不写入 project-1/** —— 所有路径限定在 PROJECT_ROOT 内。
#   3. 下载并发受限，避免 I/O 抢占正在运行的训练进程。
#   4. 断点续传 + 前置空间检查 (PLAN K10)。
# ============================================================================

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ml-user/workdir/project-2}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

RAW_DIR="${DATA_ROOT}/raw"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${RAW_DIR}" "${MODEL_ROOT}" "${LOG_DIR}"

# 镜像一键切换: 默认官方源; HF_ENDPOINT=https://hf-mirror.com 可切镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/hf}"

# 对外部训练进程友好: 限制单流带宽不设, 但限制并发进程数
MAX_PARALLEL="${MAX_PARALLEL:-3}"

PY="${PROJECT_ROOT}/.venv/bin/python"
HF="${PROJECT_ROOT}/.venv/bin/hf"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 安全停止 (问题 P4 的教训) -----------------------------------------------
# ⚠️ 切勿用 `pkill -f "<脚本名>"` 停止下载: 该模式会同时匹配到调用它的
#    外层 shell 自身的命令行, 从而把执行命令的 shell 一起杀掉 (实测 exit 144)。
#    正确做法: 记录 PID 到 run 目录, 按 PID 精确 kill。
RUN_DIR="${PROJECT_ROOT}/.run"; mkdir -p "$RUN_DIR"
record_pid() { echo "$$" > "${RUN_DIR}/$(basename "$0").pid"; }
stop_by_pidfile() {
  local pf="${RUN_DIR}/$1.pid"
  [[ -f "$pf" ]] || { warn "无 pidfile: $pf"; return 1; }
  local pid; pid=$(cat "$pf")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" && log "已停止 $1 (pid $pid)"
    # 同时清理其 curl 子进程, 只匹配本项目的路径前缀
    for c in $(ps -eo pid,args | grep -F "data/raw/" | grep -F "[c]url" | awk '{print $1}'); do
      kill -TERM "$c" 2>/dev/null
    done
  else
    warn "pid $pid 不存在"
  fi
}

# --- 归档完整性校验 (防止跨源续传导致的损坏, 问题 P5) ------------------------
verify_zip() {
  local f="$1"
  [[ -f "$f" ]] || { warn "缺失: $f"; return 1; }
  if unzip -tq "$f" >/dev/null 2>&1; then
    log "完整性 OK: $(basename "$f")"; return 0
  else
    warn "归档损坏 (需删除重下): $f"; return 1
  fi
}

# --- 空间检查: 不足则退出, 不半途爆盘 (PLAN K10) ----------------------------
require_space() {
  local need_gb="$1" path="${2:-$PROJECT_ROOT}"
  local avail_gb
  avail_gb=$(df -BG --output=avail "$path" | tail -1 | tr -dc '0-9')
  if (( avail_gb < need_gb )); then
    die "空间不足: 需要 ${need_gb}GB, 仅剩 ${avail_gb}GB (路径 $path)"
  fi
  log "空间检查通过: 需要 ${need_gb}GB, 剩余 ${avail_gb}GB"
}

# --- 可续传下载 --------------------------------------------------------------
# 用法: fetch_url <url> <输出文件> [期望字节数]
fetch_url() {
  local url="$1" out="$2" expect="${3:-0}"
  mkdir -p "$(dirname "$out")"

  if [[ -f "$out" && "$expect" -gt 0 ]]; then
    local have; have=$(stat -c%s "$out")
    if [[ "$have" -eq "$expect" ]]; then
      log "已完整, 跳过: $(basename "$out") (${have} bytes)"; return 0
    fi
  fi

  log "下载: $url"
  curl -fL -C - --retry 5 --retry-delay 5 --retry-connrefused \
       --connect-timeout 30 -o "$out" "$url" \
    || { warn "下载失败: $url"; return 1; }

  if [[ "$expect" -gt 0 ]]; then
    local have; have=$(stat -c%s "$out")
    [[ "$have" -eq "$expect" ]] || { warn "大小不符 ($have != $expect): $out"; return 1; }
  fi
  return 0
}

# --- HF 快照下载 (走 hf CLI, 自带断点续传) -----------------------------------
# 用法: fetch_hf <repo_id> <repo_type:model|dataset> <local_dir> [include...]
fetch_hf() {
  local repo="$1" rtype="$2" dest="$3"; shift 3
  mkdir -p "$dest"
  local args=(download "$repo" --repo-type "$rtype" --local-dir "$dest")
  for inc in "$@"; do args+=(--include "$inc"); done
  "$HF" "${args[@]}" 2>&1 | sed 's/^/    /'
}

fetch_hf_parquet() {  # 便捷: 下载整个 dataset 仓库
  local repo="$1" dest="$2"
  fetch_hf "$repo" dataset "$dest"
}

# --- 并发闸门: 同时最多 MAX_PARALLEL 个后台任务 ------------------------------
throttle() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do sleep 5; done
}

wait_all() {
  local rc=0
  for pid in $(jobs -p); do wait "$pid" || rc=1; done
  return $rc
}

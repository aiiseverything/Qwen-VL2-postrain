#!/usr/bin/env bash
# 下载基座模型 Qwen/Qwen2-VL-2B-Instruct (~4.4 GB, 非 gated)
# 实测: HF API gated=false, downloads=1,989,324
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REPO="Qwen/Qwen2-VL-2B-Instruct"
DEST="${MODEL_ROOT}/Qwen2-VL-2B-Instruct"

require_space 10 "$PROJECT_ROOT"
log "下载基座模型: $REPO -> $DEST"

fetch_hf "$REPO" model "$DEST" \
  "*.json" "*.safetensors" "*.txt" "*.py" "*.model" "*.jinja"

log "模型下载完成。目录内容:"
ls -la "$DEST"
du -sh "$DEST"

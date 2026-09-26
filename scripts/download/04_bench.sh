#!/usr/bin/env bash
# 评测数据
#   MME         lmms-lab-encoder/MME    (旧 lmms-lab/MME 已 307 迁移) —— 全量 ~2GB
#   MathVista   AI4Math/MathVista       —— testmini 1000 题
#   Video-MME   lmms-eval/Video-MME     —— ⚠️ 仅标注; 视频数百 GB, 见 PLAN K7
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
record_pid

require_space 10 "$PROJECT_ROOT"
OUT="${RAW_DIR}/bench"; mkdir -p "$OUT"

log "下载评测数据"
fetch_hf "lmms-lab-encoder/MME" dataset "$OUT/MME"       || warn "MME 失败"
fetch_hf "AI4Math/MathVista"    dataset "$OUT/MathVista" || warn "MathVista 失败"
fetch_hf "lmms-eval/Video-MME"  dataset "$OUT/Video-MME" || warn "Video-MME 失败"

log "评测数据下载结束:"; du -sh "$OUT"/* 2>/dev/null
warn "Video-MME 视频本体未下载 (数百 GB)。上机前先 df -h (PLAN K7)。"

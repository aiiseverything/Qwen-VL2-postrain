#!/usr/bin/env bash
# DPO 偏好对数据 (目标 1 万条) —— 体量小, 本机全量下载
#   openbmb/RLHF-V-Dataset    ~5.7K   (旧的 HaoyeZhang/RLHF-V-Dataset 已 307 迁移至此)
#   zhiqings/LLaVA-RLHF-Data  ~4.3K
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
record_pid

require_space 15 "$PROJECT_ROOT"
OUT="${RAW_DIR}/dpo"; mkdir -p "$OUT"

log "下载 DPO 数据"
fetch_hf "openbmb/RLHF-V-Dataset"   dataset "$OUT/RLHF-V-Dataset"   || warn "RLHF-V 失败"
fetch_hf "zhiqings/LLaVA-RLHF-Data" dataset "$OUT/LLaVA-RLHF-Data"  || warn "LLaVA-RLHF 失败"

log "DPO 数据下载结束:"; du -sh "$OUT"/* 2>/dev/null

#!/usr/bin/env bash
# SFT 标注下载 (注意: 仅标注, 图像见 01b_images.sh —— 问题 P1)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
record_pid

require_space 5 "$PROJECT_ROOT"
OUT="${RAW_DIR}/annotations"; mkdir -p "$OUT"

log "下载 SFT 标注 JSON"
fetch_hf "liuhaotian/LLaVA-Instruct-150K" dataset "$OUT/LLaVA-Instruct-150K" \
  "llava_v1_5_mix665k.json" || warn "LLaVA 标注失败"
fetch_hf "Lin-Chen/ShareGPT4V" dataset "$OUT/ShareGPT4V" \
  "sharegpt4v_instruct_gpt4-vision_cap100k.json" \
  "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json" || warn "ShareGPT4V 标注失败"
fetch_hf "MBZUAI/VideoInstruct-100K" dataset "$OUT/VideoInstruct-100K" \
  "VideoInstruct100K.json" || warn "VideoInstruct 标注失败"

log "SFT 标注下载结束:"; du -sh "$OUT"/* 2>/dev/null

#!/usr/bin/env bash
# RLVR 可验证奖励数据 —— 供 GRPO 200 step 使用
#   hiyouga/geometry3k              2,101 train  (EasyR1 官方示例, 用于冒烟)
#   leonardPKU/GEOQA_R1V_Train_8K   ~8K          (主训练集)
#   FanqingM/MMK12                  多学科补充
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
record_pid

require_space 10 "$PROJECT_ROOT"
OUT="${RAW_DIR}/rlvr"; mkdir -p "$OUT"

log "下载 RLVR 数据"
fetch_hf "hiyouga/geometry3k"            dataset "$OUT/geometry3k"  || warn "geometry3k 失败"
fetch_hf "leonardPKU/GEOQA_R1V_Train_8K" dataset "$OUT/geoqa_r1v"   || warn "GEOQA 失败"
fetch_hf "FanqingM/MMK12"                dataset "$OUT/mmk12"       || warn "MMK12 失败"

log "RLVR 数据下载结束:"; du -sh "$OUT"/* 2>/dev/null

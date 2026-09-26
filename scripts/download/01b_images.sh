#!/usr/bin/env bash
# ============================================================================
# 下载 SFT 所需的**上游图像归档**
# ----------------------------------------------------------------------------
# ⚠️ 问题 P1 (见 docs/05): LLaVA-Instruct-150K / ShareGPT4V
#    **只含标注 JSON，不含图像本体**。图像必须从 COCO / GQA / TextVQA /
#    VisualGenome 等上游单独取回。PLAN §7.1 所称「全量图像 ~100 GB」即指此。
#
# ⚠️ 问题 P3 (见 docs/05): 学术原始站点在本机**极慢**。实测 30–50MB 区间下载速率:
#      images.cocodataset.org  (COCO)  0.05 MB/s   → 19.3GB 需 ~107 小时
#      downloads.cs.stanford.edu (GQA) 0.03 MB/s   → 21.8GB 需 ~198 小时
#      HF 镜像 pcuenq/coco-2017-mirror 4.15 MB/s   → 快 83 倍
#      HF 镜像 Feeky929/GQA-images     3.40 MB/s   → 快 113 倍
#    故 COCO / GQA 改用 HF 镜像; TextVQA 原始站点 1.49 MB/s 可接受, 保留原源。
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IMG_DIR="${RAW_DIR}/images"
mkdir -p "$IMG_DIR"

require_space 80 "$PROJECT_ROOT"

log "开始下载上游图像归档 (并发上限 ${MAX_PARALLEL}, 避免抢占训练 I/O)"

# COCO train2017 —— 19.34 GB —— 走 HF 镜像 (原始站点 0.05 MB/s, 不可用)
throttle; ( fetch_url \
  "https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/main/train2017.zip" \
  "$IMG_DIR/coco_train2017.zip" ) &

# GQA images —— 21.82 GB —— 走 HF 镜像 (原始站点 0.03 MB/s, 不可用)
throttle; ( fetch_url \
  "https://huggingface.co/datasets/Feeky929/GQA-images/resolve/main/images.zip" \
  "$IMG_DIR/gqa_images.zip" ) &

# TextVQA —— 7.07 GB —— 原始站点 1.49 MB/s, 可接受
throttle; ( fetch_url \
  "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
  "$IMG_DIR/textvqa_images.zip" 7072297970 ) &

wait_all || warn "部分图像归档下载失败, 重跑本脚本可断点续传"

# VisualGenome —— 官方源 visualgenome.org 本机不可达 (curl code=000), 走 HF 镜像
throttle; ( fetch_hf "BoyangZ/VisualGenome_VG_100K_1_and_2" dataset "$IMG_DIR/vg" \
              "images.zip" "images2.zip" ) &

wait_all || warn "VG 下载失败 (可单独重跑)"

log "图像归档下载结束。当前占用:"
du -sh "$IMG_DIR"/* 2>/dev/null
log "下一步: 解压 —— bash scripts/download/01c_extract_images.sh"

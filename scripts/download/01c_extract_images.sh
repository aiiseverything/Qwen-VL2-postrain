#!/usr/bin/env bash
# ============================================================================
# 解压上游图像归档 -> data/raw/images/{coco,gqa,textvqa,vg}
# ----------------------------------------------------------------------------
# 解压前做**完整性校验** (问题 P5): 跨源续传过的 zip 可能静默损坏,
# unzip -t 能在解压前就发现问题, 而不是解压到一半才炸。
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IMG="${RAW_DIR}/images"
DEST="${IMG}/extracted"; mkdir -p "$DEST"

check_and_extract() {
  local zip="$1" name="$2" target="$3"
  [[ -f "$zip" ]] || { warn "缺失跳过: $zip"; return 1; }
  verify_zip "$zip" || { die "归档损坏: $zip —— 删除后重跑 01b_images.sh"; }
  [[ -d "$target" ]] && { log "已解压, 跳过: $target"; return 0; }
  log "解压 $name -> $target"
  mkdir -p "$target"
  unzip -q -o "$zip" -d "$target"
}

# COCO: zip 内是 train2017/*.jpg -> 目标 coco/train2017/
check_and_extract "$IMG/coco_train2017.zip" "COCO" "$DEST/coco_tmp" \
  && { mkdir -p "$IMG/coco"; [[ -d "$IMG/coco/train2017" ]] || mv "$DEST/coco_tmp/train2017" "$IMG/coco/" 2>/dev/null; }

# GQA: zip 内是 images/*.jpg
check_and_extract "$IMG/gqa_images.zip" "GQA" "$IMG/gqa" \
  && { [[ -d "$IMG/gqa/images" ]] || mv "$IMG/gqa/images" "$IMG/gqa/" 2>/dev/null || true; }

# TextVQA: zip 内是 train_images/*.jpg
check_and_extract "$IMG/textvqa_images.zip" "TextVQA" "$IMG/textvqa" \
  && { [[ -d "$IMG/textvqa/train_images" ]] || mv "$IMG/textvqa/train_images" "$IMG/textvqa/" 2>/dev/null || true; }

# VisualGenome: images.zip -> VG_100K, images2.zip -> VG_100K_2
for z in images.zip images2.zip; do
  [[ -f "$IMG/vg/$z" ]] && { verify_zip "$IMG/vg/$z" && unzip -q -o "$IMG/vg/$z" -d "$IMG/vg"; } || true
done

log "解压完成。抽查各前缀图像数:"
for d in coco/train2017 gqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2; do
  n=$(ls "$IMG/$d" 2>/dev/null | wc -l)
  printf "  %-26s %8s 张\n" "$d" "$n"
done

warn "若某前缀为 0, 抽取脚本会静默跳过对应样本 —— 请看 stats_report.md 的丢弃明细。"

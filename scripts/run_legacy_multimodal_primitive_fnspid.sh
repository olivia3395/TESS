#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
SCALE="${SCALE:-legacy_standard}"
MODE="${MODE:-both}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tess_basic/real_fnspid_legacy_standard}"

run_one() {
  local primitive_source="$1"
  local output_dir="$2"
  python -m experiments.train_tess_basic \
    --root "$ROOT" \
    --dataset fnspid \
    --model legacy_multimodal_primitive \
    --primitive-source "$primitive_source" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --scale "$SCALE" \
    --seed "$SEED" \
    --output-dir "$output_dir" \
    --overwrite
}

case "$MODE" in
  text)
    run_one text "$OUT_ROOT/legacy_multimodal_primitive_text"
    ;;
  gt)
    run_one gt "$OUT_ROOT/legacy_multimodal_primitive_gt"
    ;;
  both)
    run_one text "$OUT_ROOT/legacy_multimodal_primitive_text"
    run_one gt "$OUT_ROOT/legacy_multimodal_primitive_gt"
    ;;
  *)
    echo "MODE must be one of: text, gt, both" >&2
    exit 2
    ;;
esac

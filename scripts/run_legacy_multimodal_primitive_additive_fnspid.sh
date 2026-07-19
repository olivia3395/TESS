#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
SCALE="${SCALE:-legacy_standard}"
MODE="${MODE:-all}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tess_basic/real_fnspid_legacy_standard}"

run_one() {
  local experiment_name="$1"
  local primitive_source="$2"
  local text_delta_scale="$3"
  local output_dir="$OUT_ROOT/$experiment_name"

  python -m experiments.train_tess_basic \
    --root "$ROOT" \
    --dataset fnspid \
    --model legacy_multimodal_primitive_additive \
    --primitive-source "$primitive_source" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --scale "$SCALE" \
    --seed "$SEED" \
    --text-delta-scale "$text_delta_scale" \
    --output-dir "$output_dir" \
    --overwrite
}

case "$MODE" in
  additive_text_scale0|text_scale0)
    run_one additive_text_scale0 text 0.0
    ;;
  additive_text_scale1|text_scale1)
    run_one additive_text_scale1 text 1.0
    ;;
  additive_gt_scale1|gt_scale1)
    run_one additive_gt_scale1 gt 1.0
    ;;
  all)
    run_one additive_text_scale0 text 0.0
    run_one additive_text_scale1 text 1.0
    run_one additive_gt_scale1 gt 1.0
    ;;
  *)
    echo "MODE must be one of: additive_text_scale0, additive_text_scale1, additive_gt_scale1, all" >&2
    exit 2
    ;;
esac

echo
echo "After the runs complete, summarize with:"
echo "python -m experiments.summarize_additive_results --out-root \"$OUT_ROOT\""

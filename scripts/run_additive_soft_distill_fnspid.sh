#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
SCALE="${SCALE:-legacy_standard}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tess_basic/real_fnspid_legacy_standard}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$OUT_ROOT/additive_gt_scale1/best.pt}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/additive_text_soft_distill_delta_lam0p1}"

python -m experiments.train_additive_distill \
  --root "$ROOT" \
  --dataset fnspid \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
  --student-model legacy_multimodal_primitive_additive_soft \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --lambda-delta-distill 0.1 \
  --lambda-pred-distill 0.0 \
  --student-text-delta-scale 1.0 \
  --teacher-text-delta-scale 1.0 \
  --experiment-name additive_text_soft_distill_delta_lam0p1 \
  --output-dir "$OUT_DIR" \
  --overwrite

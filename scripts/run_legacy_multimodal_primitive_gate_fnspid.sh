#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
GATE_LOSS_WEIGHT="${GATE_LOSS_WEIGHT:-0.1}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/tess_basic/real_fnspid_legacy_standard/legacy_multimodal_primitive_gate_text}"

python -m experiments.train_tess_basic \
  --root "$ROOT" \
  --dataset fnspid \
  --model legacy_multimodal_primitive_gate \
  --primitive-source text \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale legacy_standard \
  --seed "$SEED" \
  --gate-loss-weight "$GATE_LOSS_WEIGHT" \
  --output-dir "$OUT_DIR" \
  --overwrite

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
DATASET="${DATASET:-fnspid}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/tess_basic/real_fnspid_legacy_standard/legacy_timesnet}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
D_MODEL="${D_MODEL:-512}"
E_LAYERS="${E_LAYERS:-2}"
TOP_K="${TOP_K:-5}"
NUM_KERNELS="${NUM_KERNELS:-6}"
SCALE="${SCALE:-legacy_standard}"
OVERWRITE="${OVERWRITE:-1}"

OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

python -m experiments.train_tess_basic \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --model legacy_timesnet \
  --primitive-source none \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --d-model "$D_MODEL" \
  --e-layers "$E_LAYERS" \
  --top-k "$TOP_K" \
  --num-kernels "$NUM_KERNELS" \
  --output-dir "$OUT_DIR" \
  "${OVERWRITE_ARGS[@]}"

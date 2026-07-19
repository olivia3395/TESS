#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./}"
DATASET="${DATASET:-fnspid}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
SEED="${SEED:-2026}"
SCALE="${SCALE:-legacy_standard}"
OUT_ROOT="${OUT_ROOT:-outputs/tess_basic/real_${DATASET}}"
OVERWRITE="${OVERWRITE:-1}"
export OUT_ROOT

OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

mkdir -p "$OUT_ROOT"

python -m experiments.inspect_tess_dataset \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --output-json "$OUT_ROOT/inspect_report.json"

python -m experiments.train_tess_basic \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model numeric_mlp \
  --primitive-source none \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT/numeric_mlp" \
  "${OVERWRITE_ARGS[@]}"

python -m experiments.train_tess_basic \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model tiny_temporal \
  --primitive-source none \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT/tiny_temporal" \
  "${OVERWRITE_ARGS[@]}"

python -m experiments.train_tess_basic \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model tess_nogate \
  --primitive-source text \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT/tess_mlp_text_nogate" \
  "${OVERWRITE_ARGS[@]}"

python -m experiments.train_tess_basic \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model tiny_temporal_tess \
  --primitive-source text \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT/tiny_tess_text_nogate" \
  "${OVERWRITE_ARGS[@]}"

python -m experiments.train_tess_basic \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --model tiny_temporal_tess \
  --primitive-source gt \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --output-dir "$OUT_ROOT/tiny_tess_gt_oracle" \
  "${OVERWRITE_ARGS[@]}"

python - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
rows = [
    "numeric_mlp",
    "tiny_temporal",
    "tess_mlp_text_nogate",
    "tiny_tess_text_nogate",
    "tiny_tess_gt_oracle",
]
print("\ncomparison")
print("name,best_epoch,best_vali_mse,best_reloaded_test_mse,final_test_mse,train_loss_decreased,best_reload_ok")
for name in rows:
    summary_path = out_root / name / "summary.json"
    summary = json.loads(summary_path.read_text())
    print(
        f"{name},{summary['best_epoch']},"
        f"{summary['best_vali_mse']:.8f},"
        f"{summary['best_reloaded_test_mse']:.8f},"
        f"{summary['final_test_mse']:.8f},"
        f"{summary['train_loss_decreased']},"
        f"{summary['best_reload_check']['passed']}"
    )
PY

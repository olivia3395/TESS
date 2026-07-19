#!/usr/bin/env bash
set -euo pipefail

MOCK_ROOT="${MOCK_ROOT:-/tmp/tess_mock_root}"

python -m experiments.make_mock_tess_data \
  --root "$MOCK_ROOT" \
  --dataset fnspid \
  --n-train 64 \
  --n-vali 16 \
  --n-test 16 \
  --seq-len 6 \
  --pred-len 6 \
  --num-samples 3 \
  --overwrite

python -m experiments.train_tess_basic \
  --root "$MOCK_ROOT" \
  --dataset fnspid \
  --model numeric_mlp \
  --primitive-source none \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cpu \
  --scale raw \
  --output-dir outputs/tess_basic/mock_numeric \
  --overwrite

python -m experiments.train_tess_basic \
  --root "$MOCK_ROOT" \
  --dataset fnspid \
  --model tess_nogate \
  --primitive-source text \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cpu \
  --scale raw \
  --output-dir outputs/tess_basic/mock_text_nogate \
  --overwrite

python -m experiments.train_tess_basic \
  --root "$MOCK_ROOT" \
  --dataset fnspid \
  --model tess_nogate \
  --primitive-source gt \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cpu \
  --scale raw \
  --output-dir outputs/tess_basic/mock_oracle \
  --overwrite

echo "Metrics and checkpoints:"
echo "  outputs/tess_basic/mock_numeric/{metrics.json,summary.json,best.pt,last.pt}"
echo "  outputs/tess_basic/mock_text_nogate/{metrics.json,summary.json,best.pt,last.pt}"
echo "  outputs/tess_basic/mock_oracle/{metrics.json,summary.json,best.pt,last.pt}"

python - <<'PY'
import json
from pathlib import Path

rows = [
    ("numeric", Path("outputs/tess_basic/mock_numeric/summary.json")),
    ("text_nogate", Path("outputs/tess_basic/mock_text_nogate/summary.json")),
    ("oracle", Path("outputs/tess_basic/mock_oracle/summary.json")),
]
print("\ncomparison")
print("name,best_epoch,best_vali_mse,final_test_mse,best_reloaded_test_mse,train_loss_decreased,best_reload_ok")
for name, path in rows:
    summary = json.loads(path.read_text())
    print(
        f"{name},{summary['best_epoch']},"
        f"{summary['best_vali_mse']:.8f},"
        f"{summary['final_test_mse']:.8f},"
        f"{summary['best_reloaded_test_mse']:.8f},"
        f"{summary['train_loss_decreased']},"
        f"{summary['best_reload_check']['passed']}"
    )
PY

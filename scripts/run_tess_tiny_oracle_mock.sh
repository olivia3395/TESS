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
  --model tiny_temporal \
  --primitive-source none \
  --epochs 80 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cpu \
  --scale raw \
  --output-dir outputs/tess_basic/mock_tiny_temporal \
  --overwrite

python -m experiments.train_tess_basic \
  --root "$MOCK_ROOT" \
  --dataset fnspid \
  --model tiny_temporal_tess \
  --primitive-source gt \
  --epochs 80 \
  --batch-size 16 \
  --lr 1e-3 \
  --device cpu \
  --scale raw \
  --output-dir outputs/tess_basic/mock_tiny_oracle \
  --overwrite

python - <<'PY'
import json
from pathlib import Path

rows = [
    ("tiny_temporal", Path("outputs/tess_basic/mock_tiny_temporal/summary.json")),
    ("tiny_oracle", Path("outputs/tess_basic/mock_tiny_oracle/summary.json")),
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

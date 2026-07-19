#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user_home/hyh/workspace/TESS-RC2"
DATASET="fnspid"

BASE_URL="http://127.0.0.1:18000/v1"
API_KEY="EMPTY"
MODEL="qwen3-8b"

NUM_SAMPLES=8
SEED=42

TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MAX_TOKENS=64
THINKING_MODE="off"

SPLITS=("train" "vali" "test")
PRIMITIVES=("distribution_shift" "volatility" "shape" "temporal_influence")

LOG_DIR="${ROOT}/logs/full_real_$(date +%m%d_%H%M)"
mkdir -p "$LOG_DIR"

cd "$ROOT"

echo "[0] Logs will be written to: $LOG_DIR"
echo "[1] Compile check"
python -m compileall primitive_inference_rc2 | tee "${LOG_DIR}/compile.log"

echo "[2] vLLM model endpoint"
curl -s "${BASE_URL}/models" | tee "${LOG_DIR}/models.json"
echo

for SPLIT in "${SPLITS[@]}"; do
  for PRIM in "${PRIMITIVES[@]}"; do
    echo "============================================================"
    echo "[sampled] split=${SPLIT}, primitive=${PRIM}, num_samples=${NUM_SAMPLES}"
    echo "============================================================"

    python -m primitive_inference_rc2.sampled_inference \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --primitive "$PRIM" \
      --backend openai-compatible \
      --base-url "$BASE_URL" \
      --api-key "$API_KEY" \
      --model "$MODEL" \
      --prompt-source auto \
      --num-samples "$NUM_SAMPLES" \
      --seed "$SEED" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --max-tokens "$MAX_TOKENS" \
      --thinking-mode "$THINKING_MODE" \
      2>&1 | tee "${LOG_DIR}/sampled_${SPLIT}_${PRIM}.log"

    echo "[gate] split=${SPLIT}, primitive=${PRIM}"

    python -m primitive_inference_rc2.build_gate_cache \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --primitive "$PRIM" \
      2>&1 | tee "${LOG_DIR}/gate_${SPLIT}_${PRIM}.log"

    echo "[done] split=${SPLIT}, primitive=${PRIM}"
    echo
  done

  echo "============================================================"
  echo "[grouped] split=${SPLIT}"
  echo "============================================================"

  python -m primitive_inference_rc2.build_grouped_gate_cache \
    --root "$ROOT" \
    --dataset "$DATASET" \
    --splits "$SPLIT" \
    --primitives "${PRIMITIVES[@]}" \
    2>&1 | tee "${LOG_DIR}/grouped_${SPLIT}.log"
done

echo "============================================================"
echo "[summary]"
echo "============================================================"

python - <<'PY' | tee "${LOG_DIR}/summary.log"
import json
from pathlib import Path

ROOT = Path("/home/user_home/hyh/workspace/TESS-RC2")
dataset = "fnspid"
splits = ["train", "vali", "test"]
primitives = ["distribution_shift", "volatility", "shape", "temporal_influence"]

for split in splits:
    print(f"\n=== split={split} ===")
    for prim in primitives:
        p = ROOT / "data_cache" / "sampled_inference" / dataset / prim / f"{split}.json"
        data = json.loads(p.read_text(encoding="utf-8"))

        parse_rates = [float(r.get("parse_rate", 0.0)) for r in data]
        self_cons = [float(r.get("self_consistency", 0.0)) for r in data]
        margins = [float(r.get("margin", 0.0)) for r in data]

        pred_counts = {}
        for r in data:
            pred = r.get("pred_label")
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        n = len(data)
        print(
            f"[{prim}] records={n}, "
            f"parse_mean={sum(parse_rates)/n:.4f}, "
            f"self_consistency_mean={sum(self_cons)/n:.4f}, "
            f"margin_mean={sum(margins)/n:.4f}, "
            f"pred_counts={pred_counts}"
        )

    grouped_path = ROOT / "data_cache" / "gate_grouped" / dataset / f"{split}.json"
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    print(f"[grouped] records={len(grouped)} path={grouped_path}")
PY

echo "Full real inference finished."
echo "Logs: $LOG_DIR"
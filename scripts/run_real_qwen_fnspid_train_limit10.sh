# #!/usr/bin/env bash
# set -euo pipefail

# # Real vLLM small-sample smoke run for TESS-RC2 primitive inference.

# # This version uses bash arrays to avoid fragile backslash line continuations.

# ROOT="/home/user_home/hyh/workspace/TESS-RC2"
# DATASET="fnspid"
# SPLIT="train"

# BASE_URL="http://127.0.0.1:18000/v1"
# API_KEY="EMPTY"

# # This must match the vLLM --served-model-name.

# # If you want MODEL=qwen3-8b here, start vLLM with:

# # --served-model-name qwen3-8b

# MODEL="qwen3-8b"

# NUM_SAMPLES=4
# LIMIT=10
# SEED=42

# TEMPERATURE=0.7
# TOP_P=0.8
# TOP_K=20
# MAX_TOKENS=64
# THINKING_MODE="off"

# PRIMITIVES=(
# "distribution_shift"
# "volatility"
# "shape"
# "temporal_influence"
# )

# cd "$ROOT"

# echo "[1/5] Checking Python package compilation..."
# python -m compileall primitive_inference_rc2

# echo "[2/5] Checking vLLM server models endpoint..."
# curl -s "${BASE_URL}/models" | head -c 1000
# echo
# echo

# echo "[3/5] Running real sampled inference + flat gate cache..."
# for PRIM in "${PRIMITIVES[@]}"; do
# echo "========== Primitive: ${PRIM} | Split: ${SPLIT} =========="

# sampled_cmd=(
# python -m primitive_inference_rc2.sampled_inference
# --root "$ROOT"
# --dataset "$DATASET"
# --split "$SPLIT"
# --primitive "$PRIM"
# --backend openai-compatible
# --base-url "$BASE_URL"
# --api-key "$API_KEY"
# --model "$MODEL"
# --prompt-source auto
# --num-samples "$NUM_SAMPLES"
# --limit "$LIMIT"
# --seed "$SEED"
# --temperature "$TEMPERATURE"
# --top-p "$TOP_P"
# --top-k "$TOP_K"
# --max-tokens "$MAX_TOKENS"
# --thinking-mode "$THINKING_MODE"
# )

# echo "[sampled] ${sampled_cmd[*]}"
# "${sampled_cmd[@]}"

# gate_cmd=(
# python -m primitive_inference_rc2.build_gate_cache
# --root "$ROOT"
# --dataset "$DATASET"
# --split "$SPLIT"
# --primitive "$PRIM"
# )

# echo "[gate] ${gate_cmd[*]}"
# "${gate_cmd[@]}"

# echo
# done

# echo "[4/5] Building grouped gate cache..."
# grouped_cmd=(
# python -m primitive_inference_rc2.build_grouped_gate_cache
# --root "$ROOT"
# --dataset "$DATASET"
# --splits "$SPLIT"
# --primitives "${PRIMITIVES[@]}"
# )

# echo "[grouped] ${grouped_cmd[*]}"
# "${grouped_cmd[@]}"

# echo "[5/5] Quick output summary..."
python - <<'PY'
import json
from pathlib import Path

ROOT = Path("/home/user_home/hyh/workspace/TESS-RC2")
dataset = "fnspid"
split = "train"
primitives = ["distribution_shift", "volatility", "shape", "temporal_influence"]

for prim in primitives:
    p = ROOT / "data_cache" / "sampled_inference" / dataset / prim / f"{split}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    parse_rates = [float(r.get("parse_rate", 0.0)) for r in data]

    preds = {}
    for r in data:
        pred = r.get("pred_label")
        preds[pred] = preds.get(pred, 0) + 1

    mean_parse = sum(parse_rates) / len(parse_rates) if parse_rates else 0.0
    print(
        f"[sampled] {prim}: "
        f"records={len(data)}, "
        f"mean_parse_rate={mean_parse:.3f}, "
        f"pred_counts={preds}"
    )

grouped_path = ROOT / "data_cache" / "gate_grouped" / dataset / f"{split}.json"
grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
print(f"[grouped] records={len(grouped)} path={grouped_path}")
PY

echo "Done."

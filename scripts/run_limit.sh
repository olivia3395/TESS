ROOT=/home/user_home/hyh/workspace/TESS-RC2
BASE_URL=http://127.0.0.1:18000/v1
MODEL=qwen3-8b
DATASET=fnspid

PRIMITIVES=(distribution_shift volatility shape temporal_influence)
SPLITS=(train vali test)

for split in "${SPLITS[@]}"; do
  for prim in "${PRIMITIVES[@]}"; do
    echo "===== split=${split}, primitive=${prim} ====="

    python -m primitive_inference_rc2.sampled_inference \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$split" \
      --primitive "$prim" \
      --backend openai-compatible \
      --base-url "$BASE_URL" \
      --api-key EMPTY \
      --model "$MODEL" \
      --prompt-source auto \
      --num-samples 8 \
      --limit 1000 \
      --seed 42 \
      --temperature 0.7 \
      --top-p 0.8 \
      --top-k 20 \
      --max-tokens 64 \
      --thinking-mode off

    python -m primitive_inference_rc2.build_gate_cache \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$split" \
      --primitive "$prim"
  done
done

python -m primitive_inference_rc2.build_grouped_gate_cache \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --splits train vali test \
  --primitives "${PRIMITIVES[@]}"
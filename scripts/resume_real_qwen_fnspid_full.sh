#!/usr/bin/env bash
set -euo pipefail

# Resume full real Qwen/vLLM primitive inference for fnspid.
# This script skips completed <split, primitive> sampled cache files by record count.
# It avoids Python heredoc blocks to prevent indentation corruption.

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

LOG_DIR="${ROOT}/logs/resume_real_$(date +%m%d_%H%M)"
mkdir -p "$LOG_DIR"

cd "$ROOT"

expected_count() {
  local split="$1"
  case "$split" in
    train) echo 6842 ;;
    vali) echo 1424 ;;
    test) echo 1567 ;;
    *)
      echo "Unknown split: $split" >&2
      exit 2
      ;;
  esac
}

sampled_path() {
  local split="$1"
  local prim="$2"
  echo "${ROOT}/data_cache/sampled_inference/${DATASET}/${prim}/${split}.json"
}

json_count() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo -1
    return 0
  fi

  python -c 'exec("import json, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\ntry:\n    data = json.loads(p.read_text(encoding=\"utf-8\"))\n    print(len(data) if isinstance(data, list) else -2)\nexcept Exception:\n    print(-3)\n")' "$path"
}

json_brief_summary() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo "missing file: $path"
    return 0
  fi

  python -c 'exec("import json, sys\nfrom pathlib import Path\np = Path(sys.argv[1])\ndata = json.loads(p.read_text(encoding=\"utf-8\"))\nn = len(data)\nparse_rates = [float(r.get(\"parse_rate\", 0.0)) for r in data]\nself_cons = [float(r.get(\"self_consistency\", 0.0)) for r in data]\nmargins = [float(r.get(\"margin\", 0.0)) for r in data]\npred_counts = {}\nfor r in data:\n    pred = r.get(\"pred_label\")\n    pred_counts[pred] = pred_counts.get(pred, 0) + 1\nparse_mean = sum(parse_rates) / n if n else 0.0\nself_mean = sum(self_cons) / n if n else 0.0\nmargin_mean = sum(margins) / n if n else 0.0\nprint(f\"records={n}, parse_mean={parse_mean:.4f}, self_consistency_mean={self_mean:.4f}, margin_mean={margin_mean:.4f}, pred_counts={pred_counts}\")\n")' "$path"
}

echo "[0] Logs will be written to: $LOG_DIR"

echo "[1] Compile check"
python -m compileall primitive_inference_rc2 | tee "${LOG_DIR}/compile.log"

echo "[2] Check vLLM endpoint"
curl -s "${BASE_URL}/models" | tee "${LOG_DIR}/models.json"
echo

echo "[3] Current cache status"
for split in "${SPLITS[@]}"; do
  expected="$(expected_count "$split")"
  echo "=== ${split} expected=${expected} ===" | tee -a "${LOG_DIR}/status_before.log"

  for prim in "${PRIMITIVES[@]}"; do
    path="$(sampled_path "$split" "$prim")"
    count="$(json_count "$path")"

    if [[ "$count" == "$expected" ]]; then
      status="OK"
    elif [[ "$count" == "-1" ]]; then
      status="MISSING"
    elif [[ "$count" == "-2" ]]; then
      status="NOT_LIST_JSON"
    elif [[ "$count" == "-3" ]]; then
      status="BROKEN_JSON"
    else
      status="INCOMPLETE"
    fi

    echo "${prim}: ${count} records [${status}]" | tee -a "${LOG_DIR}/status_before.log"
  done

  echo | tee -a "${LOG_DIR}/status_before.log"
done

echo "[4] Resume sampled inference + flat gate cache"

for split in "${SPLITS[@]}"; do
  for prim in "${PRIMITIVES[@]}"; do
    expected="$(expected_count "$split")"
    path="$(sampled_path "$split" "$prim")"
    count="$(json_count "$path")"

    echo "============================================================"
    echo "split=${split}, primitive=${prim}, current_count=${count}, expected=${expected}"
    echo "============================================================"

    if [[ "$count" == "$expected" ]]; then
      echo "[skip sampled] ${split}/${prim} already complete."
      echo "[rebuild gate] ${split}/${prim}"

      python -m primitive_inference_rc2.build_gate_cache \
        --root "$ROOT" \
        --dataset "$DATASET" \
        --split "$split" \
        --primitive "$prim" \
        2>&1 | tee "${LOG_DIR}/gate_${split}_${prim}.log"

      echo
      continue
    fi

    if [[ -f "$path" ]]; then
      backup="${path}.bak_incomplete_$(date +%m%d_%H%M%S)"
      echo "[backup incomplete] $path -> $backup"
      mv "$path" "$backup"
    fi

    echo "[run sampled] ${split}/${prim}"

    python -m primitive_inference_rc2.sampled_inference \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$split" \
      --primitive "$prim" \
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
      2>&1 | tee "${LOG_DIR}/sampled_${split}_${prim}.log"

    new_count="$(json_count "$path")"

    if [[ "$new_count" != "$expected" ]]; then
      echo "[ERROR] sampled output count mismatch for ${split}/${prim}: got ${new_count}, expected ${expected}" >&2
      exit 1
    fi

    echo "[build gate] ${split}/${prim}"

    python -m primitive_inference_rc2.build_gate_cache \
      --root "$ROOT" \
      --dataset "$DATASET" \
      --split "$split" \
      --primitive "$prim" \
      2>&1 | tee "${LOG_DIR}/gate_${split}_${prim}.log"

    echo
  done

  echo "============================================================"
  echo "[build grouped] split=${split}"
  echo "============================================================"

  python -m primitive_inference_rc2.build_grouped_gate_cache \
    --root "$ROOT" \
    --dataset "$DATASET" \
    --splits "$split" \
    --primitives "${PRIMITIVES[@]}" \
    2>&1 | tee "${LOG_DIR}/grouped_${split}.log"
done

echo "[5] Final summary" | tee "${LOG_DIR}/summary.log"

ALL_OK=1

for split in "${SPLITS[@]}"; do
  expected="$(expected_count "$split")"
  echo "" | tee -a "${LOG_DIR}/summary.log"
  echo "=== split=${split}, expected=${expected} ===" | tee -a "${LOG_DIR}/summary.log"

  for prim in "${PRIMITIVES[@]}"; do
    path="$(sampled_path "$split" "$prim")"
    count="$(json_count "$path")"

    if [[ "$count" == "$expected" ]]; then
      status="OK"
    else
      status="BAD"
      ALL_OK=0
    fi

    echo -n "[${prim}] count=${count} [${status}], " | tee -a "${LOG_DIR}/summary.log"
    json_brief_summary "$path" | tee -a "${LOG_DIR}/summary.log"
  done

  grouped_path="${ROOT}/data_cache/gate_grouped/${DATASET}/${split}.json"
  grouped_count="$(json_count "$grouped_path")"

  if [[ "$grouped_count" == "$expected" ]]; then
    grouped_status="OK"
  else
    grouped_status="BAD"
    ALL_OK=0
  fi

  echo "[grouped] count=${grouped_count} [${grouped_status}] path=${grouped_path}" | tee -a "${LOG_DIR}/summary.log"
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "ALL_OK=${ALL_OK}" | tee -a "${LOG_DIR}/summary.log"

if [[ "$ALL_OK" != "1" ]]; then
  echo "[ERROR] Some outputs are incomplete or invalid. See ${LOG_DIR}/summary.log" >&2
  exit 1
fi

echo "Resume full real inference finished."
echo "Logs: $LOG_DIR"
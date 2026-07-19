#!/usr/bin/env bash
set -euo pipefail

# Run real primitive inference for non-FNSPID datasets with sampling=4.
#
# Usage example:
#   chmod +x scripts/run_other_datasets_sampling4.sh
#   DATA_ROOT=/home/user_home/hyh/workspace/TESS-RC2 \
#   BASE_URL=http://localhost:8000/v1 \
#   MODEL=<your-vllm-model-name> \
#   bash scripts/run_other_datasets_sampling4.sh
#
# Optional:
#   DATASETS="bitcoin electricity environment" NUM_SAMPLES=4 OVERWRITE=0 bash scripts/run_other_datasets_sampling4.sh

DATA_ROOT="${DATA_ROOT:-/home/user_home/hyh/workspace/TESS-RC2}"
DATASETS="${DATASETS:-fnspid bitcoin electricity environment}"

PRIMITIVES="${PRIMITIVES:-distribution_shift volatility shape temporal_influence}"
SPLITS="${SPLITS:-train vali test}"

NUM_SAMPLES="${NUM_SAMPLES:-4}"
PROMPT_SOURCE="${PROMPT_SOURCE:-auto}"

BACKEND="${BACKEND:-openai-compatible}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-64}"
THINKING_MODE="${THINKING_MODE:-off}"

SEED="${SEED:-42}"

# OVERWRITE=0 means: skip sampled inference file if it already exists.
# Gate/grouped/audit are still rebuilt after sampled caches are available.
OVERWRITE="${OVERWRITE:-0}"

LOG_DIR="${LOG_DIR:-${DATA_ROOT}/outputs/primitive_sampling4_logs}"
mkdir -p "$LOG_DIR"

cd "$DATA_ROOT"

echo "=== Primitive inference sampling=4 run ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "DATASETS=${DATASETS}"
echo "PRIMITIVES=${PRIMITIVES}"
echo "SPLITS=${SPLITS}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "BACKEND=${BACKEND}"
echo "BASE_URL=${BASE_URL}"
echo "MODEL=${MODEL:-<none>}"
echo "PROMPT_SOURCE=${PROMPT_SOURCE}"
echo "OVERWRITE=${OVERWRITE}"
echo "LOG_DIR=${LOG_DIR}"
echo

if [[ "$BACKEND" == "openai-compatible" && -z "$MODEL" ]]; then
  echo "[ERROR] MODEL is empty. Please set MODEL=<your-vllm-model-name> for openai-compatible backend." >&2
  exit 1
fi

run_logged() {
  local log_name="$1"
  shift
  echo
  echo ">>> $*"
  "$@" 2>&1 | tee "${LOG_DIR}/${log_name}.log"
}

sampled_cache_path() {
  local dataset="$1"
  local primitive="$2"
  local split="$3"
  echo "${DATA_ROOT}/data_cache/sampled_inference/${dataset}/${primitive}/${split}.json"
}

for dataset in $DATASETS; do
  echo
  echo "============================================================"
  echo "DATASET=${dataset}"
  echo "============================================================"

  for primitive in $PRIMITIVES; do
    echo
    echo "---------------- GT cache: dataset=${dataset}, primitive=${primitive} ----------------"
    run_logged "gt_${dataset}_${primitive}" \
      python -m primitive_inference_rc2.build_gt_cache \
        --root "$DATA_ROOT" \
        --dataset "$dataset" \
        --primitive "$primitive"
  done

  for primitive in $PRIMITIVES; do
    for split in $SPLITS; do
      out_path="$(sampled_cache_path "$dataset" "$primitive" "$split")"

      if [[ "$OVERWRITE" != "1" && -f "$out_path" ]]; then
        echo
        echo "[SKIP sampled] existing file: $out_path"
        continue
      fi

      echo
      echo "---------------- sampled inference: dataset=${dataset}, primitive=${primitive}, split=${split} ----------------"

      cmd=(
        python -m primitive_inference_rc2.sampled_inference
        --root "$DATA_ROOT"
        --dataset "$dataset"
        --split "$split"
        --primitive "$primitive"
        --backend "$BACKEND"
        --prompt-source "$PROMPT_SOURCE"
        --num-samples "$NUM_SAMPLES"
        --seed "$SEED"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --top-k "$TOP_K"
        --max-tokens "$MAX_TOKENS"
        --thinking-mode "$THINKING_MODE"
      )

      if [[ "$BACKEND" == "openai-compatible" ]]; then
        cmd+=(--base-url "$BASE_URL" --api-key "$API_KEY" --model "$MODEL")
      fi

      run_logged "sampled_${dataset}_${primitive}_${split}" "${cmd[@]}"
    done
  done

  for primitive in $PRIMITIVES; do
    for split in $SPLITS; do
      echo
      echo "---------------- gate cache: dataset=${dataset}, primitive=${primitive}, split=${split} ----------------"
      run_logged "gate_${dataset}_${primitive}_${split}" \
        python -m primitive_inference_rc2.build_gate_cache \
          --root "$DATA_ROOT" \
          --dataset "$dataset" \
          --split "$split" \
          --primitive "$primitive"
    done
  done

  echo
  echo "---------------- grouped gate cache + audit: dataset=${dataset} ----------------"
  run_logged "grouped_${dataset}" \
    python -m primitive_inference_rc2.build_grouped_gate_cache \
      --root "$DATA_ROOT" \
      --dataset "$dataset" \
      --splits $SPLITS \
      --primitives $PRIMITIVES

  echo
  echo "---------------- compact status: dataset=${dataset} ----------------"
  DATA_ROOT="$DATA_ROOT" DATASET="$dataset" PRIMITIVES="$PRIMITIVES" SPLITS="$SPLITS" python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
dataset = os.environ["DATASET"]
primitives = os.environ["PRIMITIVES"].split()
splits = os.environ["SPLITS"].split()

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

for split in splits:
    raw_path = root / "dataset" / dataset / f"{split}.json"
    raw = load_json(raw_path)
    expected = len(raw) if isinstance(raw, list) else None
    print(f"\n=== dataset={dataset}, split={split}, expected={expected} ===")

    for primitive in primitives:
        path = root / "data_cache" / "sampled_inference" / dataset / primitive / f"{split}.json"
        records = load_json(path)
        if not isinstance(records, list):
            print(f"[{primitive}] MISSING path={path}")
            continue

        count = len(records)
        ok = "OK" if expected is not None and count == expected else "INCOMPLETE"
        parse_vals = [r.get("parse_rate") for r in records if isinstance(r.get("parse_rate"), (int, float))]
        sc_vals = [r.get("self_consistency") for r in records if isinstance(r.get("self_consistency"), (int, float))]
        margin_vals = [r.get("margin") for r in records if isinstance(r.get("margin"), (int, float))]
        pred_counts = Counter(r.get("pred_label") for r in records)

        parse_mean = sum(parse_vals) / len(parse_vals) if parse_vals else None
        sc_mean = sum(sc_vals) / len(sc_vals) if sc_vals else None
        margin_mean = sum(margin_vals) / len(margin_vals) if margin_vals else None

        print(
            f"[{primitive}] count={count} [{ok}], "
            f"parse_mean={parse_mean:.4f}" if parse_mean is not None else f"[{primitive}] count={count} [{ok}], parse_mean=None",
            end=""
        )
        print(
            f", self_consistency_mean={sc_mean:.4f}" if sc_mean is not None else ", self_consistency_mean=None",
            end=""
        )
        print(
            f", margin_mean={margin_mean:.4f}" if margin_mean is not None else ", margin_mean=None",
            end=""
        )
        print(f", pred_counts={dict(pred_counts)}")

    grouped_path = root / "data_cache" / "gate_grouped" / dataset / f"{split}.json"
    grouped = load_json(grouped_path)
    grouped_count = len(grouped) if isinstance(grouped, list) else None
    grouped_ok = "OK" if expected is not None and grouped_count == expected else "INCOMPLETE"
    print(f"[grouped] count={grouped_count} [{grouped_ok}] path={grouped_path}")
PY

done

echo
echo "All requested datasets finished."
echo "Logs: ${LOG_DIR}"
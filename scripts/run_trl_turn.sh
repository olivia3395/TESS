#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: scripts/run_trl_turn.sh [options]

Defaults:
  --base-alias FNSPID/ver_camf
  --model      MultiModal_Baseline
  --checkpoint saved/MultiModal_Baseline/FNSPID/ver_camf/best_epoch_Sep-25-2025-22-33-14/model.pt
  --turn       1
  --max-concurrency 500
  --scores-root output/FNSPID

Options:
  --base-alias <alias>
  --turn <id>
  --model <name>
  --checkpoint <path>
  --variants <v1 v2 ...>
  --prompt-config <path>
  --pipeline-config <path>
  --dataset-root <path>
  --output-root <path>
  --scores-root <path>
  --max-concurrency <n>
  --dry-run
  --no-embedding
  -h, --help                 Show this help and exit
EOF
}

ARGS=("--base-alias" "FNSPID/ver_camf" "--model" "MultiModal_Baseline" "--checkpoint" "saved/MultiModal_Baseline/FNSPID/ver_camf/best_epoch_Sep-25-2025-22-33-14/model.pt" "--turn" "1" "--max-concurrency" "500" "--scores-root" "output/FNSPID")
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --base-alias|--turn|--model|--checkpoint|--prompt-config|--pipeline-config|--dataset-root|--output-root|--scores-root|--max-concurrency)
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --variants)
      shift
      VARIANTS=()
      while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
        VARIANTS+=("$1")
        shift
      done
      if [[ ${#VARIANTS[@]} -gt 0 ]]; then
        ARGS+=("--variants" "${VARIANTS[@]}")
      fi
      ;;
    --dry-run|--no-embedding)
      ARGS+=("$1")
      shift
      ;;
    *)
      echo "[run_trl_turn] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

${PYTHON_BIN} -m data_generator.text_reinforcement.pipeline "${ARGS[@]}"

# auto-run when no arguments provided beyond defaults
if [[ $# -eq 0 ]]; then
  export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src:$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/model_trainer:${PYTHONPATH}"
  ${PYTHON_BIN} -m data_generator.text_reinforcement.pipeline "${ARGS[@]}"
  exit $?
fi

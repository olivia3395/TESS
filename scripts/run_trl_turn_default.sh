#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/src/model_trainer:${PYTHONPATH:-}"

BASE_ALIAS="FNSPID/ver_camf"
MODEL_NAME="MultiModal_Baseline"
CHECKPOINT="${ROOT_DIR}/saved/MultiModal_Baseline/FNSPID/ver_camf/best_epoch_Sep-25-2025-22-33-14/model.pt"
OUTPUT_ROOT="${ROOT_DIR}/output"
SCORES_ROOT="${OUTPUT_ROOT}/FNSPID"
TURN_ID="1"

# 清理旧产物
declare -a CLEAN_PATHS=(
  "${OUTPUT_ROOT}/FNSPID/ver_camf/train_scores.jsonl"
  "${OUTPUT_ROOT}/FNSPID/ver_camf/MultiModal_Baseline/train_scores.jsonl"
  "${OUTPUT_ROOT}/FNSPID/ver_camf/MultiModal_Baseline/vali_scores.jsonl"
  "${OUTPUT_ROOT}/FNSPID/ver_camf/MultiModal_Baseline/test_scores.jsonl"
  "${ROOT_DIR}/dataset/FNSPID/trl_turn${TURN_ID}/dpo_pairs.jsonl"
  "${ROOT_DIR}/metrics/trl_turn${TURN_ID}.json"
)
for path in "${CLEAN_PATHS[@]}"; do
  rm -f "$path"
done
rm -rf "${ROOT_DIR}/dataset/FNSPID/ver_turn${TURN_ID}_aug1"
rm -rf "${ROOT_DIR}/dataset/FNSPID/ver_turn${TURN_ID}_aug2"
rm -rf "${OUTPUT_ROOT}/FNSPID/ver_turn${TURN_ID}_aug1"
rm -rf "${OUTPUT_ROOT}/FNSPID/ver_turn${TURN_ID}_aug2"

# 重新生成基线 ver_camf 分数
${PYTHON_BIN} - <<'PY'
from pathlib import Path
from data_generator.text_reinforcement.turn_dataset_builder import TurnDatasetBuilder
builder = TurnDatasetBuilder(
    dataset_alias='FNSPID/ver_camf',
    dataset_name='FNSPID',
    dataset_version='ver_camf',
    model_name='MultiModal_Baseline',
    checkpoint_path=Path('saved/MultiModal_Baseline/FNSPID/ver_camf/best_epoch_Sep-25-2025-22-33-14/model.pt'),
    batch_size=32,
    auto_generate_embedding=False,
)
builder.run(Path('output'))
PY

# 运行完整 TRL 流水线
${PYTHON_BIN} -m data_generator.text_reinforcement.pipeline \
  --base-alias "${BASE_ALIAS}" \
  --turn "${TURN_ID}" \
  --model "${MODEL_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --max-concurrency 500 \
  --scores-root "${SCORES_ROOT}" \
  --output-root "${OUTPUT_ROOT}" || {
    status=$?
    echo "[WARN] pipeline exited with status ${status}" >&2
    exit ${status}
  }

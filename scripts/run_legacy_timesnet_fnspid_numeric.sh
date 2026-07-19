#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"
LEGACY_SRC="$ROOT/legacy/src"
LEGACY_MODEL_TRAINER="$ROOT/legacy/src/model_trainer"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/legacy/dataset/FNSPID}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/legacy_timesnet_fnspid}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$OUT_ROOT/checkpoints}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-$OUT_ROOT/output}"
LOG_PATH="${LOG_PATH:-$OUT_ROOT/run.log}"

DEVICE="${DEVICE:-cpu}"
USE_GPU="false"
GPU_ID="${GPU_ID:-0}"
if [[ "$DEVICE" == cuda* ]]; then
  USE_GPU="true"
fi

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0005}"
SEED="${SEED:-42}"
PATIENCE="${PATIENCE:-3}"
D_MODEL="${D_MODEL:-512}"
E_LAYERS="${E_LAYERS:-2}"
TOP_K="${TOP_K:-5}"
NUM_KERNELS="${NUM_KERNELS:-6}"

mkdir -p "$OUT_ROOT" "$CHECKPOINT_ROOT" "$RUN_OUTPUT_ROOT"

export PYTHONPATH="$LEGACY_SRC:$LEGACY_MODEL_TRAINER:${PYTHONPATH:-}"

{
  echo "legacy TimesNet numeric FNSPID"
  echo "root=$ROOT"
  echo "dataset_root=$DATASET_ROOT"
  echo "output_root=$RUN_OUTPUT_ROOT"
  echo "checkpoint_root=$CHECKPOINT_ROOT"
  echo "device=$DEVICE use_gpu=$USE_GPU gpu_id=$GPU_ID"
  echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR seed=$SEED patience=$PATIENCE"
  echo "embedding_size=$D_MODEL e_layers=$E_LAYERS top_k=$TOP_K num_kernels=$NUM_KERNELS"
  python -m model_trainer.main \
    -d FNSPID \
    -m TimesNet \
    --primitive-source none \
    -g "$GPU_ID" \
    --config "use_gpu=$USE_GPU" \
    --config "seed=[$SEED]" \
    --config "dataset_root=$DATASET_ROOT" \
    --config "dataset_version=ver_primitive" \
    --config "train_file=ver_primitive/train.json" \
    --config "vali_file=ver_primitive/vali.json" \
    --config "test_file=ver_primitive/test.json" \
    --config "seq_len=5" \
    --config "pred_len=5" \
    --config "label_len=0" \
    --config "batch_size=$BATCH_SIZE" \
    --config "epochs=$EPOCHS" \
    --config "patience=$PATIENCE" \
    --config "learning_rate=[$LR]" \
    --config "embedding_size=$D_MODEL" \
    --config "e_layers=$E_LAYERS" \
    --config "top_k=$TOP_K" \
    --config "num_kernels=$NUM_KERNELS" \
    --config "output_dir=$RUN_OUTPUT_ROOT" \
    --config "checkpoint_dir=$CHECKPOINT_ROOT" \
    --config "export_sample_metrics=true"
} 2>&1 | tee "$LOG_PATH"

python -m experiments.compare_timesnet_legacy_vs_rc2 \
  --legacy-root "$OUT_ROOT" \
  --print-only

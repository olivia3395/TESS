#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

env_name=$(basename "${CONDA_DEFAULT_ENV:-}")
if [[ "${env_name}" != "hui" ]]; then
  echo "[ERROR] 请先执行 'conda activate hui' 后再运行本脚本" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/src/model_trainer:${PYTHONPATH:-}"

NPROC="${DPO_NPROC:-1}"
MASTER_PORT="${DPO_MASTER_PORT:-29500}"

# 默认根据 NPROC 设置可见 GPU，确保多卡训练可用
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${NPROC}" -gt 1 ]]; then
    default_gpus=$(seq -s ',' 0 $((NPROC - 1)))
    export CUDA_VISIBLE_DEVICES="${default_gpus}"
    echo "[INFO] 未检测到 CUDA_VISIBLE_DEVICES，默认使用 GPU:${default_gpus}。"
  else
    export CUDA_VISIBLE_DEVICES="0"
    echo "[INFO] 未检测到 CUDA_VISIBLE_DEVICES，默认使用 GPU:0。如需多卡请在运行前手动设置。"
  fi
else
  echo "[INFO] 使用外部指定的 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

IFS=',' read -ra __gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#__gpu_list[@]}" -lt "${NPROC}" ]]; then
  echo "[ERROR] NPROC=${NPROC} 但可见 GPU 数量仅为 ${#__gpu_list[@]}，请检查 CUDA_VISIBLE_DEVICES 设置。" >&2
  exit 1
fi

DATA_PATH="${ROOT_DIR}/dataset/FNSPID/trl_turn1/dpo_pairs_scaled.jsonl"
MODEL_PATH="${ROOT_DIR}/pretrain_model/ReasoningModel/Qwen3-1.7B"
OUTPUT_DIR="${ROOT_DIR}/src/data_generator/output/dpo_qwen3_turn1"
LOG_DIR="${ROOT_DIR}/src/data_generator/dpo_log"

EPOCHS="${DPO_EPOCHS:-10}"
LR="${DPO_LR:-5e-5}"
BETA="${DPO_BETA:-0.1}"
TRAIN_BATCH="${DPO_TRAIN_BATCH:-1}"
GRAD_ACCUM="${DPO_GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${DPO_MAX_SEQ_LEN:-2048}"
GRAD_CHECKPOINT="${DPO_GRAD_CHECKPOINT:-true}"
USE_4BIT="${DPO_USE_4BIT:-true}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

RESUME_ARG="--resume"
if [[ "${DPO_FRESH:-0}" == "1" ]]; then
  RESUME_ARG="--no-resume"
  echo "[INFO] 检测到 DPO_FRESH=1，禁用断点续训，从头开始训练。"
else
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    latest_ckpt=$(ls -1d "${OUTPUT_DIR}/checkpoint-"* | sort -V | tail -n1)
    echo "[INFO] 检测到已有检查点 ${latest_ckpt}，启用断点续训。"
  else
    echo "[INFO] 未发现历史检查点，将从头开始训练。"
  fi
fi

if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  echo "[INFO] 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减少显存碎片。"
fi

GRAD_ARG="--gradient-checkpointing"
if [[ "${GRAD_CHECKPOINT}" == "false" ]]; then
  GRAD_ARG="--no-gradient-checkpointing"
fi

FOURBIT_ARG="--load-4bit"
if [[ "${USE_4BIT}" == "false" ]]; then
  FOURBIT_ARG="--no-load-4bit"
fi

if [[ "${NPROC}" -gt 1 ]]; then
  TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
  if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN="$(command -v torchrun || true)"
  fi
  if [[ -z "${TORCHRUN_BIN}" ]]; then
    echo "[ERROR] 未找到 torchrun，可通过 'pip install torch' 或设置 PATH 后重试" >&2
    exit 1
  fi
  echo "[INFO] 检测到需要使用 ${NPROC} 张卡，使用 torchrun 启动分布式训练。"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT}"
  LAUNCH_CMD=("${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC}" --module data_generator.dpo_trainer --)
else
  LAUNCH_CMD=("${PYTHON_BIN}" -m data_generator.dpo_trainer)
fi

"${LAUNCH_CMD[@]}" \
  --data-path "$DATA_PATH" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --log-dir "$LOG_DIR" \
  --epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --beta "$BETA" \
  --train-batch "$TRAIN_BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --max-seq-length "$MAX_SEQ_LEN" \
  ${GRAD_ARG} \
  ${FOURBIT_ARG} \
  ${RESUME_ARG}

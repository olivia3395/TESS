#!/usr/bin/env bash
set -euo pipefail

# 中文说明：解析仓库根目录，确保脚本在任意路径下调用都能找到源码与数据
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
SCRIPT_NAME="$(basename "$0")"

GPU_ID=0
MODEL="MultiModal_Baseline"
DATASET="FNSPID"
EPOCH_OVERRIDE=1       
NUM_WORKERS=""       
PREFETCH_FACTOR=""
FULL_RUN=false

usage() {
  cat <<'EOF'
用法：scripts/train_patchtst.sh [选项]

选项：
  --gpu <id>              指定 GPU 编号（默认 0）
  --dataset <name>        指定数据集（默认 FNSPID）
  --epochs <n>            覆盖训练轮数；与 --full-run 互斥
  --num-workers <n>       指定 DataLoader worker 数
  --prefetch-factor <n>   指定 DataLoader 预取批次大小（需配合 worker>0 使用）
  --full-run              使用配置文件中的完整轮数训练
  -h, --help              显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="${2:-0}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:-FNSPID}"
      shift 2
      ;;
    --epochs)
      EPOCH_OVERRIDE="${2:-1}"
      FULL_RUN=false
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:-}"
      shift 2
      ;;
    --prefetch-factor)
      PREFETCH_FACTOR="${2:-}"
      shift 2
      ;;
    --full-run)
      FULL_RUN=true
      EPOCH_OVERRIDE=""
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[${SCRIPT_NAME}] 未识别的参数：$1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[${SCRIPT_NAME}] 未找到源码目录：${SRC_DIR}" >&2
  exit 1
fi

CMD=(python3 "${SRC_DIR}/model_trainer/main.py" --model "${MODEL}" --dataset "${DATASET}" --gpu "${GPU_ID}")

if [[ -n "${EPOCH_OVERRIDE}" ]]; then
  CMD+=(--epochs "${EPOCH_OVERRIDE}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  CMD+=(--num-workers "${NUM_WORKERS}")
fi
if [[ -n "${PREFETCH_FACTOR}" ]]; then
  CMD+=(--prefetch-factor "${PREFETCH_FACTOR}")
fi

# 中文说明：为模型代码设置 PYTHONPATH，避免外部环境干扰
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# 中文说明：执行训练命令，并在出错时立即停止
printf '[%s] 启动命令：%s\n' "${SCRIPT_NAME}" "${CMD[*]}"
"${CMD[@]}"

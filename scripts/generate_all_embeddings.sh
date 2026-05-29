#!/bin/bash

# generate_all_embeddings.sh - 为 FNSPID / Environment / Electricity / Bitcoin 四个数据集的 natural 和 structured 系列生成 embedding

set -e  # 遇到错误时退出

# 配置参数
MODEL_PATH="/public/home/maoyaoxin/llh/MMTSF/hf_home/Qwen3-Embedding-8B"
BATCH_SIZE=32
DEVICE="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"  # 或根据需要改为 "cpu"
DATASET_VERSION=""  # 如果需要指定版本可以在这里设置

# 数据集别名数组（四个数据集“三三组合”的 natural + structured 系列）
DATASETS=(
  "Bitcoin/ver_camf"
  "Environment/ver_camf"
  "Electricity/ver_camf"
  "Bitcoin/ver_shape_temporal_shape_volatility_structured"
  "Electricity/ver_global_temporal_shape_volatility_natural"
  "Environment/ver_global_temporal_shape_volatility_natural"

)

# 日志文件
LOG_FILE="embedding_generation.log"

# 创建日志目录
mkdir -p logs

echo "开始为以下数据集生成embedding:" | tee "$LOG_FILE"
printf '%s\n' "${DATASETS[@]}" | tee -a "$LOG_FILE"
echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "批次大小: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "设备: $DEVICE" | tee -a "$LOG_FILE"
echo "========================" | tee -a "$LOG_FILE"

# 逐个处理每个数据集
for DATASET_ALIAS in "${DATASETS[@]}"; do
    echo "[$(date)] 开始处理数据集: $DATASET_ALIAS" | tee -a "$LOG_FILE"
    
    # 构建命令
    CMD="python scripts/generate_qwen_embeddings.py \
        --alias $DATASET_ALIAS \
        --model-path $MODEL_PATH \
        --batch-size $BATCH_SIZE \
        --device $DEVICE"
    
    # 如果指定了版本，则添加版本参数
    if [ -n "$DATASET_VERSION" ]; then
        CMD="$CMD --dataset-version $DATASET_VERSION"
    fi
    
    echo "执行命令: $CMD" | tee -a "$LOG_FILE"
    
    # 执行命令并将输出保存到单独的日志文件
    DATASET_LOG="logs/$(echo $DATASET_ALIAS | tr '/' '_').log"
    
    if eval $CMD 2>&1 | tee "$DATASET_LOG"; then
        echo "[$(date)] 成功完成数据集 $DATASET_ALIAS 的embedding生成" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] 处理数据集 $DATASET_ALIAS 时出错" | tee -a "$LOG_FILE"
        echo "查看详细日志: $DATASET_LOG" | tee -a "$LOG_FILE"
        # 可选: 出错时停止执行
        # exit 1
    fi
    
    echo "------------------------" | tee -a "$LOG_FILE"
done

echo "[$(date)] 所有数据集处理完成" | tee -a "$LOG_FILE"
echo "总日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "各数据集详细日志位于 logs/ 目录中" | tee -a "$LOG_FILE"
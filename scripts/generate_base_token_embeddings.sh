#!/bin/bash

# generate_base_token_embeddings.sh - 为Electiricity数据集的基础版本（base）生成token embedding

set -e  # 遇到错误时退出

# 配置参数
BASE_DIR="dataset/Electricity"
FIELD_NAME="news"  # 字段名

# 基础版本数据集列表（6个）
DATASETS=(
"ver_global_shape"
"ver_global_temporal_shape"
"ver_global_volatility"
"ver_shape_temporal_shape"
"ver_volatility_shape"
"ver_volatility_temporal_shape"
)

# 日志文件
LOG_FILE="token_embedding_generation.log"

# 创建日志目录
mkdir -p logs

echo "开始为以下数据集生成token embedding:" | tee "$LOG_FILE"
printf '%s\n' "${DATASETS[@]}" | tee -a "$LOG_FILE"
echo "基础目录: $BASE_DIR" | tee -a "$LOG_FILE"
echo "字段名: $FIELD_NAME" | tee -a "$LOG_FILE"
echo "模型路径: 使用Python脚本默认路径LLM模型 (/ssd/hf_home/models/Qwen3-8B)" | tee -a "$LOG_FILE"
echo "========================" | tee -a "$LOG_FILE"

# 逐个处理每个数据集
for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_PATH="${BASE_DIR}/${DATASET_NAME}"
    
    echo "[$(date)] 开始处理数据集: $DATASET_NAME" | tee -a "$LOG_FILE"
    
    # 检查数据集目录是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        echo "[$(date)] 错误: 数据集目录不存在: $DATASET_PATH" | tee -a "$LOG_FILE"
        echo "跳过数据集: $DATASET_NAME" | tee -a "$LOG_FILE"
        echo "------------------------" | tee -a "$LOG_FILE"
        continue
    fi
    
    # 构建命令（使用Python脚本的默认模型路径）
    CMD="python scripts/generate_qwen_token_embeddings.py \
        --dataset-path $DATASET_PATH \
        --field-name $FIELD_NAME"
    
    echo "执行命令: $CMD" | tee -a "$LOG_FILE"
    
    # 执行命令并将输出保存到单独的日志文件
    DATASET_LOG="logs/token_$(echo $DATASET_NAME | tr '/' '_').log"
    
    if eval $CMD 2>&1 | tee "$DATASET_LOG"; then
        echo "[$(date)] 成功完成数据集 $DATASET_NAME 的token embedding生成" | tee -a "$LOG_FILE"
    else
        echo "[$(date)] 处理数据集 $DATASET_NAME 时出错" | tee -a "$LOG_FILE"
        echo "查看详细日志: $DATASET_LOG" | tee -a "$LOG_FILE"
        # 可选: 出错时停止执行
        # exit 1
    fi
    
    echo "------------------------" | tee -a "$LOG_FILE"
done

echo "[$(date)] 所有数据集处理完成" | tee -a "$LOG_FILE"
echo "总日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "各数据集详细日志位于 logs/ 目录中" | tee -a "$LOG_FILE"

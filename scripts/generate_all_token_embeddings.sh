#!/bin/bash

# generate_all_token_embeddings.sh - 为指定数据集别名数组生成token级别embeddings
# 使用 generate_qwen_token_level_embeddings.py 脚本

set -e  # 遇到错误时退出

# 配置参数
MODEL_PATH="/ssd/hf_home/models/Qwen3-8B"  # 或根据需要修改为Qwen3-Embedding模型
MAX_LENGTH=256
BATCH_SIZE=32
FIELD_NAME="news"  # 要生成embeddings的字段名

# 数据集别名数组（可以根据需要修改）
DATASETS=(
    "FNSPID/ver_global_shape_volatility_natural"
    "FNSPID/ver_global_shape_volatility_structured"
    "FNSPID/ver_global_temporal_shape_volatility_natural"
    "FNSPID/ver_global_temporal_shape_volatility_structured"
    "FNSPID/ver_shape_temporal_shape_volatility_natural"
    "FNSPID/ver_shape_temporal_shape_volatility_structured"
)

# 日志文件
LOG_FILE="token_embedding_generation.log"

# 创建日志目录
mkdir -p logs

echo "开始为以下数据集生成token级别embeddings:" | tee "$LOG_FILE"
printf '%s\n' "${DATASETS[@]}" | tee -a "$LOG_FILE"
echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "最大序列长度: $MAX_LENGTH" | tee -a "$LOG_FILE"
echo "批次大小: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "字段名: $FIELD_NAME" | tee -a "$LOG_FILE"
echo "========================" | tee -a "$LOG_FILE"

# 逐个处理每个数据集
for DATASET_ALIAS in "${DATASETS[@]}"; do
    echo "[$(date)] 开始处理数据集: $DATASET_ALIAS" | tee -a "$LOG_FILE"

    # 从数据集别名提取数据集名称
    # 例如: FNSPID/ver_global_shape_volatility_natural -> FNSPID
    DATASET_NAME=$(echo "$DATASET_ALIAS" | cut -d'/' -f1)

    # 构建数据集路径
    DATASET_PATH="dataset/$DATASET_NAME"

    echo "数据集名称: $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "数据集路径: $DATASET_PATH" | tee -a "$LOG_FILE"

    # 检查数据集路径是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        echo "[$(date)] 错误: 数据集路径 $DATASET_PATH 不存在" | tee -a "$LOG_FILE"
        continue
    fi

    # 构建命令
    CMD="python scripts/generate_qwen_token_level_embeddings.py \
        --dataset-path $DATASET_PATH \
        --field-name $FIELD_NAME \
        --model-path $MODEL_PATH \
        --max-length $MAX_LENGTH \
        --batch-size $BATCH_SIZE"

    echo "执行命令: $CMD" | tee -a "$LOG_FILE"

    # 执行命令并将输出保存到单独的日志文件
    DATASET_LOG="logs/$(echo $DATASET_ALIAS | tr '/' '_')_token.log"

    if eval $CMD 2>&1 | tee "$DATASET_LOG"; then
        echo "[$(date)] 成功完成数据集 $DATASET_ALIAS 的token级别embedding生成" | tee -a "$LOG_FILE"

        # 检查输出文件是否存在
        OUTPUT_FILE="$DATASET_PATH/embedding_qwen/all_token_embeddings.pt"
        if [ -f "$OUTPUT_FILE" ]; then
            echo "输出文件: $OUTPUT_FILE" | tee -a "$LOG_FILE"
            # 显示文件大小
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo "文件大小: $FILE_SIZE" | tee -a "$LOG_FILE"
        else
            echo "警告: 输出文件 $OUTPUT_FILE 未找到" | tee -a "$LOG_FILE"
        fi
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



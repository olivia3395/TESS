#!/bin/bash

# generate_token_embeddings_batch.sh - 批量生成指定数据集别名的token级别embeddings
# 用法: ./generate_token_embeddings_batch.sh [dataset_alias1] [dataset_alias2] ...
# 例如: ./generate_token_embeddings_batch.sh FNSPID/ver_camf Environment/ver_temp

# set -e  # 注释掉，允许单个数据集失败时继续处理其他数据集

# 默认配置参数
MODEL_PATH="/public/home/maoyaoxin/llh/MMTSF/hf_home/Qwen3-8B"
MAX_LENGTH=24
BATCH_SIZE=32
FIELD_NAME="news"

# 函数：显示使用说明
show_usage() {
    echo "用法: $0 [选项] [数据集别名...]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path PATH    模型路径 (默认: $MODEL_PATH)"
    echo "  -l, --max-length LENGTH  最大序列长度 (默认: $MAX_LENGTH)"
    echo "  -b, --batch-size SIZE    批次大小 (默认: $BATCH_SIZE)"
    echo "  -f, --field-name NAME    字段名 (默认: $FIELD_NAME)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "数据集别名示例:"
    echo "  FNSPID/ver_camf"
    echo "  Environment/ver_temp"
    echo "  Electricity/best"
    echo ""
    echo "如果不指定数据集别名，将使用内置的默认列表。"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -f|--field-name)
            FIELD_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "错误: 未知选项 $1"
            show_usage
            exit 1
            ;;
        *)
            # 非选项参数，应该是数据集别名
            break
            ;;
    esac
done

# 如果没有指定数据集别名，使用默认列表
if [ $# -eq 0 ]; then
    echo "未指定数据集别名，使用默认列表..."
    DATASETS=(
        "Electricity/ver_synchronized_global_shape_temporal_shape"
        # "Electricity/ver_synchronized_global_temporal_shape_volatility"
        "Electricity/ver_synchronized_global_shape_volatility"
        "Electricity/ver_synchronized_shape_temporal_shape_volatility"
        # "Electricity/ver_global_shape_temporal_shape"
        # "Electricity/ver_global_temporal_shape_volatility"
        # "Electricity/ver_global_shape_volatility"
        # "Electricity/ver_shape_temporal_shape_volatility"

    )
else
    # 使用命令行指定的数据集别名
    DATASETS=("$@")
fi

# 日志文件
LOG_FILE="token_embedding_batch_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p logs

echo "开始批量生成token级别embeddings:" | tee "$LOG_FILE"
echo "配置参数:" | tee -a "$LOG_FILE"
echo "  模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  最大序列长度: $MAX_LENGTH" | tee -a "$LOG_FILE"
echo "  批次大小: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  字段名: $FIELD_NAME" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "数据集列表:" | tee -a "$LOG_FILE"
printf '  %s\n' "${DATASETS[@]}" | tee -a "$LOG_FILE"
echo "========================" | tee -a "$LOG_FILE"

# 统计信息
TOTAL_DATASETS=${#DATASETS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

# 逐个处理每个数据集
for DATASET_ALIAS in "${DATASETS[@]}"; do
    echo "[$(date)] [$((++CURRENT_INDEX))/$TOTAL_DATASETS] 开始处理数据集: $DATASET_ALIAS" | tee -a "$LOG_FILE"

    # 从数据集别名提取数据集名称
    DATASET_NAME=$(echo "$DATASET_ALIAS" | cut -d'/' -f1)

    # 构建数据集路径（使用完整的别名路径，包含版本子目录）
    DATASET_PATH="dataset/$DATASET_ALIAS"

    echo "  数据集名称: $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "  数据集路径: $DATASET_PATH" | tee -a "$LOG_FILE"

    # 检查数据集路径是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        echo "  [$(date)] 错误: 数据集路径 $DATASET_PATH 不存在" | tee -a "$LOG_FILE"
        ((FAIL_COUNT++))
        echo "------------------------" | tee -a "$LOG_FILE"
        continue
    fi

    # 构建命令
    CMD="python scripts/generate_qwen_token_level_embeddings.py \
        --dataset-path $DATASET_PATH \
        --field-name $FIELD_NAME \
        --model-path $MODEL_PATH \
        --max-length $MAX_LENGTH \
        --batch-size $BATCH_SIZE"

    echo "  执行命令: $CMD" | tee -a "$LOG_FILE"

    # 执行命令并将输出保存到单独的日志文件
    DATASET_LOG="logs/$(echo $DATASET_ALIAS | tr '/' '_')_token_$(date +%H%M%S).log"

    # 执行命令并捕获退出码（全局set -e已禁用，确保单个数据集失败不影响其他数据集）
    eval $CMD 2>&1 | tee "$DATASET_LOG"
    CMD_EXIT_CODE=${PIPESTATUS[0]}  # 获取eval的退出码，而不是tee的
    
    if [ $CMD_EXIT_CODE -eq 0 ]; then
        echo "  [$(date)] ✓ 成功完成数据集 $DATASET_ALIAS" | tee -a "$LOG_FILE"

        # 检查输出文件是否存在
        OUTPUT_FILE="$DATASET_PATH/embedding_qwen/all_token_embeddings.pt"
        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo "  输出文件: $OUTPUT_FILE (${FILE_SIZE})" | tee -a "$LOG_FILE"
        else
            echo "  警告: 输出文件 $OUTPUT_FILE 未找到" | tee -a "$LOG_FILE"
        fi

        ((SUCCESS_COUNT++))
    else
        echo "  [$(date)] ✗ 处理数据集 $DATASET_ALIAS 时出错" | tee -a "$LOG_FILE"
        echo "  查看详细日志: $DATASET_LOG" | tee -a "$LOG_FILE"
        ((FAIL_COUNT++))
    fi

    echo "------------------------" | tee -a "$LOG_FILE"
done

# 输出统计信息
echo "[$(date)] 批量处理完成" | tee -a "$LOG_FILE"
echo "统计信息:" | tee -a "$LOG_FILE"
echo "  总数据集数: $TOTAL_DATASETS" | tee -a "$LOG_FILE"
echo "  成功: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "  失败: $FAIL_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "总日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "各数据集详细日志位于 logs/ 目录中" | tee -a "$LOG_FILE"

# 根据结果设置退出码
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有数据集处理成功！" | tee -a "$LOG_FILE"
    exit 0
else
    echo "⚠️  有 $FAIL_COUNT 个数据集处理失败，请检查日志。" | tee -a "$LOG_FILE"
    exit 1
fi

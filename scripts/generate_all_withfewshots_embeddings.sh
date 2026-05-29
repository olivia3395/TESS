#!/bin/bash

# 生成六个新注册数据集的embedding
# 该脚本会循环调用generate_qwen_embeddings.py来生成embedding

echo "开始生成新数据集的embedding..."

# 定义数据集别名数组
DATASETS=(
    "FNSPID/ver_camf_case"
    # "FNSPID/ver_8BInstruct_withfewshots_globalonly_case"
    # "FNSPID/ver_8BInstruct_withfewshots_trendonly_case"

)

# 循环处理每个数据集
for dataset in "${DATASETS[@]}"; do
    echo "=================================================="
    echo "正在处理数据集: $dataset"
    echo "=================================================="
    
    python scripts/generate_qwen_embeddings.py \
        --alias "$dataset" \
        --model-path "pretrain_model/EmbeddingModel/Qwen3-Embedding-8B" \
        --batch-size 32\
        --device "cuda:0"
    
    # 检查命令执行状态
    if [ $? -eq 0 ]; then
        echo "✅ 成功生成 $dataset 的embedding"
    else
        echo "❌ 生成 $dataset 的embedding时出错"
        exit 1
    fi
    
    echo "" # 空行分隔
done

echo "=================================================="
echo "所有数据集的embedding生成完成!"
echo "=================================================="
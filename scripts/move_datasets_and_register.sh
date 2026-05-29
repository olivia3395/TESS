#!/bin/bash

# move_datasets_and_register.sh - 移动12个natural和structured系列数据集并注册

set -e  # 遇到错误时退出

# 定义源目录和目标目录
SOURCE_DIR="dataset/FNSPID/label"
TARGET_DIR="dataset/FNSPID"

# 定义12个需要移动的数据集名称
DATASETS=(
  "ver_global_shape_natural"
  "ver_global_temporal_natural"
  "ver_global_volatility_natural"
  "ver_shape_temporal_natural"
  "ver_shape_volatility_natural"
  "ver_temporal_volatility_natural"
  "ver_global_shape_structured"
  "ver_global_temporal_structured"
  "ver_global_volatility_structured"
  "ver_shape_temporal_structured"
  "ver_shape_volatility_structured"
  "ver_temporal_volatility_structured"
)

# 输出要处理的数据集列表
echo "将要移动的12个数据集:"
for dataset in "${DATASETS[@]}"; do
  echo "  - $dataset"
done

echo ""
echo "请手动执行以下操作："
echo "1. 将以下12个数据集从 $SOURCE_DIR 移动到 $TARGET_DIR:"
for dataset in "${DATASETS[@]}"; do
  echo "   mv $SOURCE_DIR/$dataset $TARGET_DIR/$dataset"
done

echo ""
echo "2. 将以下条目添加到 src/model_trainer/configs/dataset/index.yaml 中:"
for dataset in "${DATASETS[@]}"; do
  echo "
  FNSPID/$dataset:
    dataset_name: FNSPID
    embeddings:
      news:
        path: $dataset/embedding_qwen/all_embeddings.pt
        splits:
          test: test_news
          train: train_news
          vali: vali_news
    root: dataset/FNSPID
    splits:
      test: $dataset/test.json
      train: $dataset/train.json
      vali: $dataset/vali.json
    version: $dataset"
done

echo ""
echo "操作完成！"
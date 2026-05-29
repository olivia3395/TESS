# Token-Level Embedding Generation Script

这个脚本基于原始的 `generate_qwen_token_embeddings.py`，但去掉了池化过程，直接保存token级别的embeddings和attention masks。

## 📄 输出格式

新的嵌套字典格式：
```python
{
    'train': {
        'embeddings': torch.Tensor([N, L, D]),     # token级别embeddings
        'attention_mask': torch.Tensor([N, L]),     # attention mask, 1=真实token, 0=padding
    },
    'vali': {
        'embeddings': torch.Tensor([N, L, D]),
        'attention_mask': torch.Tensor([N, L]),
    },
    'test': {
        'embeddings': torch.Tensor([N, L, D]),
        'attention_mask': torch.Tensor([N, L]),
    }
}
```

## 🚀 使用方法

### 单个数据集

```bash
cd /path/to/MMTSF_LIB

python scripts/generate_qwen_token_level_embeddings.py \
    --dataset-path dataset/FNSPID \
    --field-name news \
    --max-length 256 \
    --batch-size 32
```

### 批量处理数据集

#### 方法1：使用默认数据集列表

```bash
./scripts/generate_all_token_embeddings.sh
```

#### 方法2：指定自定义数据集列表

```bash
./scripts/generate_token_embeddings_batch.sh \
    FNSPID/ver_camf \
    Environment/ver_temp \
    Electricity/best
```

#### 方法3：使用自定义参数

```bash
./scripts/generate_token_embeddings_batch.sh \
    --model-path /path/to/model \
    --max-length 512 \
    --batch-size 16 \
    --field-name news \
    FNSPID/ver_camf \
    Environment/ver_temp
```

### 参数说明

- `--dataset-path`: 数据集路径，包含 `train.json`, `vali.json`, `test.json`
- `--field-name`: 要生成embeddings的字段名（如 'news'）
- `--model-path`: Qwen3模型路径（默认: `/ssd/hf_home/models/Qwen3-8B`）
- `--max-length`: 最大序列长度（默认: 512）
- `--batch-size`: 批处理大小（默认: 8）

### 示例

```bash
python scripts/generate_qwen_token_level_embeddings.py \
    --dataset-path dataset/FNSPID \
    --field-name news \
    --max-length 256 \
    --batch-size 16
```

## 📁 输出文件

### 单个数据集处理

脚本会在 `dataset-path/embedding_qwen/` 目录下生成：

```
dataset/FNSPID/embedding_qwen/
└── all_token_embeddings.pt
```

### 批量处理

批量脚本会为每个数据集分别生成embeddings：

```
dataset/
├── FNSPID/embedding_qwen/all_token_embeddings.pt
├── Environment/embedding_qwen/all_token_embeddings.pt
└── Electricity/embedding_qwen/all_token_embeddings.pt
```

## 🔧 集成到训练流程

### 1. 更新数据集配置

在 `configs/dataset/index.yaml` 中添加：

```yaml
aliases:
  FNSPID/token_level:
    dataset_name: FNSPID
    embeddings:
      news:
        path: embedding_qwen/all_token_embeddings.pt
        splits:
          test: test
          train: train
          vali: vali
    root: dataset/FNSPID
    splits:
      test: test.json
      train: train.json
      vali: vali.json
```

### 2. 使用动态dropout模型

```bash
PYTHONPATH=src python src/model_trainer/main.py \
    --model MultiModal_Baseline_Dynamic_Dropout \
    --dataset FNSPID \
    --dataset-alias FNSPID/token_level \
    --gpu 0
```

## ⚡ 性能优化

- **批处理**: 使用 `--batch-size` 参数控制批大小，平衡内存和速度
- **序列长度**: 使用 `--max-length` 控制最大序列长度，避免过长的padding
- **GPU加速**: 如果有GPU，会自动使用（通过transformers库）

## 🔍 输出验证

运行后检查输出：

```python
import torch
data = torch.load('dataset/FNSPID/embedding_qwen/all_token_embeddings.pt')
print(data['train']['embeddings'].shape)  # [N, L, D]
print(data['train']['attention_mask'].shape)  # [N, L]
```

## 📊 与原脚本的区别

| 特性 | 原脚本 | 新脚本 |
|------|--------|--------|
| 输出维度 | [N, D] | [N, L, D] |
| 池化方式 | avg/max pooling | 无池化，保留所有tokens |
| Attention Mask | 无 | 有，标记真实tokens |
| 文件格式 | 平坦字典 | 嵌套字典 |
| 适用模型 | 句子级别模型 | Token级别模型 |

## 🐛 故障排除

1. **内存不足**: 减小 `--batch-size` 或 `--max-length`
2. **模型加载失败**: 检查 `--model-path` 是否正确
3. **字段不存在**: 确认数据中的字段名与 `--field-name` 匹配
4. **磁盘空间**: Token级别embeddings会比句子级别大得多

## 🎯 最佳实践

1. **序列长度**: 根据你的数据分布选择合适的 `--max-length`
2. **批大小**: 在内存允许的情况下尽量增大以加速处理
3. **字段选择**: 确保 `--field-name` 对应的字段包含文本数据
4. **存储空间**: Token级别embeddings通常是句子级别的 L 倍大

## 🔄 批量处理最佳实践

### 内存管理
- **GPU内存**: 如果有多个GPU，可以使用 `CUDA_VISIBLE_DEVICES` 指定特定GPU
- **CPU内存**: 大数据集建议适当调小 `--batch-size`
- **磁盘空间**: Token级别embeddings文件较大，注意存储空间

### 并行处理
```bash
# 后台运行批量处理
nohup ./scripts/generate_token_embeddings_batch.sh \
    FNSPID/ver_camf Environment/ver_temp > batch_log.txt 2>&1 &
```

### 错误恢复
- 脚本会在出错时继续处理其他数据集
- 检查 `logs/` 目录下的详细日志
- 失败的数据集可以单独重新运行

### 监控进度
```bash
# 查看实时进度
tail -f token_embedding_batch_*.log

# 查看各数据集状态
ls -la logs/*_token_*.log
```

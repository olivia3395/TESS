# BERT模型下载指南

## 问题说明

当使用 `MultiModal_Baseline_Dynamic_Dropout_BERT` 模型时，如果网络无法访问 Hugging Face Hub，会出现SSL连接错误。需要预先下载BERT模型到本地。

## 下载方法

### 方法1：使用下载脚本（推荐）

使用项目提供的下载脚本：

```bash
cd /public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB
python scripts/download_bert_model.py
```

这将下载 `bert-base-uncased` 模型到 `/ssd/hf_home/models/bert-base-uncased/`

**自定义参数：**

```bash
# 下载其他BERT模型
python scripts/download_bert_model.py --model-name bert-large-uncased

# 指定输出目录
python scripts/download_bert_model.py --output-dir /your/custom/path
```

### 方法2：使用Python直接下载

```python
from transformers import BertModel, BertTokenizer
import os

# 设置输出路径
output_dir = '/ssd/hf_home/models/bert-base-uncased'
os.makedirs(output_dir, exist_ok=True)

# 下载并保存
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"模型已保存到: {output_dir}")
```

### 方法3：使用命令行工具（如果已安装huggingface-cli）

```bash
huggingface-cli download bert-base-uncased --local-dir /ssd/hf_home/models/bert-base-uncased
```

## 配置使用

### 方法A：使用默认路径（推荐）

如果模型下载到 `/ssd/hf_home/models/bert-base-uncased`，代码会自动检测并使用，**无需额外配置**。

### 方法B：在配置文件中指定路径

如果模型下载到其他位置，在配置文件中设置：

```yaml
# MultiModal_Baseline_Dynamic_Dropout_BERT.yaml
bert_model_path: /path/to/your/bert-base-uncased
```

## 模型文件结构

下载后的模型目录应包含以下文件：

```
bert-base-uncased/
├── config.json          # 模型配置
├── pytorch_model.bin    # 模型权重（或 model.safetensors）
├── vocab.txt            # 词汇表
└── tokenizer_config.json # Tokenizer配置
```

## 验证下载

检查模型是否下载成功：

```bash
ls -lh /ssd/hf_home/models/bert-base-uncased/
```

应该看到上述文件都存在。

## 测试加载

```python
from transformers import BertModel
model = BertModel.from_pretrained('/ssd/hf_home/models/bert-base-uncased', local_files_only=True)
print("✓ 模型加载成功！")
```

## 网络问题处理

如果下载时遇到网络问题：

1. **使用代理：**
   ```bash
   export http_proxy=your_proxy_url
   export https_proxy=your_proxy_url
   python scripts/download_bert_model.py
   ```

2. **使用镜像站点：**
   ```python
   # 设置环境变量
   export HF_ENDPOINT=https://hf-mirror.com
   python scripts/download_bert_model.py
   ```

3. **手动下载：**
   - 访问 https://huggingface.co/bert-base-uncased
   - 下载所有文件到 `/ssd/hf_home/models/bert-base-uncased/`

## 常见问题

**Q: 下载后仍然报错？**
A: 确保模型路径正确，并且所有必需文件都已下载。

**Q: 可以使用其他BERT模型吗？**
A: 可以，但需要确保维度匹配。代码中使用 `bert-base-uncased` (768维)，如果使用其他模型，需要修改 `bert_dim` 参数。

**Q: 模型会自动更新吗？**
A: 不会。如果使用本地路径（`bert_model_path`），模型不会自动更新。需要手动重新下载。







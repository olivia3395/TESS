#!/usr/bin/env python3
"""
Generate sentence-level embeddings using Qwen3-8B word embeddings + BERT Transformer.
流程: Qwen Embedding (4096) -> 投影到BERT维度 (768) -> BERT Encoder -> CLS提取 -> 投影到4096维
输出: [batch_size, 4096] 形状的embeddings
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BertModel


class BertTransformerEncoder(nn.Module):
    """
    BERT Transformer编码器，用于将Qwen embeddings转换为BERT空间并编码。
    流程: Qwen Embedding (4096) -> 投影 (768) -> 添加CLS -> BERT -> 提取CLS -> 投影 (4096)
    """
    def __init__(self, qwen_dim=4096, bert_dim=768, max_length=512, bert_model_name='bert-base-uncased'):
        super(BertTransformerEncoder, self).__init__()
        
        self.qwen_dim = qwen_dim
        self.bert_dim = bert_dim
        self.max_length = max_length
        
        # 投影层：Qwen 4096 -> BERT 768
        self.projection = nn.Linear(qwen_dim, bert_dim)
        
        # 加载BERT模型
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        # 冻结BERT参数（不做MLM预训练）
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # CLS token embedding（可学习的，用于序列开头）
        cls_token_init = torch.randn(1, 1, bert_dim) * 0.02  # 小随机初始化
        self.cls_token_embedding = nn.Parameter(cls_token_init)
        
        # 最终投影层：BERT 768 -> 4096
        self.final_projection = nn.Linear(bert_dim, qwen_dim)
        
    def forward(self, word_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_embeddings: [batch_size, seq_len, qwen_dim] Qwen word embeddings
            attention_mask: [batch_size, seq_len] attention mask (1 for real tokens, 0 for padding)
        Returns:
            cls_embeddings: [batch_size, qwen_dim] CLS token embeddings after projection to 4096
        """
        batch_size, seq_len, _ = word_embeddings.shape
        
        # 1. 投影到BERT维度
        bert_embeddings = self.projection(word_embeddings)  # [B, L, 768]
        
        # 2. 添加CLS token在序列开头
        cls_tokens = self.cls_token_embedding.expand(batch_size, -1, -1)  # [B, 1, 768]
        sequence_embeddings = torch.cat([cls_tokens, bert_embeddings], dim=1)  # [B, L+1, 768]
        
        # 3. 扩展attention_mask，在开头添加CLS token的位置（CLS总是有效的）
        # attention_mask格式: 1=真实token, 0=padding token
        # 例如: [1, 1, 1, ..., 1, 0, 0, ..., 0]  (前N个是真实token，后面是padding)
        cls_attention_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([cls_attention_mask, attention_mask], dim=1)  # [B, L+1]
        # 扩展后: [1, 1, 1, ..., 1, 0, 0, ..., 0] (CLS + 真实tokens + padding)
        
        # 4. 通过BERT模型（使用inputs_embeds参数）
        # BERT内部对attention_mask的处理机制：
        # 1. BERT会将attention_mask从[B, L+1]转换为[B, 1, 1, L+1]格式
        # 2. 将0值（padding）转换为-10000.0，1值（真实token）保持为0
        # 3. 在self-attention计算中：
        #    - 计算attention_scores = Q @ K^T / sqrt(d_k)
        #    - 加上mask: attention_scores = attention_scores + mask_bias
        #    - padding位置的score变成极小值（-10000），经过softmax后权重≈0
        #    - 真实token位置的权重正常分布
        # 4. 这样padding token就不会参与attention计算，也不会影响输出embedding
        bert_outputs = self.bert_model(
            inputs_embeds=sequence_embeddings,
            attention_mask=extended_attention_mask,  # BERT内部会自动转换为attention bias来屏蔽padding
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        bert_output = bert_outputs.last_hidden_state  # [B, L+1, 768]
        
        # 5. 提取CLS token（第一个位置的embedding）
        cls_embedding = bert_output[:, 0, :]  # [B, 768]
        
        # 6. 投影回4096维
        final_embedding = self.final_projection(cls_embedding)  # [B, 4096]
        
        return final_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings using Qwen + BERT Transformer")
    parser.add_argument('--qwen-model-path', default='/ssd/hf_home/models/Qwen3-8B',
                       help='Local Qwen3-8B model path (default: /ssd/hf_home/models/Qwen3-8B)')
    parser.add_argument('--bert-model-name', default='bert-base-uncased',
                       help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('--dataset-path', required=True,
                       help='Dataset path containing the specified field')
    parser.add_argument('--field-name', required=True,
                       help='Field name to generate embeddings for')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length for tokenization (default: 512)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available else cpu)')
    return parser.parse_args()


def load_text_data(dataset_path: str, field_name: str) -> Dict[str, List[str]]:
    """Load text data from dataset."""
    splits = ['train.json', 'vali.json', 'test.json']
    text_data = {}

    for split in splits:
        file_path = os.path.join(dataset_path, split)
        abs_file_path = os.path.abspath(file_path)
        if not os.path.exists(abs_file_path):
            print(f"Warning: File {abs_file_path} does not exist, skipping.")
            continue

        with open(abs_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [item[field_name] for item in data if field_name in item]
        text_data[split.replace('.json', '')] = texts
        print(f"Loaded {len(texts)} {field_name} entries from {split}")

    return text_data


def generate_embeddings_batch(
    texts: List[str], 
    qwen_model_path: str,
    bert_transformer: BertTransformerEncoder,
    max_length: int, 
    batch_size: int,
    device: str
) -> torch.Tensor:
    """
    Generate sentence-level embeddings for a list of texts.
    Returns: embeddings shaped [batch_size, 4096]
    """
    abs_model_path = os.path.abspath(qwen_model_path)

    # Load Qwen tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, trust_remote_code=True)

    # Try loading as CausalLM first (for Qwen3-8B), fallback to AutoModel
    try:
        qwen_model = AutoModelForCausalLM.from_pretrained(abs_model_path, trust_remote_code=True)
    except:
        qwen_model = AutoModel.from_pretrained(abs_model_path, trust_remote_code=True)

    # Get embedding layer
    embedding_layer = qwen_model.get_input_embeddings()
    embedding_layer = embedding_layer.to(device)
    bert_transformer = bert_transformer.to(device)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize batch
        tokens = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

        input_ids = tokens['input_ids'].to(device)  # Shape: [batch_size, seq_len]
        attention_mask = tokens['attention_mask'].to(device)  # Shape: [batch_size, seq_len]

        # Get Qwen word embeddings
        with torch.no_grad():
            word_embeddings = embedding_layer(input_ids)  # Shape: [batch_size, seq_len, 4096]
            
            # 通过BERT Transformer
            sentence_embeddings = bert_transformer(word_embeddings, attention_mask)  # [batch_size, 4096]

        all_embeddings.append(sentence_embeddings.cpu())

        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}, "
              f"output shape: {sentence_embeddings.shape}")

    # Concatenate all batches
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings
    else:
        return torch.empty(0, 4096)


def save_embeddings(embeddings_dict: Dict[str, torch.Tensor], output_path: str, field_name: str):
    """
    Save sentence-level embeddings in nested format.
    Format:
    {
        'train': {
            'embeddings': torch.Tensor([N, 4096]),
        },
        'vali': {...},
        'test': {...}
    }
    """
    # Convert to nested format (注意：这里不再需要attention_mask，因为输出是句子级别)
    nested_dict = {}
    for split, embeddings in embeddings_dict.items():
        nested_dict[split] = {
            'embeddings': embeddings,
        }

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    torch.save(nested_dict, abs_output_path)
    print(f"Embeddings saved to {abs_output_path}")
    print(f"Output directory: {os.path.dirname(abs_output_path)}")

    for split, data in nested_dict.items():
        embeddings_shape = data['embeddings'].shape
        print(f"{split} - embeddings: {embeddings_shape}")


def main():
    args = parse_args()

    # Load text data
    print(f"Loading {args.field_name} data...")
    text_data = load_text_data(args.dataset_path, args.field_name)

    if not text_data:
        print("No data loaded, exiting.")
        return

    # Initialize BERT Transformer encoder
    print(f"Initializing BERT Transformer encoder (BERT model: {args.bert_model_name})...")
    bert_transformer = BertTransformerEncoder(
        qwen_dim=4096,
        bert_dim=768,
        max_length=args.max_length,
        bert_model_name=args.bert_model_name
    )
    bert_transformer.eval()  # 设置为评估模式
    
    print(f"Using device: {args.device}")

    # Create embedding output directory
    embedding_dir = os.path.join(args.dataset_path, "embedding_qwen_bert")
    os.makedirs(embedding_dir, exist_ok=True)

    # Generate embeddings
    print("Generating sentence-level embeddings with BERT Transformer...")
    embeddings_dict = {}
    for split, texts in text_data.items():
        if not texts:
            print(f"Skipping empty {split} split")
            continue

        print(f"Processing {split} split with {len(texts)} entries...")
        embeddings = generate_embeddings_batch(
            texts, args.qwen_model_path, bert_transformer, args.max_length, args.batch_size, args.device
        )
        embeddings_dict[split] = embeddings

    # Save embeddings
    output_path = os.path.join(embedding_dir, "all_bert_transformer_embeddings.pt")
    save_embeddings(embeddings_dict, output_path, args.field_name)

    print("Done!")


if __name__ == "__main__":
    main()

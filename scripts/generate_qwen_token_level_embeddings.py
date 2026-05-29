#!/usr/bin/env python3
"""
Generate token-level embeddings using Qwen3-8B vocabulary for specified field.
Outputs token embeddings with attention masks in the new nested format.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qwen3-8B token-level embeddings for specified field")
    parser.add_argument('--model-path', default='/ssd/hf_home/models/Qwen3-8B',
                       help='Local Qwen3-8B model path (default: /ssd/hf_home/models/Qwen3-8B)')
    parser.add_argument('--dataset-path', required=True,
                       help='Dataset path containing the specified field')
    parser.add_argument('--field-name', required=True,
                       help='Field name to generate embeddings for')
    parser.add_argument('--max-length', type=int, default=24,
                       help='Maximum sequence length for tokenization (default: 512)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 8)')
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


def generate_token_embeddings_batch(texts: List[str], model_path: str, max_length: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate token-level embeddings for a list of texts using Qwen3-8B.
    Returns: (embeddings, attention_masks) both shaped [batch_size, seq_len, hidden_dim] and [batch_size, seq_len]
    """
    # Resolve absolute path
    abs_model_path = os.path.abspath(model_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, trust_remote_code=True)

    # Try loading as CausalLM first (for Qwen3-8B), fallback to AutoModel (for Embedding models)
    try:
        model = AutoModelForCausalLM.from_pretrained(abs_model_path, trust_remote_code=True)
    except:
        model = AutoModel.from_pretrained(abs_model_path, trust_remote_code=True)

    # Get embedding layer
    embedding_layer = model.get_input_embeddings()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings = []
    all_attention_masks = []

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

        input_ids = tokens['input_ids']  # Shape: [batch_size, seq_len]
        attention_mask = tokens['attention_mask']  # Shape: [batch_size, seq_len]

        # Get token embeddings
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids)  # Shape: [batch_size, seq_len, hidden_size]

        all_embeddings.append(token_embeddings)
        all_attention_masks.append(attention_mask)

        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}, "
              f"shape: {token_embeddings.shape}")

    # Concatenate all batches
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        final_attention_masks = torch.cat(all_attention_masks, dim=0)
        return final_embeddings, final_attention_masks
    else:
        # Get hidden size from model config
        try:
            hidden_size = getattr(model.config, 'hidden_size', 4096)
        except:
            hidden_size = 4096
        return torch.empty(0, max_length, hidden_size), torch.empty(0, max_length, dtype=torch.long)


def save_token_embeddings(embeddings_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                         output_path: str, field_name: str):
    """
    Save token embeddings in the new nested format with attention masks.
    Format:
    {
        'train': {
            'embeddings': torch.Tensor([N, L, D]),
            'attention_mask': torch.Tensor([N, L])
        },
        'vali': {...},
        'test': {...}
    }
    """
    # Convert to nested format
    nested_dict = {}
    for split, (embeddings, attention_mask) in embeddings_dict.items():
        nested_dict[split] = {
            'embeddings': embeddings,
            'attention_mask': attention_mask
        }

    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    torch.save(nested_dict, abs_output_path)
    print(f"Token embeddings saved to {abs_output_path}")
    print(f"Output directory: {os.path.dirname(abs_output_path)}")

    for split, data in nested_dict.items():
        embeddings_shape = data['embeddings'].shape
        mask_shape = data['attention_mask'].shape
        print(f"{split} - embeddings: {embeddings_shape}, attention_mask: {mask_shape}")


def main():
    args = parse_args()

    # Load text data
    print(f"Loading {args.field_name} data...")
    text_data = load_text_data(args.dataset_path, args.field_name)

    if not text_data:
        print("No data loaded, exiting.")
        return

    # Create embedding output directory
    embedding_dir = os.path.join(args.dataset_path, "embedding_qwen")
    os.makedirs(embedding_dir, exist_ok=True)

    # Generate token-level embeddings
    print("Generating token-level embeddings...")
    token_embeddings = {}
    for split, texts in text_data.items():
        if not texts:
            print(f"Skipping empty {split} split")
            continue

        print(f"Processing {split} split with {len(texts)} entries...")
        embeddings, attention_masks = generate_token_embeddings_batch(
            texts, args.model_path, args.max_length, args.batch_size
        )
        token_embeddings[split] = (embeddings, attention_masks)

    # Save token embeddings
    output_path = os.path.join(embedding_dir, "all_token_embeddings.pt")
    save_token_embeddings(token_embeddings, output_path, args.field_name)

    print("Done!")


if __name__ == "__main__":
    main()

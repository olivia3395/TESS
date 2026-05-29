#!/usr/bin/env python3
"""
Generate token embeddings using Qwen3-8B vocabulary for specified field.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Qwen3-8B token embeddings for specified field")
    parser.add_argument('--model-path', default='/ssd/hf_home/models/Qwen3-8B', 
                       help='Local Qwen3-8B model path (default: /ssd/hf_home/models/Qwen3-8B)')
    parser.add_argument('--dataset-path', required=True,
                       help='Dataset path containing the specified field')
    parser.add_argument('--field-name', required=True,
                       help='Field name to generate embeddings for')
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


def generate_embeddings_for_split(texts: List[str], model_path: str) -> torch.Tensor:
    """Generate average pooled embeddings for a list of texts using Qwen3-8B."""
    # Resolve absolute path
    abs_model_path = os.path.abspath(model_path)
    
    # Load tokenizer and model with proper parameters to avoid HFValidationError
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, trust_remote_code=True)
    
    # Try loading as CausalLM first (for Qwen3-8B), fallback to AutoModel (for Embedding models)
    try:
        model = AutoModelForCausalLM.from_pretrained(abs_model_path, trust_remote_code=True)
    except:
        model = AutoModel.from_pretrained(abs_model_path, trust_remote_code=True)
    
    # Get embedding layer
    embedding_layer = model.get_input_embeddings()
    
    # Process each text
    embeddings = []
    for text in texts:
        # Tokenize the text
        # Note: padding=True ensures attention_mask is returned even for single texts
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = tokens['input_ids'][0]  # Shape: [seq_len]
        # Get attention_mask, default to all ones if not present (shouldn't happen with padding=True)
        attention_mask = tokens.get('attention_mask', torch.ones_like(input_ids))[0]  # Shape: [seq_len]
        
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids)  # Shape: [seq_len, hidden_size]
        
        # Calculate average pooling considering attention mask
        # attention_mask: 1 for valid tokens, 0 for padding tokens
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(masked_embeddings, dim=0)
        count_nonzero = torch.count_nonzero(attention_mask)
        avg_embedding = sum_embeddings / count_nonzero if count_nonzero > 0 else torch.zeros_like(sum_embeddings)
        
        embeddings.append(avg_embedding)
        
        # print(f"Processed text with seq_len={len(input_ids)}, hidden_dim={token_embeddings.shape[1]}")
    
    # Stack all embeddings
    if embeddings:
        return torch.stack(embeddings)
    else:
        # Get hidden size from model config or use default
        try:
            hidden_size = getattr(model.config, 'hidden_size', 4096)
        except:
            hidden_size = 4096  # Default fallback
        return torch.empty(0, hidden_size)  # Return empty tensor with correct hidden dimension


def generate_max_embeddings_for_split(texts: List[str], model_path: str) -> torch.Tensor:
    """Generate max pooled embeddings for a list of texts using Qwen3-8B."""
    # Resolve absolute path
    abs_model_path = os.path.abspath(model_path)
    
    # Load tokenizer and model with proper parameters to avoid HFValidationError
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path, trust_remote_code=True)
    
    # Try loading as CausalLM first (for Qwen3-8B), fallback to AutoModel (for Embedding models)
    try:
        model = AutoModelForCausalLM.from_pretrained(abs_model_path, trust_remote_code=True)
    except:
        model = AutoModel.from_pretrained(abs_model_path, trust_remote_code=True)
    
    # Get embedding layer
    embedding_layer = model.get_input_embeddings()
    
    # Process each text
    embeddings = []
    for text in texts:
        # Tokenize the text
        # Note: padding=True ensures attention_mask is returned even for single texts
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = tokens['input_ids'][0]  # Shape: [seq_len]
        # Get attention_mask, default to all ones if not present (shouldn't happen with padding=True)
        attention_mask = tokens.get('attention_mask', torch.ones_like(input_ids))[0]  # Shape: [seq_len]
        
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids)  # Shape: [seq_len, hidden_size]
        
        # Calculate max pooling considering attention mask
        # attention_mask: 1 for valid tokens, 0 for padding tokens
        # Use masked_fill to set padding positions to -inf so they don't affect max pooling
        masked_embeddings = token_embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), float('-inf'))
        max_embedding = torch.max(masked_embeddings, dim=0)[0]  # Shape: [hidden_size]
        embeddings.append(max_embedding)
        
        # print(f"Processed text with seq_len={len(input_ids)}, hidden_dim={token_embeddings.shape[1]}")
    
    # Stack all embeddings
    if embeddings:
        return torch.stack(embeddings)
    else:
        # Get hidden size from model config or use default
        try:
            hidden_size = getattr(model.config, 'hidden_size', 4096)
        except:
            hidden_size = 4096  # Default fallback
        return torch.empty(0, hidden_size)  # Return empty tensor with correct hidden dimension


def save_embeddings(embeddings_dict: Dict[str, torch.Tensor], output_path: str, field_name: str):
    """Save embeddings in the same format as generate_qwen_embeddings.py."""
    # Modify keys to match target format
    modified_dict = {}
    for key, tensor in embeddings_dict.items():
        new_key = f"{key}_{field_name}"
        modified_dict[new_key] = tensor
    
    abs_output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    
    torch.save(modified_dict, abs_output_path)
    print(f"Embeddings saved to {abs_output_path}")
    
    for key, tensor in modified_dict.items():
        print(f"{key} embeddings shape: {tensor.shape}")


def main():
    args = parse_args()
    
    # Load text data
    print(f"Loading {args.field_name} data...")
    text_data = load_text_data(args.dataset_path, args.field_name)
    
    # Create embedding output directory
    embedding_dir = os.path.join(args.dataset_path, "embedding_qwen")
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Generate average pooled embeddings
    print("Generating average pooled embeddings...")
    avg_embeddings = {}
    for split, texts in text_data.items():
        print(f"Processing {split} split with {len(texts)} entries...")
        avg_embeddings[split] = generate_embeddings_for_split(texts, args.model_path)
    
    # Save average pooled embeddings with correct key format
    avg_output_path = os.path.join(embedding_dir, "embeddings_avg.pt")
    save_embeddings(avg_embeddings, avg_output_path, args.field_name)
    
    # Generate max pooled embeddings
    print("Generating max pooled embeddings...")
    max_embeddings = {}
    for split, texts in text_data.items():
        print(f"Processing {split} split with {len(texts)} entries...")
        max_embeddings[split] = generate_max_embeddings_for_split(texts, args.model_path)
    
    # Save max pooled embeddings with correct key format
    max_output_path = os.path.join(embedding_dir, "embeddings_max.pt")
    save_embeddings(max_embeddings, max_output_path, args.field_name)
    
    print("Done!")


if __name__ == "__main__":
    main()
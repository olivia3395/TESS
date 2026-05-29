#!/usr/bin/env python3
"""Generate last-layer hidden states for Qwen (or compatible) models.

Supports multi-GPU execution via Hugging Face Accelerate. Results are saved as
torch tensors containing token-level hidden states and attention masks per
split (train/vali/test).
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import torch
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_trainer.utils.dataset_registry import DatasetRegistry, DatasetRegistryError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM hidden states for a dataset alias")
    parser.add_argument('--alias', required=True, help='Dataset alias, e.g. FNSPID/ver_camf')
    parser.add_argument('--dataset-version', help='Optional dataset version override')
    parser.add_argument('--model-path', required=True, help='Hugging Face model path or local directory')
    parser.add_argument('--output-path', help='Override output path (defaults to registry llm_hidden path)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per process')
    parser.add_argument('--max-length', type=int, default=1024, help='Maximum token length')
    parser.add_argument('--disable-chat-template', action='store_true', help='Skip chat template formatting')
    parser.add_argument('--enable-thinking', action='store_true', help='Enable thinking tokens in chat template')
    parser.add_argument('--add-generation-prompt', action='store_true', help='Add generation prompt in chat template')
    parser.add_argument('--dtype', default='float16', help='Model dtype (float16, bfloat16, float32, auto)')
    return parser.parse_args()


def resolve_registry(alias: str, dataset_version: Optional[str]) -> Dict:
    overrides = {'version': dataset_version} if dataset_version else None
    try:
        return DatasetRegistry.get(alias, overrides=overrides)
    except DatasetRegistryError as exc:
        raise RuntimeError(f'Failed to resolve dataset alias {alias}: {exc}') from exc


def load_split_records(dataset_root: str, relative_path: str) -> List[Dict]:
    path = os.path.abspath(os.path.join(dataset_root, relative_path.lstrip('/')))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Split file not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_texts(records: Sequence[Dict]) -> List[str]:
    texts = []
    for item in records:
        if isinstance(item, dict):
            texts.append(item.get('news') or item.get('prompt') or '')
        else:
            texts.append(str(item))
    return texts


def build_inputs(tokenizer, texts: Sequence[str], use_chat_template: bool, enable_thinking: bool, add_generation_prompt: bool) -> List[str]:
    if not use_chat_template:
        return list(texts)
    formatted = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except AttributeError:
            formatted_text = text
        formatted.append(formatted_text)
    return formatted


def load_model(model_path: str, dtype: str, device: torch.device):
    dtype_map = {
        'auto': None,
        'float32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower(), None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def encode_hidden(
    tokenizer,
    model,
    texts: Sequence[str],
    start_index: int,
    batch_size: int,
    accelerator: Accelerator,
    max_length: int,
    use_chat_template: bool,
    enable_thinking: bool,
    add_generation_prompt: bool,
    hidden_dim: int,
) -> Dict[str, torch.Tensor]:
    inputs = build_inputs(tokenizer, texts, use_chat_template, enable_thinking, add_generation_prompt)
    hidden_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    iterator = range(0, len(inputs), batch_size)
    device = accelerator.device

    progress = tqdm(
        iterator,
        desc=f'Encoding (rank {accelerator.local_process_index})',
        disable=not accelerator.is_local_main_process,
    )
    for offset in progress:
        batch_texts = inputs[offset:offset + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens.get('attention_mask', torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_last = outputs.hidden_states[-1].detach().cpu().float()
        mask_cpu = attention_mask.detach().cpu().bool()

        seq_len = hidden_last.shape[1]
        if seq_len < max_length:
            pad_tokens = max_length - seq_len
            hidden_last = F.pad(hidden_last, (0, 0, 0, pad_tokens), value=0.0)
            mask_cpu = F.pad(mask_cpu, (0, pad_tokens), value=False)

        hidden_chunks.append(hidden_last)
        mask_chunks.append(mask_cpu)

    if hidden_chunks:
        local_hidden = torch.cat(hidden_chunks, dim=0)
        local_mask = torch.cat(mask_chunks, dim=0)
    else:
        local_hidden = torch.empty((0, max_length, hidden_dim), dtype=torch.float32)
        local_mask = torch.empty((0, max_length), dtype=torch.bool)

    return {
        'start': start_index,
        'count': local_hidden.shape[0],
        'hidden': local_hidden,
        'mask': local_mask,
    }


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    registry = resolve_registry(args.alias, args.dataset_version)
    dataset_root = registry.get('root')
    if not dataset_root:
        raise RuntimeError('Dataset registry entry missing root path')

    embeddings = registry.get('embeddings', {})
    hidden_spec = embeddings.get('llm_hidden')
    if not hidden_spec and not args.output_path:
        raise RuntimeError('Dataset registry does not define llm_hidden path; specify --output-path')

    output_rel = args.output_path or hidden_spec.get('path')
    if not output_rel:
        raise RuntimeError('Missing output path for hidden states')
    output_abs = os.path.abspath(os.path.join(dataset_root, output_rel.lstrip('/')))
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        pad_token = tokenizer.eos_token or tokenizer.unk_token or '<pad>'
        tokenizer.add_special_tokens({'pad_token': pad_token})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    device = accelerator.device
    base_model = load_model(args.model_path, args.dtype, device)
    model = accelerator.prepare(base_model)
    hidden_dim = accelerator.unwrap_model(model).config.hidden_size

    splits = registry.get('splits', {})
    default_hidden_keys = {'train': 'train_hidden', 'vali': 'vali_hidden', 'test': 'test_hidden'}
    default_mask_keys = {'train': 'train_mask', 'vali': 'vali_mask', 'test': 'test_mask'}
    output_dict: Dict[str, torch.Tensor] = {}

    for split_name, rel_path in splits.items():
        records = load_split_records(dataset_root, rel_path)
        texts = extract_texts(records)
        if not texts:
            continue

        total = len(texts)
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        per_proc = math.ceil(total / world_size)
        start = rank * per_proc
        end = min(start + per_proc, total)
        local_texts = texts[start:end]

        if accelerator.is_main_process:
            print(f'Encoding split {split_name}: {total} samples with {world_size} process(es)')

        local_result = encode_hidden(
            tokenizer,
            model,
            local_texts,
            start,
            args.batch_size,
            accelerator,
            args.max_length,
            not args.disable_chat_template,
            args.enable_thinking,
            args.add_generation_prompt,
            hidden_dim,
        )

        temp_path = None
        if local_result['count'] > 0:
            temp_path = f"{output_abs}.rank{rank}.tmp"
            torch.save(local_result, temp_path)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            parts = []
            for proc_rank in range(world_size):
                part_path = f"{output_abs}.rank{proc_rank}.tmp"
                if os.path.isfile(part_path):
                    parts.append(torch.load(part_path, map_location='cpu'))
                    os.remove(part_path)

            if not parts:
                continue

            total = sum(p['count'] for p in parts)
            seq_len = max((p['hidden'].shape[1] for p in parts if p['count'] > 0), default=args.max_length)
            final_hidden = torch.zeros((total, seq_len, hidden_dim), dtype=torch.float32)
            final_mask = torch.zeros((total, seq_len), dtype=torch.bool)

            for part in parts:
                count = part['count']
                if count == 0:
                    continue
                start_idx = part['start']
                final_hidden[start_idx:start_idx + count] = part['hidden']
                final_mask[start_idx:start_idx + count] = part['mask']

            output_dict[default_hidden_keys.get(split_name, f'{split_name}_hidden')] = final_hidden
            output_dict[default_mask_keys.get(split_name, f'{split_name}_mask')] = final_mask

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        torch.save(output_dict, output_abs)
        print(f'Saved hidden states to {output_abs}')

    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()

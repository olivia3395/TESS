#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based Information Certainty classification with retry + multi-GPU support.
Only the prompt/regex are customized; output仅新增字段 info_certainty_code (High=3, Medium=2, Low=1)。
"""

import argparse
import json
import os
import socket
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


PROMPT_TEMPLATE = """You are a financial analyst. Your task is to analyze the certainty of the information presented in the following news text and classify it into one of three categories based on the level of confidence conveyed:

High Confidence: The information in the text is presented with strong certainty, using definitive or conclusive language. Examples include terms like “confirmed,” “certain,” “definite,” etc.

Medium Confidence: The information in the text expresses a moderate level of certainty, but also includes some hedging or uncertainty. Examples include terms like “likely,” “expected,” “may,” etc.

Low Confidence: The information in the text is presented with ambiguity, lacking clear certainty. This can include terms like “possibly,” “uncertain,” “might,” “potentially,” etc.

Instructions:

Analyze the following news text {news} and identify how certain the information presented is.

Classify the text based on the level of confidence it expresses using the categories provided.

Output:
You MUST output in the following EXACT format with no extra text:

Information Certainty: <High Confidence / Medium Confidence / Low Confidence>

--------------------------------------------------------------------------------------

Historical Data:{historical_data}
Company: {company_name}
News: {news}
"""

LABEL_TO_CODE = {
    "High Confidence": 3,
    "Medium Confidence": 2,
    "Low Confidence": 1,
}
LABEL_PATTERN = re.compile(r"Information Certainty:\s*(High Confidence|Medium Confidence|Low Confidence)", re.IGNORECASE)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def find_free_port() -> int:
    # Fallback to static port if socket bind is not permitted
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            return s.getsockname()[1]
    except Exception:
        return 29507


def parse_args():
    parser = argparse.ArgumentParser(description="Generate info certainty codes (1-3) with retry and multi-GPU")
    parser.add_argument('--data-path', type=str, default='dataset/FNSPID/ver_camf', help='Input dataset dir')
    parser.add_argument('--output-path', type=str, default='dataset/FNSPID/ver_camf_info_certainty_llmcode', help='Output dir')
    parser.add_argument('--model-path', type=str, default='pretrain_model/EmbeddingModel/Qwen3-Embedding-8B', help='HF model path')
    parser.add_argument('--splits', nargs='+', default=['train'], help='Splits to process')
    parser.add_argument('--batch-size', type=int, default=4, help='Generation batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id (single GPU mode)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for DDP')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Max new tokens')
    parser.add_argument('--retries', type=int, default=5, help='Retries when parsing fails')
    parser.add_argument('--test-limit', type=int, default=None, help='Optional cap for debugging')
    parser.add_argument('--master-port', type=int, default=None, help='DDP master port override')
    return parser.parse_args()


def _parse_gpu_list(gpu_string: str) -> List[int]:
    entries = [token.strip() for token in gpu_string.split(',') if token.strip()]
    if not entries:
        raise ValueError('Invalid --gpus argument: no GPU ids found')
    return [int(e) for e in entries]


def load_model_and_tokenizer(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(news: str, company_name: str = "Unknown") -> str:
    return PROMPT_TEMPLATE.format(news=news, company_name=company_name)


def classify_single(model, tokenizer, prompt: str, device: torch.device, args) -> int:
    for _ in range(args.retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        m = LABEL_PATTERN.search(text)
        if m:
            label = m.group(1).title()
            label = {
                "High confidence": "High Confidence",
                "Medium confidence": "Medium Confidence",
                "Low confidence": "Low Confidence",
                "High Confidence": "High Confidence",
                "Medium Confidence": "Medium Confidence",
                "Low Confidence": "Low Confidence",
            }.get(label, label)
            if label in LABEL_TO_CODE:
                return LABEL_TO_CODE[label]
    # Parsing failed after retries; emit code 0 as unknown
    return 0


def _process_range(data_slice, args, device: torch.device):
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    results = []
    for item in tqdm(data_slice, desc=f"rank{device}"):
        news = str(item.get("news", ""))
        prompt = build_prompt(news=news, company_name="Unknown")
        code = classify_single(model, tokenizer, prompt, device, args)
        new_item = dict(item)
        new_item["info_certainty_code"] = code
        results.append(new_item)
    return results


def _split_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
    base = total // parts
    rem = total % parts
    ranges = []
    start = 0
    for i in range(parts):
        extra = 1 if i < rem else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


def _ddp_worker(local_rank: int, base_args: dict, split: str):
    gpu_ids = base_args.get("gpu_ids", [])
    world_size = len(gpu_ids)
    torch.cuda.set_device(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(base_args.get("master_port", 29507))
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    args = argparse.Namespace(**base_args)
    args.device = f"cuda:{local_rank}"

    in_file = Path(args.data_path) / f"{split}.json"
    data = json.load(in_file.open())
    if args.test_limit:
        data = data[: args.test_limit]

    ranges = _split_ranges(len(data), world_size)
    start, end = ranges[local_rank]
    partial = data[start:end]
    print(f"[GPU {local_rank}] split={split} range={start}-{end}")
    processed = _process_range(partial, args, torch.device(args.device))

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (local_rank, processed))

    if local_rank == 0:
        gathered.sort(key=lambda x: x[0])
        merged = []
        for _, part in gathered:
            merged.extend(part)
        out_dir = Path(args.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.json"
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        print(f"{split} saved to {out_path} ({len(merged)} samples)")

    dist.destroy_process_group()


def process_split_ddp(split: str, args):
    gpu_ids = _parse_gpu_list(args.gpus)
    master_port = args.master_port or find_free_port()
    base_args = vars(args).copy()
    base_args["gpu_ids"] = gpu_ids
    base_args["master_port"] = master_port
    mp.set_start_method("spawn", force=True)
    mp.spawn(_ddp_worker, args=(base_args, split), nprocs=len(gpu_ids), join=True)


def process_split_single(split: str, args):
    device = torch.device(args.device or "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    in_file = Path(args.data_path) / f"{split}.json"
    data = json.load(in_file.open())
    if args.test_limit:
        data = data[: args.test_limit]

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.json"

    enriched = _process_range(data, args, device)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
    print(f"{split} saved to {out_path} ({len(enriched)} samples)")


def main():
    args = parse_args()
    if args.gpus:
        for split in args.splits:
            process_split_ddp(split, args)
    else:
        args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        for split in args.splits:
            process_split_single(split, args)


if __name__ == "__main__":
    main()

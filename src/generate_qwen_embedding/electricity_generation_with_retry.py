#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Electricity load generation script with validation and retry.
Adapts the improved FNSPID logic to the Electricity dataset (48 half-hour steps).
"""

import argparse
import gc
import json
import os
import random
import re
import socket
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch.distributed as dist
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm




def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def load_electricity_data(data_path: str, split: str) -> list:
    file_path = os.path.join(data_path, f"{split}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _parse_gpu_list(gpu_string: str) -> List[int]:
    entries = [token.strip() for token in gpu_string.split(",") if token.strip()]
    if not entries:
        raise ValueError("Invalid --gpus argument: no GPU ids found")
    try:
        return [int(entry) for entry in entries]
    except ValueError as exc:
        raise ValueError(f"Invalid GPU id in --gpus: {gpu_string}") from exc


def _split_dataset_ranges(total_size: int, num_gpus: int) -> list:
    base = total_size // num_gpus
    remainder = total_size % num_gpus
    ranges = []
    start = 0
    for i in range(num_gpus):
        extra = 1 if i < remainder else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


def _distributed_embedding_worker(local_rank: int, base_args: dict, split_name: str):
    torch.cuda.set_device(local_rank)
    world_size = len(base_args.get("gpu_ids", [0]))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(base_args.get("master_port", "29507"))

    import datetime
    timeout = datetime.timedelta(hours=4)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size, timeout=timeout)
    # dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    args = argparse.Namespace(**base_args)
    args.gpu_id = local_rank

    original_data = load_electricity_data(args.data_path, split_name)
    if args.test_limit is not None:
        original_data = original_data[: args.test_limit]

    data_ranges = _split_dataset_ranges(len(original_data), world_size)
    start_idx, end_idx = data_ranges[local_rank]
    partial_data = original_data[start_idx:end_idx]

    print(
        f"GPU {local_rank}: Processing {split_name} split, "
        f"indices {start_idx}-{end_idx} ({len(partial_data)} samples)"
    )

    result = _process_data_on_gpu(args, partial_data, split_name, local_rank)

    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, (local_rank, result))

    if local_rank == 0:
        all_results.sort(key=lambda x: x[0])
        merged_result = _merge_results([r[1] for r in all_results])
        _save_merged_result(merged_result, args.output_path, split_name)

    dist.destroy_process_group()


STEP_LINE_PATTERN = re.compile(
    r"(?:\d+\s*:)?\s*Step\s*(\d*)\s*:.*?(?:[sS]trength)[:\s]*([A-Za-z]+).*?(?:[tT]rend)[:\s]*([A-Za-z]+)",
    re.IGNORECASE,
)

GLOBAL_TREND_PATTERN = re.compile(
    r"Global\s*trend\s*[:\s]*\s*(Rising|Falling)", re.IGNORECASE
)

STRENGTH_ENCODING = {
    "emerging": 1,
    "moderate": 2,
    "significant": 3,
    "prominent": 4,
    "dominant": 5,
}

TREND_ENCODING = {
    "rising": 1,
    "falling": -1,
}


def extract_structured_trend_fields(news_text: str, num_steps: int) -> Tuple[List[int], List[int], int]:
    step1_pattern = re.compile(r"(Step\s*1[:\s]|Step1[:\s])", re.IGNORECASE)
    step1_match = step1_pattern.search(news_text)
    if step1_match:
        news_text = news_text[step1_match.start() :]

    strengths = [0] * num_steps
    trends = [0] * num_steps

    matches = []
    for match in STEP_LINE_PATTERN.finditer(news_text):
        step_num_str = match.group(1).strip()
        raw_strength = match.group(2).strip().lower()
        raw_trend = match.group(3).strip().lower()
        step_idx = int(step_num_str) - 1 if step_num_str else 0
        matches.append((step_idx, raw_strength, raw_trend))

    if not matches:
        loose_pattern = re.compile(
            r"(?:\d+\s*[:\.]\s*)?[Aa]nalysis:.*?[sS]trength[:\s]*([A-Za-z]+).*?[tT]rend[:\s]*([A-Za-z]+)",
            re.IGNORECASE,
        )
        for i, match in enumerate(loose_pattern.finditer(news_text)):
            raw_strength = match.group(1).strip().lower()
            raw_trend = match.group(2).strip().lower()
            step_idx = i if i < num_steps else num_steps - 1
            matches.append((step_idx, raw_strength, raw_trend))

    if not matches:
        lines = news_text.split("\n")
        step_count = 0
        i = 0
        while i < len(lines) and step_count < num_steps:
            line = lines[i].strip()
            if ("analysis" in line.lower() or re.search(r"Step\d*:", line, re.IGNORECASE)) and ":" in line:
                strength = None
                trend = None
                for j in range(1, 4):
                    if i + j < len(lines):
                        next_line = lines[i + j].strip()
                        if "strength:" in next_line.lower() and not strength:
                            strength_match = re.search(
                                r"[sS]trength[:\s]*([A-Za-z]+)", next_line, re.IGNORECASE
                            )
                            if strength_match:
                                strength = strength_match.group(1).strip().lower()
                        elif "trend:" in next_line.lower() and not trend:
                            trend_match = re.search(
                                r"[tT]rend[:\s]*([A-Za-z]+)", next_line, re.IGNORECASE
                            )
                            if trend_match:
                                trend = trend_match.group(1).strip().lower()
                if strength and trend:
                    step_idx = step_count
                    matches.append((step_idx, strength, trend))
                    step_count += 1
            i += 1

    for step_idx, raw_strength, raw_trend in matches:
        if 0 <= step_idx < num_steps:
            strengths[step_idx] = STRENGTH_ENCODING.get(raw_strength, 0)
            trends[step_idx] = TREND_ENCODING.get(raw_trend, -1)

    global_match = GLOBAL_TREND_PATTERN.search(news_text)
    if global_match:
        global_trend = TREND_ENCODING.get(global_match.group(1).strip().lower(), -1)
    else:
        global_trend = 0

    return strengths, trends, global_trend


def is_extraction_successful(strengths: List[int], trends: List[int]) -> bool:
    for strength, trend in zip(strengths, trends):
        if strength == 0 or trend == 0:
            return False
    return True


def generate_text_and_extract_hidden_states_with_retry(
    model,
    tokenizer,
    prompts,
    batch_size=4,
    max_new_tokens=1024,
    device="cuda",
    original_data=None,
    num_steps: int = 48,
):
    generated_texts = []
    all_hidden_states = []
    all_attention_masks = []
    
    # 添加token统计变量
    total_input_tokens = 0
    total_output_tokens = 0

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating texts"):
            batch_prompts = prompts[i : i + batch_size]
            batch_original_data = original_data[i : i + batch_size] if original_data else [None] * len(batch_prompts)

            for prompt_idx, (prompt, orig_item) in enumerate(zip(batch_prompts, batch_original_data)):
                generated_text = None
                retry_count = 0
                max_retries = 5

                while retry_count < max_retries:
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    model_inputs = tokenizer(
                        [text], return_tensors="pt", padding=True, truncation=True, max_length=1024
                    )
                    input_ids = model_inputs["input_ids"].to(device)
                    attention_mask = model_inputs["attention_mask"].to(device)
                    
                    # 统计输入token数量
                    input_token_count = input_ids.shape[1]
                    total_input_tokens += input_token_count

                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                    output_ids = generated_ids[0][len(input_ids[0]) :]
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    # 统计输出token数量
                    output_token_count = len(output_ids)
                    total_output_tokens += output_token_count
                    
                    # 移除每个样本的打印，只保留最终统计

                    if "<think>" in generated_text and "</think>" in generated_text:
                        after_think_end = generated_text.split("</think>", 1)
                        if len(after_think_end) > 1:
                            generated_text = after_think_end[1].strip()
                        else:
                            before_think = generated_text.split("<think>", 1)[0]
                            generated_text = before_think.strip()
                    elif "<think>" in generated_text:
                        before_think = generated_text.split("<think>", 1)[0]
                        generated_text = before_think.strip()

                    step1_match = re.search(r"(?:\d+\s*:)?\s*Step\s*1\s*:", generated_text, re.IGNORECASE)
                    if step1_match:
                        generated_text = generated_text[step1_match.start() :]

                    strengths, trends, global_trend = extract_structured_trend_fields(
                        generated_text, num_steps=num_steps
                    )

                    if is_extraction_successful(strengths, trends):
                        break

                    retry_count += 1
                    time.sleep(random.uniform(0.1, 0.5))

                if generated_text is None or retry_count >= max_retries:
                    print(f"Failed to generate valid text for sample {i+prompt_idx} after {max_retries} retries")
                    if generated_text is None:
                        generated_text = ""

                generated_texts.append(generated_text.strip())
        
        # 打印总体token使用统计
        print(f"\nTotal token usage statistics:")
        print(f"  Total input tokens: {total_input_tokens}")
        print(f"  Total output tokens: {total_output_tokens}")
        print(f"  Total tokens: {total_input_tokens + total_output_tokens}")
        print(f"  Average output tokens per sample: {total_output_tokens / len(prompts) if prompts else 0:.2f}")
        print(f"  Max new tokens setting: {max_new_tokens}")

    return generated_texts, None, None


def _merge_results(results):
    merged_data = []
    all_hidden_states = []
    all_attention_masks = []

    for result in results:
        merged_data.extend(result["data"])
        if result["hidden_states"] is not None:
            all_hidden_states.append(result["hidden_states"])
        if result["attention_masks"] is not None:
            all_attention_masks.append(result["attention_masks"])

    merged_hidden_states = torch.cat(all_hidden_states, dim=0) if all_hidden_states else None
    merged_attention_masks = torch.cat(all_attention_masks, dim=0) if all_attention_masks else None

    return {
        "data": merged_data,
        "hidden_states": merged_hidden_states,
        "attention_masks": merged_attention_masks,
    }


def _process_data_on_gpu(args, data, split_name: str, gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"GPU {gpu_id}: Using device: {device}")

    print(f"GPU {gpu_id}: Loading Qwen3-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()

    prompts = []
    for item in data:
        historical_data = item.get("historical_data", "")
        news = item.get("news") or item.get("prompt", "")
        few_shots = args.few_shots
        values = [float(x.strip()) for x in historical_data.split(",") if x.strip()]
        mean_value = sum(values) / len(values) if values else 0.0
        prompt = args.prompt_template.format(
            historical_data=historical_data,
            news=news,
            few_shots=few_shots,
            mean_value=mean_value,
        )
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts and extracting embeddings for {split_name}...")
    generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states_with_retry(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        args.max_new_tokens,
        device,
        data,
        num_steps=args.num_steps,
    )

    new_data = []
    for original_item, generated_text in zip(data, generated_texts):
        new_item = deepcopy(original_item)
        new_item["news"] = generated_text
        strengths, trends, global_trend = extract_structured_trend_fields(generated_text, num_steps=args.num_steps)
        new_item["step_strengths"] = strengths
        new_item["step_trends"] = trends
        new_item["global_trend"] = global_trend
        new_data.append(new_item)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "data": new_data,
        "hidden_states": hidden_states,
        "attention_masks": attention_masks,
    }


def _save_merged_result(result, output_path: str, split_name: str):
    output_file = os.path.join(output_path, f"{split_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["data"], f, ensure_ascii=False, indent=2)
    print(f"Saved generated {split_name} data to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qwen texts for Electricity dataset with retry and extraction over 48 steps"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_camf",
        help="Path to Electricity dataset split folder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_generated_withfewshots",
        help="Output path for generated texts",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/ssd/hf_home/models/Qwen3-8B",
        help="Path to Qwen3-8B model",
    )
    parser.add_argument("--splits", nargs="+", default=["train","vali","test"], help="Dataset splits to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation and encoding")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--num-steps", type=int, default=48, help="Number of step outputs expected from the model")

    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are a professional electricity demand analyst. Your task is to analyze electricity load movements and related contextual news to determine trend direction and strength for each of the 48 time steps, then conclude the global trend.\n"
            "\n"
            "Instructions:\n"
            "1. The field historical_data contains exactly 48 comma-separated past electricity load values for a single region or market (e.g., NSW). Index these values in order as Step1 (the earliest in this window) through Step48 (the latest).\n"
            "2. For each of the 48 steps (Step1–Step48), you will analyze that step's load value from the sequence {{{historical_data}}} compared to the historical mean value {mean_value}.\n"
            "3. Determine the trend direction strictly by the numeric rule: Rising if the step load value > {mean_value}, Falling if the step load value < {mean_value}.\n"
            "4. Determine the trend strength based on the magnitude of the difference between the step load value and {mean_value}, choosing ONLY from: Emerging, Moderate, Significant, Prominent, Dominant.\n"
            "5. Determine the Global trend by comparing the predicted future mean electricity load with the historical mean value {mean_value}. If the predicted future mean is greater than {mean_value}, the Global trend is Rising; if it is lower, the Global trend is Falling.\n"
            "6. Use the information in News (e.g., weather, economic activity, policy changes, and other meta-domain factors) only to support your brief analyses in natural language. However, the labels trend and strength for each step must always follow the numeric rules above and must not be changed based on your own assumptions.\n"
            "\n"
            "You MUST output in the following EXACT format with no extra text:\n"
            "Step1: Analysis:<brief analysis for step 1>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "Step2: Analysis:<brief analysis for step 2>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "...\n"
            "Step48: Analysis:<brief analysis for step 48>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "Global trend:<Rising or Falling>\n"
            "\n"
            "\n"
            "Historical data: {{{historical_data}}}\n"
            "Mean value: {mean_value}\n"
            "News: {news}\n"
            "Few_shots: {few_shots}\n"
            "In the few_shots examples, the field trends uses -1 to represent Falling and 1 to represent Rising, and the field step_strengths uses integers 1-5 to represent, from weakest to strongest, the five strength labels: Emerging (1), Moderate (2), Significant (3), Prominent (4), Dominant (5).\n"
            "Provide your analysis in the exact format specified above:"
        ),
        help="Prompt template for electricity demand generation over 48 steps",
    )

    parser.add_argument(
        "--test-limit", type=int, default=None, help="Limit number of samples for quick tests (default: None, all)"
    )
    parser.add_argument("--gpus", type=str, help="Comma separated GPU ids for parallel processing")
    parser.add_argument("--few-shots", type=str, default="src/generate_qwen_embedding/few_shot_samples_elec.txt", help="few_shots")

    args = parser.parse_args()
    args.few_shots = open(args.few_shots, "r", encoding="utf-8").read()

    if args.gpus:
        gpu_ids = _parse_gpu_list(args.gpus)
    else:
        gpu_ids = [args.gpu_id]

    os.makedirs(args.output_path, exist_ok=True)

    if len(gpu_ids) > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        master_port = find_free_port()
        os.environ.setdefault("MASTER_PORT", str(master_port))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)

        base_args = vars(args)
        base_args["gpu_ids"] = gpu_ids
        base_args["master_port"] = master_port

        for split_name in args.splits:
            print(f"Processing {split_name} split with {len(gpu_ids)} GPUs...")
            mp.spawn(
                _distributed_embedding_worker,
                nprocs=len(gpu_ids),
                args=(base_args, split_name),
                join=True,
            )
    else:
        print("Using single GPU/CPU processing...")
        for split_name in args.splits:
            print(f"Processing {split_name} split...")
            dataset = load_electricity_data(args.data_path, split_name)
            if args.test_limit is not None:
                dataset = dataset[: args.test_limit]
            result = _process_data_on_gpu(args, dataset, split_name, args.gpu_id)
            _save_merged_result(result, args.output_path, split_name)


if __name__ == "__main__":
    main()

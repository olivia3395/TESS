#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate temporal influence shape labels with retry and code extraction."""

import argparse
import json
import os
import random
import re
import socket
import time
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_free_port():
    """Return an available port for distributed init."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def load_fnspid_data(data_path, split):
    """Load FNSPID split."""
    file_path = os.path.join(data_path, f"{split}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _parse_gpu_list(gpu_string: str):
    """Parse comma-separated GPU ids."""
    entries = [token.strip() for token in gpu_string.split(",") if token.strip()]
    if not entries:
        raise ValueError("Invalid --gpus argument: no GPU ids found")
    try:
        return [int(entry) for entry in entries]
    except ValueError as exc:
        raise ValueError(f"Invalid GPU id in --gpus: {gpu_string}") from exc


def _split_dataset_ranges(total_size: int, num_gpus: int) -> list:
    """Evenly split dataset indices across GPUs."""
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
    """Distributed worker that processes part of the split on one GPU."""
    torch.cuda.set_device(local_rank)

    world_size = len(base_args.get("gpu_ids", [0]))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(base_args.get("master_port", "29507"))
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)

    args = argparse.Namespace(**base_args)
    args.gpu_id = local_rank

    original_data = load_fnspid_data(args.data_path, split_name)
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


def _process_data_on_gpu(args, data, split_name, gpu_id):
    """Process a shard of data on the given GPU."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"GPU {gpu_id}: Using device: {device}")

    print(f"GPU {gpu_id}: Loading Qwen3-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()

    prompts: List[str] = []
    for item in data:
        historical_data = item.get("historical_data", "")
        news = item.get("news", "")
        mean_value = _compute_mean_value(historical_data)
        prompt = args.prompt_template.format(
            historical_data=historical_data,
            mean_value=mean_value,
            news=news,
        )
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts for {split_name}...")
    generated_texts, shape_codes = generate_text_and_extract_shape_with_retry(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        args.max_new_tokens,
        device,
    )

    new_data = []
    for original_item, generated_text, shape_code in zip(data, generated_texts, shape_codes):
        new_item = original_item.copy()
        new_item["news"] = generated_text
        new_item["code"] = shape_code
        new_data.append(new_item)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "data": new_data,
        "hidden_states": None,
        "attention_masks": None,
    }


def _merge_results(results):
    """Merge results from different GPUs."""
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
    merged_attention_masks = (
        torch.cat(all_attention_masks, dim=0) if all_attention_masks else None
    )

    return {
        "data": merged_data,
        "hidden_states": merged_hidden_states,
        "attention_masks": merged_attention_masks,
    }


SHAPE_PATTERN = re.compile(
    r"Temporal\s*Influence\s*Shape\s*:\s*(Immediate|Delayed|Sustained)",
    re.IGNORECASE,
)

SHAPE_ENCODING = {
    "immediate": 1,
    "delayed": 2,
    "sustained": 3,
}


def extract_shape_label(text: str):
    """Parse generated text and return temporal influence label and encoded code."""
    match = SHAPE_PATTERN.search(text)
    if not match:
        return "", 0

    normalized = match.group(1).strip().lower()
    code = SHAPE_ENCODING.get(normalized, 0)
    label = normalized.title()
    return label, code


def is_extraction_successful(code: int) -> bool:
    """Return True if shape code is valid."""
    return code in SHAPE_ENCODING.values()


def _strip_think_tags(generated_text: str) -> str:
    """Remove think tags from model output if present."""
    if "<think>" in generated_text and "</think>" in generated_text:
        after_think_end = generated_text.split("</think>", 1)
        if len(after_think_end) > 1:
            return after_think_end[1].strip()
        return generated_text.split("<think>", 1)[0].strip()
    if "<think>" in generated_text:
        return generated_text.split("<think>", 1)[0].strip()
    return generated_text


def generate_text_and_extract_shape_with_retry(
    model,
    tokenizer,
    prompts,
    batch_size=4,
    max_new_tokens=512,
    device="cuda",
):
    """Generate texts with the temporal influence prompt and retry on parsing failure."""
    generated_texts = []
    shape_codes = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating texts"):
            batch_prompts = prompts[i : i + batch_size]

            for prompt_idx, prompt in enumerate(batch_prompts):
                generated_text = ""
                shape_code = 0
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
                        [text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    )
                    input_ids = model_inputs["input_ids"].to(device)
                    attention_mask = model_inputs["attention_mask"].to(device)

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
                    generated_text = _strip_think_tags(generated_text)

                    _, shape_code = extract_shape_label(generated_text)
                    if is_extraction_successful(shape_code):
                        break

                    retry_count += 1
                    time.sleep(random.uniform(0.1, 0.5))

                if not is_extraction_successful(shape_code):
                    print(
                        f"Failed to extract temporal influence shape for sample {i + prompt_idx} "
                        f"after {max_retries} retries"
                    )

                generated_texts.append(generated_text.strip())
                shape_codes.append(shape_code)

    return generated_texts, shape_codes


def _compute_mean_value(historical_data: str) -> float:
    """Compute mean from the comma-separated historical data string."""
    try:
        values = [float(x.strip()) for x in historical_data.split(",") if x.strip()]
    except ValueError:
        return 0.0
    return sum(values) / len(values) if values else 0.0


def _save_merged_result(result, output_path, split_name):
    """Save merged dataset to disk."""
    output_file = os.path.join(output_path, f"{split_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["data"], f, ensure_ascii=False, indent=2)
    print(f"Saved generated {split_name} data to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate temporal influence shape texts with retry for FNSPID dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_camf",
        help="Path to FNSPID dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_temporal_shape_shape",
        help="Output path for generated texts",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/ssd/hf_home/models/Qwen3-8B",
        help="Path to Qwen3-8B model",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test","vali","train"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are a professional financial analyst. Your task is to analyze stock price movements and related news "
            "to determine the temporal influence shape of the event described in the news.\n\n"
            "Instructions:\n\n"
            "You are given historical stock prices as {{{historical_data}}} with a historical mean value {mean_value}, "
            "and news text {news}.\n\n"
            "For each of the 5 steps (Step1–Step5), analyze the stock price movement to understand how quickly and "
            "persistently the stock price responds after the news event.\n\n"
            "You must classify the event impact into one of three categories:\n\n"
            "Immediate — A clear and significant reaction appears within the first 1–2 steps.\n\n"
            "Use this label when a noticeable change in price occurs in Step1 or Step2, and the price shows a strong "
            "deviation from the mean.\n\n"
            "Delayed — Little or no response in Step1 and Step2 but a clear reaction starting at Step3 or later.\n\n"
            "Use this label when the price shows little movement in the first two steps, but a clear reaction happens in "
            "Step3 or later, after the event has some time to influence the price.\n\n"
            "Sustained — A consistent significant reaction persists across most or all steps.\n\n"
            "Use this label when the price shows consistent significant deviations from the mean over multiple steps "
            "(i.e., price deviates significantly in Step1, Step2, and Step3 or Step3, Step4, and Step5). The key here "
            "is persistence: the price continues to show significant movement over at least three consecutive steps or "
            "more.\n\n"
            "This label should apply when the price moves significantly across several steps, not just in a single "
            "step.\n\n"
            "A clear/significant reaction is interpreted as a noticeable deviation relative to the mean value, enough "
            "to suggest that the event is impacting price behavior.\n\n"
            "Output Format:\n\n"
            "Step1: Analysis: <brief analysis for step 1>\n"
            "Step2: Analysis: <brief analysis for step 2>\n"
            "Step3: Analysis: <brief analysis for step 3>\n"
            "Step4: Analysis: <brief analysis for step 4>\n"
            "Step5: Analysis: <brief analysis for step 5>\n\n"
            "Temporal Influence Shape: <Immediate or Delayed or Sustained>"
        ),
        help="Temporal influence shape prompt template",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit number of samples for quick testing",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma separated GPU ids for parallel processing",
    )
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = _parse_gpu_list(args.gpus)
    else:
        gpu_ids = [args.gpu_id]

    if len(gpu_ids) > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        master_port = find_free_port()
        os.environ.setdefault("MASTER_PORT", str(master_port))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)

        base_args = vars(args)
        base_args["gpu_ids"] = gpu_ids
        base_args["master_port"] = master_port

        os.makedirs(args.output_path, exist_ok=True)

        for split_name in args.splits:
            print(f"Processing {split_name} split with {len(gpu_ids)} GPUs...")
            mp.spawn(
                _distributed_embedding_worker,
                nprocs=len(gpu_ids),
                args=(base_args, split_name),
                join=True,
            )
    else:
        os.makedirs(args.output_path, exist_ok=True)
        for split_name in args.splits:
            print(f"Processing {split_name} split on single GPU/CPU...")
            split_data = load_fnspid_data(args.data_path, split_name)
            if args.test_limit is not None:
                split_data = split_data[: args.test_limit]
            result = _process_data_on_gpu(args, split_data, split_name, args.gpu_id)
            _save_merged_result(result, args.output_path, split_name)


if __name__ == "__main__":
    main()

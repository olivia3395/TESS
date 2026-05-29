#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate information-certainty labels with retry parsing."""

import argparse
import json
import os
import random
import re
import time
from collections import Counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import socket

try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception:  # pragma: no cover - spaCy optional fallback
    nlp = None


CERTAINTY_PATTERN = re.compile(
    r"Information\s*Certainty\s*:\s*(High Confidence|Medium Confidence|Low Confidence)",
    re.IGNORECASE,
)

CERTAINTY_ENCODING = {
    "low confidence": 1,
    "medium confidence": 2,
    "high confidence": 3,
}


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


def _compute_mean_value(historical_data: str) -> float:
    """Compute mean from the comma-separated historical data string."""
    try:
        values = [float(x.strip()) for x in historical_data.split(",") if x.strip()]
    except ValueError:
        return 0.0
    return sum(values) / len(values) if values else 0.0


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


def extract_certainty_field(text: str):
    """Parse generated text and return certainty label and code."""
    match = CERTAINTY_PATTERN.search(text)
    if match:
        label = match.group(1).strip()
        code = CERTAINTY_ENCODING.get(label.lower(), 0)
        return label.title(), code
    return "", 0


def is_extraction_successful(certainty_code: int) -> bool:
    """Return True if certainty code is valid."""
    return certainty_code in CERTAINTY_ENCODING.values()


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


def extract_company_names(news_text: str):
    """Extract company names using spaCy with a fallback regex."""
    if nlp is None:
        return "Unknown Company"

    doc = nlp(news_text)
    companies = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            companies.append(ent.text)

    if not companies:
        company_patterns = [
            r"\b[A-Z][a-zA-Z]+\s+(?:Inc|Corp|Corporation|Ltd|Limited|Company|Co|Group|Technologies|Tech)\b",
            r"\b(?:Apple|Google|Microsoft|Amazon|Facebook|Tesla|Netflix|Intel|IBM|Oracle)\b",
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, news_text)
            companies.extend(matches)

    if companies:
        most_common = Counter(companies).most_common(1)
        return most_common[0][0] if most_common else companies[0]

    return "Unknown Company"


def generate_text_and_extract_certainty(
    model,
    tokenizer,
    prompts,
    batch_size,
    max_new_tokens,
    device,
    original_data=None,
    max_retries=5,
):
    """Generate texts with the certainty prompt and retry on parsing failure."""
    generated_texts = []
    certainty_labels = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating certainty"):
            batch_prompts = prompts[i : i + batch_size]
            batch_original_data = (
                original_data[i : i + batch_size] if original_data else [None] * len(batch_prompts)
            )

            for prompt_idx, (prompt, orig_item) in enumerate(
                zip(batch_prompts, batch_original_data)
            ):
                generated_text = ""
                certainty_label = ""
                certainty_code = 0
                retry_count = 0

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
                        max_length=768,
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

                    certainty_label, certainty_code = extract_certainty_field(generated_text)
                    if is_extraction_successful(certainty_code):
                        break

                    retry_count += 1
                    time.sleep(random.uniform(0.1, 0.5))

                if not is_extraction_successful(certainty_code):
                    print(
                        f"Failed to extract certainty for sample {i + prompt_idx} "
                        f"after {max_retries} retries"
                    )

                generated_texts.append(generated_text.strip())
                certainty_labels.append(certainty_label)

    return generated_texts, certainty_labels


def _save_result(result, output_path, split_name):
    """Save dataset to disk."""
    output_file = os.path.join(output_path, f"{split_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved generated {split_name} data to {output_file}")


def _merge_results(results):
    """Merge results from different GPUs."""
    merged_data = []
    for result in results:
        merged_data.extend(result["data"])
    return {"data": merged_data}


def _process_data_on_gpu(args, data, split_name, gpu_id):
    """Process a shard of data on the given GPU."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"GPU {gpu_id}: Using device: {device}")

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

    prompts = []
    for item in data:
        historical_data = item.get("historical_data", "")
        news = item.get("news", "")
        company_name = extract_company_names(news)
        mean_value = _compute_mean_value(historical_data)
        prompt = args.prompt_template.format(
            historical_data=historical_data,
            news=news,
            company_name=company_name,
            mean_value=mean_value,
        )
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts for {split_name}...")
    generated_texts, certainty_labels = generate_text_and_extract_certainty(
        model,
        tokenizer,
        prompts,
        args.batch_size,
        args.max_new_tokens,
        device,
        data,
    )

    new_data = []
    for original_item, generated_text, certainty_label in zip(
        data, generated_texts, certainty_labels
    ):
        new_item = original_item.copy()
        new_item["certainty_text"] = generated_text
        new_item["information_certainty_code"] = CERTAINTY_ENCODING.get(
            certainty_label.lower(), 0
        )
        new_data.append(new_item)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"data": new_data}


def _distributed_certainty_worker(local_rank: int, base_args: dict, split_name: str):
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
        _save_result(merged_result["data"], args.output_path, split_name)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Generate information certainty labels with retry for FNSPID dataset"
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
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_certainty",
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
        default=["test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are a professional financial analyst. Your task is to analyze the certainty of the information presented in "
            "the following news text and classify it into one of three categories based on the level of confidence conveyed. "
            "Your analysis should incorporate both the news text and the historical data, with a focus on how strong or weak "
            "the news signals appear in combination with price movements.\n\n"
            "Instructions:\n\n"
            "You are given the following inputs:\n\n"
            "News text: {news}\n\n"
            "Historical Data: {historical_data} (past stock price movements)\n\n"
            "Company name: {company_name}\n\n"
            "Use the following definitions to assess the level of confidence:\n\n"
            "High Confidence: The information in the text is presented with strong certainty, using definitive or conclusive "
            "language. Words like “confirmed,” “guaranteed,” “assured,” or “certain” indicate high confidence. If the stock "
            "price shows significant movement (greater than 5%) that aligns with the news, this indicates High Confidence. "
            "You should choose High Confidence when both the language and price action strongly align.\n\n"
            "Low Confidence: The information is presented with ambiguity, using words like “possibly,” “uncertain,” “might,” "
            "or “potentially”. The stock price should also show little to no movement (less than 2%). If the stock price "
            "fluctuates marginally and the news has a lot of hedging, then Low Confidence is appropriate. This label reflects "
            "uncertainty in both the language and the price behavior.\n\n"
            "Medium Confidence: If the news expresses moderate certainty (using terms like “likely,” “expected,” “may,” etc.), "
            "but the stock price movement is in the 2% to 5% range, use Medium Confidence. This reflects cases where the news "
            "presents some uncertainty, but there is moderate price action.\n\n"
            "Output Format:\n\n"
            "You MUST output in the following EXACT format with no extra text:\n\n"
            "Information Certainty: <High Confidence / Medium Confidence / Low Confidence>\n"
        ),
        help="Certainty prompt template for text generation",
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
                _distributed_certainty_worker,
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
            _save_result(result["data"], args.output_path, split_name)


if __name__ == "__main__":
    main()

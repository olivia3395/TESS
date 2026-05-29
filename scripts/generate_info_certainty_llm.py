#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run an LLM to classify news information certainty (High/Medium/Low) for ver_camf-style datasets.

Prompt template (fill news + company_name):
You are a financial analyst. Your task is to analyze the certainty of the information presented in the following news text and classify it into one of three categories based on the level of confidence conveyed:

High Confidence: The information in the text is presented with strong certainty, using definitive or conclusive language. Examples include terms like “confirmed,” “certain,” “definite,” etc.

Medium Confidence: The information in the text expresses a moderate level of certainty, but also includes some hedging or uncertainty. Examples include terms like “likely,” “expected,” “may,” etc.

Low Confidence: The information in the text is presented with ambiguity, lacking clear certainty. This can include terms like “possibly,” “uncertain,” “might,” “potentially,” etc.

Instructions:

Analyze the following news text {news} and identify how certain the information presented is.

Classify the text based on the level of confidence it expresses using the categories provided.

Output:
You MUST output in the following EXACT format with no extra text:

Information Certainty: <High Confidence / Medium Confidence / Low Confidence>

Company: {company_name}
News: {news}

The script retries generation when parsing fails, and saves label+score (High=3, Medium=2, Low=1) plus raw LLM output.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

Company: {company_name}
News: {news}
"""

LABEL_SCORES = {
    "High Confidence": 3,
    "Medium Confidence": 2,
    "Low Confidence": 1,
}

LABEL_PATTERN = re.compile(r"Information Certainty:\s*(High Confidence|Medium Confidence|Low Confidence)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based Information Certainty classification with retry")
    parser.add_argument("--input", default="dataset/FNSPID/ver_camf", help="Input dataset dir containing train/vali/test json")
    parser.add_argument("--output", default="dataset/FNSPID/ver_camf_info_certainty_llm", help="Output dir for enriched json")
    parser.add_argument("--model-path", default="pretrain_model/EmbeddingModel/Qwen3-Embedding-8B", help="Local HF model path")
    parser.add_argument("--device", default="cuda:0", help="Device for generation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--retries", type=int, default=5, help="Max retries when parsing fails")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(news: str, company_name: str = "Unknown") -> str:
    return PROMPT_TEMPLATE.format(news=news, company_name=company_name)


def classify_text(model, tokenizer, prompt: str, device: str, args) -> Tuple[str, str]:
    """Generate and parse a single sample with retries. Returns (label, raw_output)."""
    for attempt in range(args.retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        match = LABEL_PATTERN.search(text)
        if match:
            label = match.group(1).title()
            # Normalize capitalization to keys in LABEL_SCORES
            label = {"High confidence": "High Confidence",
                     "Medium confidence": "Medium Confidence",
                     "Low confidence": "Low Confidence"}.get(label, label)
            if label in LABEL_SCORES:
                return label, text
    raise RuntimeError("Failed to parse label after retries")


def process_split(args, split: str, model, tokenizer) -> Dict[str, int]:
    in_file = Path(args.input) / f"{split}.json"
    data = json.load(in_file.open())
    if args.max_samples:
        data = data[: args.max_samples]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.json"

    label_counter = Counter()
    enriched = []
    for item in tqdm(data, desc=f"{split}"):
        news = str(item.get("news", ""))
        company = "Unknown"
        prompt = build_prompt(news=news, company_name=company)
        try:
            label, raw = classify_text(model, tokenizer, prompt, args.device, args)
        except RuntimeError:
            label, raw = "Low Confidence", ""  # fallback
        score = LABEL_SCORES.get(label, 1)
        new_item = dict(item)
        new_item["info_certainty_label_llm"] = label
        new_item["info_certainty_score_llm"] = score
        new_item["info_certainty_raw_output"] = raw
        enriched.append(new_item)
        label_counter[label] += 1

    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
    return label_counter


def main():
    args = parse_args()
    device = args.device
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    stats = {}
    for split in ["train", "vali", "test"]:
        stats[split] = process_split(args, split, model, tokenizer)

    print("Label distribution:")
    for split, counter in stats.items():
        print(split, dict(counter))


if __name__ == "__main__":
    main()

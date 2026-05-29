#!/usr/bin/env python3
"""DPO fine-tuning pipeline for Qwen3-4B using TRL turn rewards."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset 
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig as TRLDPOConfig

PROMPT_TEMPLATE = (
    "You are a sell-side equity analyst producing a five-day market outlook.\n"
    "Historical prices: {historical}\n"
    "Use the news context to write a refreshed brief covering catalysts, technicals, and the five-day path.\n"
    "Answer in English with concise bullet points."
)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "dataset/FNSPID/trl_turn1/dpo_pairs_scaled.jsonl"
DEFAULT_MODEL_PATH = BASE_DIR / "pretrain_model/ReasoningModel/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output/dpo_qwen3_turn1"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "dpo_log"



class PatchedDPOTrainer(TRLDPOTrainer):
    """Compatibility shim for transformers>=4.44 where Trainer signatures changed."""

    def get_batch_samples(self, epoch_iterator, num_batches, device=None):
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches, device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def log(self, logs, start_time=None):
        return super().log(logs)


@dataclass
class DPOConfig:
    data_path: Path
    model_path: Path
    output_dir: Path
    log_dir: Path
    reward_key: str = "reward_scaled"
    beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    fp16: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: str = "q_proj,v_proj,k_proj,o_proj"
    resume_training: bool = True
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True


def parse_args() -> DPOConfig:
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning for Qwen3-4B")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--reward-key", default="reward_scaled")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj,k_proj,o_proj",
                        help="Comma separated projection names for LoRA")
    parser.add_argument("--resume", dest="resume", action="store_true",
                        help="Resume from the latest checkpoint if available (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Disable resuming from checkpoints")
    parser.add_argument("--load-4bit", dest="load_in_4bit", action="store_true",
                        help="Load the base model in 4-bit (QLoRA) mode (default)")
    parser.add_argument("--no-load-4bit", dest="load_in_4bit", action="store_false",
                        help="Load the base model in full precision")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage (default)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                        help="Disable gradient checkpointing")
    parser.set_defaults(resume=True, load_in_4bit=True, gradient_checkpointing=True)
    args = parser.parse_args()
    return DPOConfig(
        data_path=Path(args.data_path),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        log_dir=Path(args.log_dir),
        reward_key=args.reward_key,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        resume_training=args.resume,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def build_dataset(config: DPOConfig) -> Dataset:
    data = load_jsonl(config.data_path)
    prompts, chosen, rejected = [], [], []
    rewards = []
    for row in data:
        prompt = PROMPT_TEMPLATE.format(historical=row["historical_data"])
        prompts.append(prompt)
        chosen.append(row["news_positive"])
        rejected.append(row["news_negative"])
        rewards.append(row.get(config.reward_key, 0.0))
    ds = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
        "reward": rewards,
    })
    return ds


def _infer_cuda_device() -> int | None:
    if not torch.cuda.is_available():
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.cuda.current_device()


def init_model_and_tokenizer(model_path: Path, max_seq_length: int, *,
                             load_in_4bit: bool, gradient_checkpointing: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = None
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    device_index = _infer_cuda_device()
    device_str = None
    if device_index is not None:
        device_str = f"cuda:{device_index}"

    quant_config = None
    device_map = None
    if load_in_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype or torch.float16,
        )
        target_device = device_index if device_index is not None else 0
        device_map = {"": target_device}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=None if quant_config else compute_dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    if quant_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif device_str is not None:
        model = model.to(device_str)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            embedding_layer = model.get_input_embeddings()

            def _make_inputs_require_grad(module, _inp, output):
                return output.requires_grad_()

            embedding_layer.register_forward_hook(_make_inputs_require_grad)
    model.config.use_cache = False
    tokenizer.model_max_length = max_seq_length
    return model, tokenizer


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = parse_args()
    resume_from_checkpoint = None
    if cfg.resume_training and cfg.output_dir.exists():
        resume_from_checkpoint = get_last_checkpoint(str(cfg.output_dir))
        if resume_from_checkpoint:
            print(f"[INFO] Resume enabled, latest checkpoint detected at {resume_from_checkpoint}")
        else:
            print("[INFO] Resume enabled but no checkpoint found. Starting fresh training run.")

        if not resume_from_checkpoint:
            # gracefully resume from PEFT checkpoint directory naming if present
            checkpoint_dirs = sorted(cfg.output_dir.glob("checkpoint-*"))
            if checkpoint_dirs:
                resume_from_checkpoint = str(checkpoint_dirs[-1])
                print(f"[INFO] Found manual checkpoint directory {resume_from_checkpoint}, resuming from it.")

    ensure_dirs(cfg.output_dir, cfg.log_dir)

    print(f"[INFO] Loading dataset from {cfg.data_path}")
    dataset = build_dataset(cfg)
    print(f"[INFO] Loaded {len(dataset)} preference pairs")

    print(f"[INFO] Loading model from {cfg.model_path}")
    model, tokenizer = init_model_and_tokenizer(
        cfg.model_path,
        cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = cfg.fp16 and use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = cfg.fp16 and use_cuda and not use_bf16

    dpo_args = TRLDPOConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=["tensorboard"],
        disable_tqdm=False,
        logging_dir=str(cfg.log_dir),
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
    dpo_args.beta = cfg.beta
    dpo_args.max_length = cfg.max_seq_length
    dpo_args.max_prompt_length = cfg.max_seq_length
    dpo_args.do_train = True

    target_modules = [m.strip() for m in cfg.target_modules.split(',') if m.strip()]
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    trainer = PatchedDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        max_prompt_length=cfg.max_seq_length,
        peft_config=peft_config,
    )

    print("[INFO] Starting DPO training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)

    stats_path = cfg.output_dir / "dpo_training_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Training complete. Artifacts saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()

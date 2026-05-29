#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
带有验证和重新生成机制的改进版生成代码
针对Environment数据集的Shape分类任务生成文本并提取相应字段
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import gc
import spacy
from collections import Counter
import re
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
import numpy as np
import random
import time
import socket

nlp = spacy.load("en_core_web_sm")

def find_free_port():
    """
    动态查找可用端口
    通过创建临时socket绑定端口后释放，获取系统分配的可用端口
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def load_fnspid_data(data_path, split):
    """加载Environment数据集"""
    file_path = os.path.join(data_path, f"{split}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def _parse_gpu_list(gpu_string: str):
    """解析GPU列表"""
    entries = [token.strip() for token in gpu_string.split(',') if token.strip()]
    if not entries:
        raise ValueError('Invalid --gpus argument: no GPU ids found')
    try:
        return [int(entry) for entry in entries]
    except ValueError as exc:
        raise ValueError(f'Invalid GPU id in --gpus: {gpu_string}') from exc

def _split_dataset_ranges(total_size: int, num_gpus: int) -> list:
    """
    将数据集均匀分配给多个GPU
    返回每个GPU应处理的索引范围 [(start, end), ...]
    """
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
    """
    分布式工作进程，每个GPU处理数据集的一部分
    """
    # 设置当前进程的GPU设备
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式环境
    world_size = len(base_args.get('gpu_ids', [0]))
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # 使用动态分配的端口而不是静态端口
    os.environ['MASTER_PORT'] = str(base_args.get('master_port', '29507'))
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size, device_id=torch.device(f'cuda:{local_rank}'))
    
    # 为当前进程分配数据
    args = argparse.Namespace(**base_args)
    args.gpu_id = local_rank
    
    # 加载整个数据集
    original_data = load_fnspid_data(args.data_path, split_name)
    if args.test_limit is not None:
        original_data = original_data[:args.test_limit]
    # 计算当前GPU需要处理的数据范围
    data_ranges = _split_dataset_ranges(len(original_data), world_size)
    start_idx, end_idx = data_ranges[local_rank]
    partial_data = original_data[start_idx:end_idx]
    
    print(f"GPU {local_rank}: Processing {split_name} split, indices {start_idx}-{end_idx} ({len(partial_data)} samples)")
    
    # 在当前GPU上处理分配的数据
    result = _process_data_on_gpu(args, partial_data, split_name, local_rank)
    
    # 使用torch.distributed.all_gather收集所有GPU的结果
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, (local_rank, result))
    
    # 只在主GPU上合并结果并保存
    if local_rank == 0:
        # 按照GPU rank排序结果
        all_results.sort(key=lambda x: x[0])
        merged_result = _merge_results([r[1] for r in all_results])
        _save_merged_result(merged_result, args.output_path, split_name)
    
    dist.destroy_process_group()

def _process_data_on_gpu(args, data, split_name, gpu_id):
    """
    在指定GPU上处理数据
    """
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"GPU {gpu_id}: Using device: {device}")
    
    # 加载模型和分词器
    print(f"GPU {gpu_id}: Loading Qwen3-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map={"": device} if device.type == 'cuda' else None,
        trust_remote_code=True,
        local_files_only=True,
        
    )
    model = model.to(device)
    model.eval()
    
    # 构建prompt
    prompts = []
    for item in data:
        historical_data = item.get('historical_data', '')
        news = item.get('news', '')
        prompt = args.prompt_template.format(
            historical_data=historical_data, 
            news=news, 
        )
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts and extracting embeddings for {split_name}...")
    generated_texts, hidden_states, attention_masks, new_data = generate_text_and_extract_hidden_states_with_retry(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device, data
    )
    
    # 清理GPU内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'data': new_data,
        'hidden_states': hidden_states,
        'attention_masks': attention_masks,
    }

def _merge_results(results):
    """
    合并来自不同GPU的结果
    """
    merged_data = []
    all_hidden_states = []
    all_attention_masks = []
    
    for result in results:
        merged_data.extend(result['data'])
        if result['hidden_states'] is not None:
            all_hidden_states.append(result['hidden_states'])
        if result['attention_masks'] is not None:
            all_attention_masks.append(result['attention_masks'])
    
    # 合并张量
    merged_hidden_states = torch.cat(all_hidden_states, dim=0) if all_hidden_states else None
    merged_attention_masks = torch.cat(all_attention_masks, dim=0) if all_attention_masks else None
    
    return {
        'data': merged_data,
        'hidden_states': merged_hidden_states,
        'attention_masks': merged_attention_masks,
    }

# 使用正则表达式来匹配Shape行
SHAPE_PATTERN = re.compile(
    r"Shape\s*:\s*(Rise|Fall|Peak|Recover|Oscillate)",
    re.IGNORECASE,
)

SHAPE_ENCODING = {
    "Rise": 1,
    "Fall": 2,
    "Peak": 3,
    "Recover": 4,
    "Oscillate": 5,
}


def extract_shape_field(text: str) -> str:
    """
    从生成的文本中提取Shape字段
    """
    # 查找 "Shape:" 后面的标签
    match = re.search(r"[sS]hape\s*:\s*([A-Za-z]+)", text)
    if not match:
        return ''
    
    # 直接返回原始标签
    return match.group(1).strip()


def is_extraction_successful(shape_label: str) -> bool:
    """
    检查提取的标签是否有效
    """
    # 检查标签是否为有效值
    valid_labels = ['Rise', 'Fall', 'Peak', 'Recover', 'Oscillate']
    return shape_label in valid_labels


def generate_text_and_extract_hidden_states_with_retry(model, tokenizer, prompts, batch_size=4, max_new_tokens=768, device='cuda', original_data=None):
    """使用Qwen3-8B根据prompt生成新文本，并在提取失败时重试"""
    generated_texts = []
    all_hidden_states = []
    all_attention_masks = []
    new_data = []  # 初始化new_data列表
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating texts"):
            batch_prompts = prompts[i:i+batch_size]
            batch_original_data = original_data[i:i+batch_size] if original_data else [None] * len(batch_prompts)
            
            # 为批处理中的每个样本分别生成文本
            for prompt_idx, (prompt, orig_item) in enumerate(zip(batch_prompts, batch_original_data)):
                generated_text = None
                retry_count = 0
                max_retries = 3  # 最多重试5次
                
                while retry_count < max_retries:
                    # 构建消息格式
                    messages = [{"role": "user", "content": prompt}]
                    
                    # 应用chat template
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    
                    # 编码输入
                    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=768)
                    input_ids = model_inputs['input_ids'].to(device)
                    attention_mask = model_inputs['attention_mask'].to(device)
                     
                    # 生成文本
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                    
                    # 解码生成的文本
                    output_ids = generated_ids[0][len(input_ids[0]):]
                    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    # 清理思考标签
                    if '<think>' in generated_text and '</think>' in generated_text:
                        # 提取思考标签之后的内容
                        after_think_end = generated_text.split('</think>', 1)
                        if len(after_think_end) > 1:
                            # 使用标签之后的内容
                            generated_text = after_think_end[1].strip()
                        else:
                            # 如果有开始标签但没有结束标签，则移除整个思考块
                            before_think = generated_text.split('<think>', 1)[0]
                            generated_text = before_think.strip()
                    elif '<think>' in generated_text:
                        # 只有开始标签，没有结束标签，移除思考块内容
                        before_think = generated_text.split('<think>', 1)[0]
                        generated_text = before_think.strip()
                    
                    # 立即进行文本截断处理，确保从Step1开始
                    step1_match = re.search(r'(?:\d+\s*:)?\s*Step\s*1\s*:', generated_text, re.IGNORECASE)
                    if step1_match:
                        generated_text = generated_text[step1_match.start():]
                    
                    # 提取Shape字段
                    shape_label = extract_shape_field(generated_text)  # 获取原始标签
                    
                    # 检查提取是否成功
                    if is_extraction_successful(shape_label):  # 检查标签是否有效
                        break  # 成功提取，跳出重试循环
                    
                    retry_count += 1
                    # print(f"Retry {retry_count}/{max_retries} for sample {i+prompt_idx}")
                    # 添加随机延迟避免过于频繁的重试
                    time.sleep(random.uniform(0.1, 0.5))
                
                # 保存生成的文本和提取的标签
                new_item = orig_item.copy()  # 修复变量名错误，原来是original_item
                new_item['news'] = generated_text.strip() if generated_text else ""
                
                if is_extraction_successful(shape_label):  # 如果成功提取标签
                    # 直接保存原始标签而不是编码
                    new_item['shape'] = shape_label
                else:
                    new_item['shape'] = ''  # 提取失败时保存空字符串
                
                new_data.append(new_item)
    
    return generated_texts, None, None, new_data  # 添加new_data作为返回值


def _save_merged_result(result, output_path, split_name):
    """
    保存合并后的结果
    """
    # 保存新数据集
    output_file = os.path.join(output_path, f"{split_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result['data'], f, ensure_ascii=False, indent=2)
    print(f"Saved generated {split_name} data to {output_file}")

def extract_company_names(news_text):
    """
    从新闻文本中提取公司名称
    """
    doc = nlp(news_text)
    companies = []
    
    # 提取ORG类型的实体（组织机构，包括公司）
    for ent in doc.ents:
        if ent.label_ == "ORG":
            companies.append(ent.text)
    
    # 如果没有找到ORG实体，可以尝试基于常见公司后缀的正则表达式
    if not companies:
        company_patterns = [
            r'\b[A-Z][a-zA-Z]+\s+(?:Inc|Corp|Corporation|Ltd|Limited|Company|Co|Group|Technologies|Tech)\b',
            r'\b(?:Apple|Google|Microsoft|Amazon|Facebook|Tesla|Netflix|Intel|IBM|Oracle)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, news_text)
            companies.extend(matches)
    
    # 返回最常见的公司名称（如果有多个）
    if companies:
        most_common = Counter(companies).most_common(1)
        return most_common[0][0] if most_common else companies[0]
    
    return "Unknown Company"

def main():
    parser = argparse.ArgumentParser(description='Generate Qwen texts using improved approach with retry mechanism for Environment dataset')
    parser.add_argument('--data-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/Environment/ver_camf', 
                       help='Path to Environment dataset')
    parser.add_argument('--output-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/Environment/ver_shape', 
                       help='Output path for generated texts')
    parser.add_argument('--model-path', type=str, 
                       default='/ssd/hf_home/models/Qwen3-8B',
                       help='Path to Qwen3-8B model')
    parser.add_argument('--splits', nargs='+', default=['train','test','vali'], 
                       help='Dataset splits to process')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for generation and encoding')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum new tokens to generate')
    
    # Environment数据集的Shape任务提示模板
    parser.add_argument('--prompt-template', type=str, 
                   default="You are a professional environmental analyst specializing in air quality forecasting. Analyze the 48-step air quality sequence {{{historical_data}}} and related environmental news {news} to determine the overall shape pattern.\n\nClassify the shape into one of five categories:\n- Rise: Generally upward trend over the sequence. The overall pattern shows an increase from beginning to end, even if there are some temporary pullbacks or fluctuations along the way. Minor fluctuations are acceptable as long as the general direction is upward.\n- Fall: Generally downward trend over the sequence. The overall pattern shows a decrease from beginning to end, even if there are some temporary rebounds or fluctuations along the way. Minor fluctuations are acceptable as long as the general direction is downward.\n- Peak: The sequence shows an overall pattern of rising in the first half (approximately first 24 steps) and then falling in the second half (approximately last 24 steps). There may be minor fluctuations, but the dominant pattern is an increase followed by a decrease. The turning point does not need to be exactly at the midpoint.\n- Recover: The sequence shows an overall pattern of falling in the first half (approximately first 24 steps) and then rising in the second half (approximately last 24 steps). There may be minor fluctuations, but the dominant pattern is a decrease followed by an increase. The turning point does not need to be exactly at the midpoint.\n- Oscillate: The sequence shows multiple clear direction changes (three or more turning points) with alternating up and down movements. There is no dominant overall trend in one direction.\n\nImportant classification guidelines:\n- For Rise and Fall: Focus on the overall trend from start to end. Allow for minor fluctuations and temporary reversals. If the sequence generally increases from beginning to end, classify as Rise. If it generally decreases, classify as Fall.\n- For Peak and Recover: Look for a dominant pattern where the first half and second half show opposite trends. Minor fluctuations within each half are acceptable. The turning point can occur anywhere between steps 20-28, not necessarily exactly at step 24.\n- For Oscillate: Use this when there are multiple clear direction changes (at least three turning points) creating a wave-like pattern without a clear overall trend.\n- When in doubt between categories, prioritize the dominant overall pattern over minor fluctuations.\n\nAir quality-specific considerations:\n- Air quality patterns are influenced by meteorological factors (temperature, humidity, wind patterns)\n- Environmental regulations and policy changes can cause significant directional changes\n- Pollution sources, seasonal patterns, and weather conditions strongly influence short-term movements\n- Industrial activities and traffic patterns affect daily air quality fluctuations\n- Natural events (wildfires, dust storms, volcanic eruptions) can cause extreme patterns\n- For 48-step sequences, consider the full temporal context and allow for natural variability within the overall trend\n\nOutput ONLY in this exact format with no additional explanations, analysis, or context:\nShape:<Rise or Fall or Peak or Recover or Oscillate>\n\nIMPORTANT: ONLY output the specified format above with no other content before or after.",
                   help='Shape classification prompt template for text generation')

    parser.add_argument('--test-limit', type=int, default=None,
                   help='限制处理的样本数量，用于测试 (默认: None，处理所有样本)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for parallel processing')
    args = parser.parse_args()
    
    # 处理GPU参数
    if args.gpus:
        gpu_ids = _parse_gpu_list(args.gpus)
    else:
        gpu_ids = [args.gpu_id]
        
    if len(gpu_ids) > 1:
        # 设置分布式环境变量
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        # 动态分配端口而不是使用静态端口
        master_port = find_free_port()
        os.environ.setdefault('MASTER_PORT', str(master_port))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in gpu_ids)
        
        # 准备参数
        base_args = vars(args)
        base_args['gpu_ids'] = gpu_ids
        base_args['master_port'] = master_port  # 将动态分配的端口传递给worker进程
        
        # 创建输出目录
        os.makedirs(args.output_path, exist_ok=True)
        
        # 依次处理每个数据集分割
        for split_name in args.splits:
            print(f"Processing {split_name} split with {len(gpu_ids)} GPUs...")
            
            # 启动分布式处理
            mp.spawn(
                _distributed_embedding_worker,
                nprocs=len(gpu_ids),
                args=(base_args, split_name),
                join=True,
            )
    else:
        # 单GPU处理逻辑（保持原有逻辑）
        print("Using single GPU processing...")
        # 这里可以调用原有的单GPU处理逻辑
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

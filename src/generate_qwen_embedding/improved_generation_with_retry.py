#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
带有验证和重新生成机制的改进版生成代码
当趋势和强度提取失败时，重新生成文本直到成功提取为止
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
    """加载FNSPID数据集"""
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
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    
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

def extract_trend_from_news(news_text):
    """
    从生成的新闻文本中提取趋势词(Rising/Falling/Stable)
    """
    # 定义趋势关键词
    trend_keywords = ['Rising', 'Falling', 'Stable']
    
    # 直接匹配单个词的情况
    if news_text.strip() in trend_keywords:
        return news_text.strip()
    
    # 尝试从复杂文本中提取趋势词
    for keyword in trend_keywords:
        if keyword in news_text:
            return keyword
    
    # 如果找不到趋势词，默认返回Stable
    return 'Stable'

def clean_generated_text(text):
    """
    清理生成的文本，删除"Day 1"之前的所有字符
    """
    day1_index = text.find("Day 1")
    if day1_index != -1:
        return text[day1_index:]
    return text

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
        company_name = extract_company_names(news)
        few_shots = args.few_shots
        mean_value = sum([float(x.strip()) for x in historical_data.split(',')])/5
        prompt = args.prompt_template.format(
            historical_data=historical_data, 
            news=news, 
            company_name=company_name,
            few_shots=few_shots, 
            mean_value=mean_value
        )
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts and extracting embeddings for {split_name}...")
    generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states_with_retry(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device, data
    )
    
    # 创建新数据集（包含原始数据和生成的文本）
    new_data = []
    for i, (original_item, generated_text) in enumerate(zip(data, generated_texts)):
        new_item = original_item.copy()
        new_item['news'] = generated_text
        strengths, trends, global_trend = extract_structured_trend_fields(generated_text)
        new_item['step_strengths'] = strengths
        new_item['step_trends'] = trends
        new_item['global_trend'] = global_trend
        new_data.append(new_item)
    
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

# 使用更灵活的正则表达式来匹配步骤行
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

def extract_structured_trend_fields(news_text, num_steps=5):
    """解析生成文本并返回数值化的强度、趋势以及全局趋势"""
    # 添加截断逻辑，去除Step1前的文本（通常是问题重述）
    step1_pattern = re.compile(r'(Step\s*1[:\s]|Step1[:\s])', re.IGNORECASE)
    step1_match = step1_pattern.search(news_text)
    if step1_match:
        # 截取Step1及之后的内容
        news_text = news_text[step1_match.start():]
    
    strengths = [0] * num_steps  # 0 表示未解析到有效强度
    trends = [0] * num_steps    # -1 表示未解析到有效趋势
    
    # 存储所有匹配到的结果，保持顺序
    matches = []
    for match in STEP_LINE_PATTERN.finditer(news_text):
        step_num_str = match.group(1).strip()
        raw_strength = match.group(2).strip().lower()
        raw_trend = match.group(3).strip().lower()
        
        # 处理step1没有编号的情况
        if step_num_str == "":
            step_idx = 0  # 空编号默认为step1
        else:
            step_idx = int(step_num_str) - 1
            
        matches.append((step_idx, raw_strength, raw_trend))
    
    # 如果没有找到标准格式的匹配项，尝试使用更宽松的模式
    if not matches:
        # 尝试匹配更宽松的模式，例如 "Analysis: ... strength: ..., trend: ..."
        loose_pattern = re.compile(
            r"(?:\d+\s*[:\.]\s*)?[Aa]nalysis:.*?[sS]trength[:\s]*([A-Za-z]+).*?[tT]rend[:\s]*([A-Za-z]+)",
            re.IGNORECASE
        )
        
        for i, match in enumerate(loose_pattern.finditer(news_text)):
            raw_strength = match.group(1).strip().lower()
            raw_trend = match.group(2).strip().lower()
            
            # 按顺序分配步骤索引
            step_idx = i if i < num_steps else num_steps - 1
            matches.append((step_idx, raw_strength, raw_trend))
    
    # 如果仍然没有匹配项，尝试按行分割文本并逐行分析
    if not matches:
        lines = news_text.split('\n')
        step_count = 0
        i = 0
        while i < len(lines) and step_count < num_steps:
            line = lines[i].strip()
            # 检查是否包含"Analysis"关键词或Step关键词
            if ('analysis' in line.lower() or re.search(r'Step\d*:', line, re.IGNORECASE)) and ':' in line:
                # 查找接下来几行中的强度和趋势
                strength = None
                trend = None
                
                # 在接下来的几行中查找强度和趋势
                for j in range(1, 4):  # 查找接下来的3行
                    if i + j < len(lines):
                        next_line = lines[i + j].strip()
                        if 'strength:' in next_line.lower() and not strength:
                            strength_match = re.search(r'[sS]trength[:\s]*([A-Za-z]+)', next_line, re.IGNORECASE)
                            if strength_match:
                                strength = strength_match.group(1).strip().lower()
                        elif 'trend:' in next_line.lower() and not trend:
                            trend_match = re.search(r'[tT]rend[:\s]*([A-Za-z]+)', next_line, re.IGNORECASE)
                            if trend_match:
                                trend = trend_match.group(1).strip().lower()
                
                if strength and trend:
                    # 按顺序分配步骤索引
                    step_idx = step_count
                    matches.append((step_idx, strength, trend))
                    step_count += 1
            i += 1
    
    # 增加容错机制：如果Step1标签丢失，但Step2之前有匹配项，则将其作为Step1的结果
    # 查找Step2第一次出现的位置
    step2_first_index = None
    for i, (step_idx, _, _) in enumerate(matches):
        if step_idx == 1:  # Step2
            step2_first_index = i
            break
    
    # 如果找到了Step2，且Step2不是第一个匹配项
    if step2_first_index is not None and step2_first_index > 0:
        # 检查Step1是否在Step2之前已经出现过
        step1_exists_before_step2 = False
        for i in range(step2_first_index):
            if matches[i][0] == 0:  # 找到了Step1
                step1_exists_before_step2 = True
                break
        
        # 如果Step1在Step2之前没有出现，则将Step2之前第一个匹配项视为Step1
        if not step1_exists_before_step2:
            step_idx, raw_strength, raw_trend = matches[step2_first_index - 1]
            strengths[0] = STRENGTH_ENCODING.get(raw_strength, 0)
            trends[0] = TREND_ENCODING.get(raw_trend, -1)
    
    # 处理所有匹配项
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

def is_extraction_successful(strengths, trends):
    """
    检查提取是否成功
    更严格的标准是所有步骤的趋势和强度都必须被成功提取（非默认值）
    """
    # 检查是否所有步骤的趋势和强度都不是默认值
    for strength, trend in zip(strengths, trends):
        if strength == 0 or trend == 0:
            return False
    return True

def generate_text_and_extract_hidden_states_with_retry(model, tokenizer, prompts, batch_size=4, max_new_tokens=768, device='cuda', original_data=None):
    """使用Qwen3-8B根据prompt生成新文本，并在提取失败时重试"""
    generated_texts = []
    all_hidden_states = []
    all_attention_masks = []
    
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
                max_retries = 5  # 最多重试3次
                
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
                    elif '' in generated_text:
                        # 只有开始标签，没有结束标签，移除思考块内容
                        before_think = generated_text.split('<think>', 1)[0]
                        generated_text = before_think.strip()
                    
                    # 立即进行文本截断处理，确保从Step1开始
                    step1_match = re.search(r'(?:\d+\s*:)?\s*Step\s*1\s*:', generated_text, re.IGNORECASE)
                    if step1_match:
                        generated_text = generated_text[step1_match.start():]
                    
                    # 提取结构化趋势字段
                    strengths, trends, global_trend = extract_structured_trend_fields(generated_text)
                    
                    # 检查提取是否成功
                    if is_extraction_successful(strengths, trends):
                        break  # 成功提取，跳出重试循环
                    
                    retry_count += 1
                    # print(f"Retry {retry_count}/{max_retries} for sample {i+prompt_idx}")
                    # 添加随机延迟避免过于频繁的重试
                    time.sleep(random.uniform(0.1, 0.5))
                
                if generated_text is None or retry_count >= max_retries:
                    print(f"Failed to generate valid text for sample {i+prompt_idx} after {max_retries} retries")
                    # 即使失败也使用最后一次生成的文本
                    if generated_text is None:
                        generated_text = ""
                
                generated_texts.append(generated_text.strip())
    
    return generated_texts, None, None

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
    parser = argparse.ArgumentParser(description='Generate Qwen texts using improved approach with retry mechanism for FNSPID dataset')
    parser.add_argument('--data-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_camf', 
                       help='Path to FNSPID dataset')
    parser.add_argument('--output-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_8BInstruct_withfewshots', 
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
    
    # 修正提示模板，正确处理company_name作为填充字段
    parser.add_argument('--prompt-template', type=str, 
                   default="You are a professional financial analyst. Your task is to analyze stock price movements and related news to determine trend direction and strength for each step, then conclude the global trend.\
\n\nInstructions:\
\n1. For each of the 5 steps (Step1-Step5), you will analyze one steps stock price from the sequence {{{historical_data}}} compared to the historical mean value {mean_value}.\
\n2. Determine trend direction: Rising if step price > mean_value, Falling if step price < mean_value.\
\n3. Determine trend strength based on the magnitude of difference between step price and mean_value, choosing ONLY from: Emerging, Moderate, Significant, Prominent, Dominant.\
\n4. Determine the Global trend by comparing the predicted future mean value with the historical mean value {mean_value}.\
\n\nYou MUST output in the following EXACT format with no extra text:\n\
Step1: Analysis:<brief analysis for step 1>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n\
Step2: Analysis:<brief analysis for step 2>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n\
Step3: Analysis:<brief analysis for step 3>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n\
Step4: Analysis:<brief analysis for step 4>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n\
Step5: Analysis:<brief analysis for step 5>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n\
Global trend:<Rising or Falling>\n\
\nCompany: {company_name}\n\
Historical data: {{{historical_data}}}\n\
Mean value: {mean_value}\n\
News: {news}\n\
Few_shots: {few_shots}\n\
In the few_shots examples, the field trends uses -1 to represent Falling and 1 to represent Rising, and the field step_strengths uses integers 1-5 to represent, from weakest to strongest, the five strength labels: Emerging (1), Moderate (2), Significant (3), Prominent (4), Dominant (5).\n\
Provide your analysis in the exact format specified above:",
                   help='Improved prompt template for text generation')

    # ELECTRICITY_PROMPT_TEMPLATE (commented out to preserve original prompt):
    # ELECTRICITY_PROMPT_TEMPLATE = (
    #     "You are an electricity demand forecaster focused on Australian NEM regions. Use the historical half-hour load series and the provided scenario details to describe the direction and strength of expected next-day load across five phases.\n"
    #     "\n"
    #     "Inputs:\n"
    #     "- Historical load series (MW, 30-minute interval): {historical_data}\n"
    #     "- Baseline mean load from the historical series (MW): {mean_value}\n"
    #     "- Scenario context (region, dates, weekday/holiday status, weather, and local news with rationales): {news}\n"
    #     "\n"
    #     "Instructions:\n"
    #     "1. Consider the next-day outlook in five chronological phases: Step1 Overnight stability, Step2 Morning ramp, Step3 Midday plateau, Step4 Evening peak, Step5 Late-night tail.\n"
    #     "2. For each phase, judge if expected load is Rising (above mean_value) or Falling (below mean_value) using weather, holiday flags, weekday/weekend effects, and news-driven demand shifts.\n"
    #     "3. Choose strength ONLY from: Emerging, Moderate, Significant, Prominent, Dominant. Stronger labels reflect larger or more certain deviations from mean_value (e.g., extreme temperatures, major events, widespread outages).\n"
    #     "4. Set the Global trend as the overall next-day load level versus mean_value (Rising or Falling) considering the full context.\n"
    #     "\n"
    #     "You MUST output in the following EXACT format with no extra text:\n"
    #     "Step1: Analysis:<brief analysis for step 1>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
    #     "Step2: Analysis:<brief analysis for step 2>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
    #     "Step3: Analysis:<brief analysis for step 3>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
    #     "Step4: Analysis:<brief analysis for step 4>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
    #     "Step5: Analysis:<brief analysis for step 5>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
    #     "Global trend:<Rising or Falling>\n"
    #     "\n"
    #     "Do not output numeric forecasts; focus on directional reasoning linked to the context above."
    # )
    
    parser.add_argument('--test-limit', type=int, default=None,
                   help='限制处理的样本数量，用于测试 (默认: None，处理所有样本)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for parallel processing')
    parser.add_argument('--few-shots', type=str, default="src/generate_qwen_embedding/few_shot_samples.txt",
                   help=' few_shots')
    args = parser.parse_args()
    args.few_shots = open(args.few_shots, 'r', encoding='utf-8').read()  
    
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

if __name__ == "__main__":
    main()

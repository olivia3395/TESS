# src/generate_qwen_embedding/generate_qwen_embedding_distributed.py

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

nlp = spacy.load("en_core_web_sm")

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
    os.environ['MASTER_PORT'] = '29503'
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
        few_shots=args.few_shots
        mean_value = sum([float(x.strip()) for x in historical_data.split(',')])/5
        prompt = args.prompt_template.format(historical_data=historical_data, news=news, company_name=company_name,few_shots=few_shots,mean_value=mean_value)
        prompts.append(prompt)

    print(f"GPU {gpu_id}: Generating texts and extracting embeddings for {split_name}...")
    generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device
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
        #temp
        # new_item['news'] = extract_trend_from_news(generated_text)
        # new_item['news'] = clean_generated_text(new_item['news'])
        new_data.append(new_item)
    
    # 清理GPU内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    

        # 将大张量移到CPU以减少GPU内存使用
    # if hidden_states is not None:
    #     hidden_states = hidden_states.cpu()
    # if attention_masks is not None:
    #     attention_masks = attention_masks.cpu()
    
    return {
        'data': new_data,
        'hidden_states': hidden_states,
        'attention_masks': attention_masks,
        # 'generated_texts': generated_texts
    }



def _merge_results(results):
    """
    合并来自不同GPU的结果
    """
    merged_data = []
    all_hidden_states = []
    all_attention_masks = []
    # all_generated_texts = []
    
    for result in results:
        merged_data.extend(result['data'])
        if result['hidden_states'] is not None:
            all_hidden_states.append(result['hidden_states'])
        if result['attention_masks'] is not None:
            all_attention_masks.append(result['attention_masks'])
        # all_generated_texts.extend(result['generated_texts'])
    
    # 合并张量
    merged_hidden_states = torch.cat(all_hidden_states, dim=0) if all_hidden_states else None
    merged_attention_masks = torch.cat(all_attention_masks, dim=0) if all_attention_masks else None
    
    return {
        'data': merged_data,
        'hidden_states': merged_hidden_states,
        'attention_masks': merged_attention_masks,
        # 'generated_texts': all_generated_texts
    }



# STEP_LINE_PATTERN = re.compile(
#     r"Step\s*(\d)\s*:.*?strength:\s*([A-Za-z]+)\s*,\s*trend:\s*([A-Za-z]+)",
#     re.IGNORECASE,
# )
STEP_LINE_PATTERN = re.compile(
    r"Step\s*(\d*)\s*:.*?(?:[sS]trength):\s*([A-Za-z]+)\s*,\s*(?:[tT]rend):\s*([A-Za-z]+)",
    re.IGNORECASE,
)
GLOBAL_TREND_PATTERN = re.compile(
    r"Global\s*trend\s*:\s*(Rising|Falling)", re.IGNORECASE
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
    "falling": 0,
}


def extract_structured_trend_fields(news_text, num_steps=5):
    """解析生成文本并返回数值化的强度、趋势以及全局趋势"""
    strengths = [0] * num_steps  # 0 表示未解析到有效强度
    trends = [-1] * num_steps    # -1 表示未解析到有效趋势
    
    for match in STEP_LINE_PATTERN.finditer(news_text):
        step_num_str = match.group(1).strip()
        # 处理step1没有编号的情况
        if step_num_str == "":
            step_idx = 0  # 空编号默认为step1
        else:
            step_idx = int(step_num_str) - 1
            
        if 0 <= step_idx < num_steps:
            raw_strength = match.group(2).strip().lower()
            raw_trend = match.group(3).strip().lower()
            strengths[step_idx] = STRENGTH_ENCODING.get(raw_strength, 0)
            trends[step_idx] = TREND_ENCODING.get(raw_trend, -1)

    global_match = GLOBAL_TREND_PATTERN.search(news_text)
    if global_match:
        global_trend = TREND_ENCODING.get(global_match.group(1).strip().lower(), -1)
    else:
        global_trend = -1

    return strengths, trends, global_trend
def _save_merged_result(result, output_path, split_name):
    """
    保存合并后的结果
    """
    # 保存新数据集
    output_file = os.path.join(output_path, f"{split_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result['data'], f, ensure_ascii=False, indent=2)
    print(f"Saved generated {split_name} data to {output_file}")
    
    # 保存嵌入
    # if result['hidden_states'] is not None:
    #     split_file = os.path.join(output_path, f'{split_name}_news.pt')
    #     embeddings_dict = {
    #         f'{split_name}_news': result['hidden_states'],
    #         f'{split_name}_masks': result['attention_masks'],
    #         # f'{split_name}_texts': result['generated_texts']
    #     }
    #     torch.save(embeddings_dict, split_file)
    #     print(f"Saved {split_name} embeddings to {split_file}")
# 存储瓶颈，先忽略隐状态保存
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

def generate_text_and_extract_hidden_states(model, tokenizer, prompts, batch_size=4, max_new_tokens=768, device='cuda'):
    """使用Qwen3-8B根据prompt生成新文本并同时提取最后一层隐藏状态"""
    generated_texts = []
    all_hidden_states = []
    all_attention_masks = []
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating texts and extracting embeddings"):
            batch_prompts = prompts[i:i+batch_size]
            
            # 构建消息格式
            messages_batch = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                messages_batch.append(messages)
            
            # 应用chat template
            texts = []
            for messages in messages_batch:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    # enable_thinking=False  # 根据需要调整
                )
                texts.append(text)
            
            # 编码输入
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=768)
            input_ids = model_inputs['input_ids'].to(device)
            attention_mask = model_inputs['attention_mask'].to(device)
            print(len(input_ids), 7681)
            # 生成文本
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                # temperature=0.7,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
            # 解码生成的文本
            for j, input_ids_seq in enumerate(input_ids):
                output_ids = generated_ids[j][len(input_ids_seq):]
                print(len(output_ids), max_new_tokens)
                # output_ids = generated_ids[j]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                # generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)
                generated_texts.append(generated_text.strip())
            
            # 提取生成文本的隐藏状态（只提取新生成的部分）
            # for j, (full_ids, input_ids_seq) in enumerate(zip(generated_ids, input_ids)):
            #     # 只取新生成的部分
            #     new_ids = full_ids[len(input_ids_seq):]
            #     # 为了避免太长，我们只取前512个token
            #     if len(new_ids) > 512:
            #         new_ids = new_ids[:512]
                
            #     # 为新生成的部分创建attention mask
            #     new_attention_mask = torch.ones_like(new_ids)
                
            #     # 添加batch维度
            #     new_ids_batch = new_ids.unsqueeze(0)
            #     new_attention_mask_batch = new_attention_mask.unsqueeze(0)
                
            #     # 获取隐藏状态
            #     outputs = model(
            #         input_ids=new_ids_batch,
            #         attention_mask=new_attention_mask_batch,
            #         output_hidden_states=True
            #     )
                
            #     # 提取最后一层隐藏状态
            #     last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
                
            #     # 保存到CPU以节省GPU内存
            #     all_hidden_states.append(last_hidden.cpu())
            #     all_attention_masks.append(new_attention_mask_batch.cpu())
    
        if all_hidden_states:
            # 对于隐藏状态，我们需要处理不同序列长度的情况
            # 方法2: 使用填充到最大序列长度
            # max_seq_len = max(h.size(1) for h in all_hidden_states)
            max_seq_len = 512
            hidden_size = all_hidden_states[0].size(2)
            
            padded_hidden_states = []
            padded_attention_masks = []
            
            for h, mask in zip(all_hidden_states, all_attention_masks):
                current_len = h.size(1)
                batch_size = h.size(0)
                
                if current_len < max_seq_len:
                    # 填充到最大长度
                    pad_len = max_seq_len - current_len
                    h_padded = torch.cat([h, torch.zeros(batch_size, pad_len, hidden_size, dtype=h.dtype)], dim=1)
                    mask_padded = torch.cat([mask, torch.zeros(batch_size, pad_len, dtype=mask.dtype)], dim=1)
                else:
                    h_padded = h
                    mask_padded = mask
                    
                padded_hidden_states.append(h_padded)
                padded_attention_masks.append(mask_padded)
            
            # 合并所有隐藏状态
            hidden_states = torch.cat(padded_hidden_states, dim=0)
            attention_masks = torch.cat(padded_attention_masks, dim=0)
            return generated_texts, hidden_states, attention_masks
        else:
            return generated_texts, None, None

def main():
    parser = argparse.ArgumentParser(description='Generate Qwen texts and extract embeddings for FNSPID dataset')
    parser.add_argument('--data-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_camf', 
                       help='Path to FNSPID dataset')
    parser.add_argument('--output-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_gen_new_total_thinking', 
                       help='Output path for generated texts and embeddings')
    parser.add_argument('--model-path', type=str, 
                       default='/home/llh/MMTSF/MMTSF_LIB/pretrain_model/EmbeddingModel/Qwen3-4B-Instruct-2507',
                       help='Path to Qwen3-8B model')
    parser.add_argument('--splits', nargs='+', default=['train'], 
                       help='Dataset splits to process')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for generation and encoding')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--max-new-tokens', type=int, default=1268,
                       help='Maximum new tokens to generate')
    # parser.add_argument('--prompt-template', type=str, 
    #                default="You are a financial analyst focusing on {company_name} company. Analyze the company's daily stock price {historical_data} (one day's data corresponds to one Step), the historical mean value {mean_value} over the last week, and the relevant news {news}. For each day (Step1–Step5), determine one analysis sentence, a trend direction, and a trend strength, then conclude the global trend.\
    #                         Follow these core rules strictly:\
    #                         1. Trend direction for Step1–Step5:\
    #                         - For each Step i, compare the corresponding daily value hisdata_i in {historical_data} with {mean_value}.\
    #                         - If hisdata_i > {mean_value}, the trend MUST be Rising.\
    #                         - If hisdata_i < {mean_value}, the trend MUST be Falling.\
    #                         - If hisdata_i = {mean_value} completely, the trend MUST be Rising.\
    #                         - The trend field may ONLY be exactly one of these two words: Rising or Falling.\
    #                         2. Trend strength for Step1–Step5:\
    #                         - Determine strength based on the magnitude of hisdata_i - {mean_value}.\
    #                         - The strength field MUST be exactly one of the following words which 'Emerging' is the weakest and 'Dominant' is the strongest to describe the five degrees strength of the trend:\
    #                             Emerging, Moderate, Significant, Prominent, Dominant.\
    #                         3. Global trend:\
    #                         - Infer a predicted future mean value from {historical_data}, {mean_value}, and {news}.\
    #                         - If predicted future mean > {mean_value}, output Global trend: Rising.\
    #                         - If predicted future mean < {mean_value}, output Global trend: Falling.\
    #                         - The global trend may ONLY be Rising or Falling.\
    #                         OUTPUT FORMAT (MUST be followed exactly):\
    #                         - You MUST output EXACTLY SIX LINES, no more and no fewer.\
    #                         - Lines 1–5 MUST each follow this template\
    #                         Step1: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
    #                         Step2: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
    #                         Step3: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
    #                         Step4: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
    #                         Step5: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
    #                         - Line 6 MUST be EXACTLY one of these two lines:\
    #                         Global trend: Rising\
    #                         Global trend: Falling\
    #                         ADDITIONAL FORMATTING CONSTRAINTS (ALL are mandatory):\
    #                         - Do NOT add any extra text before, between, or after these six lines.\
    #                         - Do NOT add bullet points, numbering, Markdown, quotes, explanations, or headings.\
    #                         - Do NOT insert any line breaks inside a single Step line; each Step (Step1–Step5) MUST be contained on exactly one line.\
    #                         - Do NOT change the labels Step1, Step2, Step3, Step4, Step5, Global trend.\
    #                         - Do NOT change the sub-labels analysis:, strength:, trend: and do NOT change their order.\
    #                         - Do NOT use any words for strength other than: Emerging, Moderate, Significant, Prominent, Dominant.\
    #                         - Do NOT use any words for trend other than: Rising, Falling.\
    #                         - Do NOT use semicolons (;) anywhere in the output.\
    #                         - Do NOT add additional fields or extra punctuation beyond commas, colons, and the final newline at the end of each line.\
    #                         - Do NOT repeat, restate, or paraphrase the task description, the rules, the input data, or the few-shot examples in your output.\
    #                         INTERNAL REASONING CONSTRAINTS:\
    #                         - You may reason step by step INTERNALLY, but you MUST NOT output your reasoning process.\
    #                         - Keep any hidden reasoning as short as possible (no more than 300 tokens), and output ONLY the final six lines in the required format.\
    #                         Here are some few-shot examples (follow their style and format, but do not copy text):\
    #                         {few_shots}\
    #                         Now, based on the current input, produce ONLY the six lines of final output in the exact required format."
    #                         ,
    #                help='Prompt template for text generation2')
    parser.add_argument('--prompt-template', type=str, 
                   default="You are a financial analyst focusing on {company_name} company. Analyze the company's daily stock price {historical_data} (one day's data corresponds to one Step), historical mean value {mean_value} last week, and relevant news {news} to determine each Step's analysis, trend direction, and trend strength, then conclude the global trend. Follow these core rules strictly:\
                            Trend direction for Step1-Step5: Compare the daily historical data {historical_data} of the corresponding Step with the historical mean value {mean_value}. Choose Rising if hisdata > mean_value, Falling if hisdata < mean_value.\
                            Trend strength for Step1-Step5: Determine based on the magnitude of the difference between the daily hisdata and mean_value. Select only from these five words: Emerging, Moderate, Significant, Prominent, Dominant.\
                            Global trend: Compare the predicted future mean value (derived from comprehensive analysis of daily data, mean_value, and news) with the historical mean value {mean_value}. Choose only Rising if future mean > mean_value or Falling if future mean < mean_value.\
                            Output only the content in the following specified format without adding any additional information:Step1: Analysis:analysis,strength:strength,trend:trend\nStep2: Analysis:analysis,strength:strength,trend:trend\nStep3: Analysis:analysis,strength:strength,trend:trend\nStep4: Analysis:analysis,strength:strength,trend:trend\nStep5: Analysis:analysis,strength:strength,trend:trend\nGlobal trend:Global trend.\
                            ",
                   help='Prompt template for text generation2')
    
    # parser.add_argument('--prompt-template', type=str, 
    #                default="You are a financial analyst focusing on {company_name} company. Analyze the company's stock price {historical_data} last weekand news {news} to provide a analyse trend and its strength,The trend strength should be selected from these five words:Emerging,Moderate,Significant,Prominent,Dominant, and the trend direction should be chosen from the three words: Rising, Stable, and Falling.Your output should follow the format below: Day 1: Analysis,strength trend; Day 2: Analysis, strength trend; Day 3: Analysis, strength trend;Day 4: Analysis, strength trend; Day 5: Analysis, strength trend.Output only the content in the specified format without adding any additional information.Here is two few_shots:{few_shots}",
    #                help='Prompt template for text generation1')
    # parser.add_argument('--prompt-template', type=str, 
    #                default="You are a financial analyst focusing on {company_name} company. Analyze the company's stock price {historical_data} last weekand news {news} to provide the trend and its strength everyday,The trend strength should be selected from these five words:Emerging,Moderate,Significant,Prominent,Dominant, and the trend direction should be chosen from the three words: Rising, Stable, and Falling.Your output should follow the format below:Day 1: strength trend;Day 2: strength trend;Day 3: strength trend;Day 4: strength trend;Day 5: strength trend.Output only the content in the specified format without adding any additional information.Here is three few_shots:{few_shots}",
    #                help='Prompt template for text generation2')
    parser.add_argument('--test-limit', type=int, default=32,
                   help='限制处理的样本数量，用于测试 (默认: None，处理所有样本)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for parallel processing')
    parser.add_argument('--few-shots', type=str, default="src/generate_qwen_embedding/few_shots_new_pro.txt",
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
        os.environ.setdefault('MASTER_PORT', '29503')  # 使用不同端口避免冲突
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in gpu_ids)
        
        # 准备参数
        base_args = vars(args)
        base_args['gpu_ids'] = gpu_ids
        
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

    # python multi-gpu.py --gpus 2,3,4,5,6,7 
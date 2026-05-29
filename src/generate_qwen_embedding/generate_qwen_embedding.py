# src/generate_qwen_embedding/generate_qwen_embedding.py
import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import gc
import accelerate
import spacy
from collections import Counter
import re
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy

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

def _distributed_embedding_worker(local_rank: int, base_args: dict, all_splits: list):
    """
    分布式工作进程，每个GPU一个进程
    """
    # 设置当前进程的GPU设备
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式环境
    world_size = len(base_args.get('gpu_ids', [0]))
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    
    # 为当前进程分配数据
    args = argparse.Namespace(**base_args)
    args.gpu_id = local_rank
    
    # 根据rank分配数据分割（循环分配）
    assigned_splits = all_splits[local_rank::world_size]
    
    if assigned_splits:
        print(f"GPU {local_rank}: Processing splits {assigned_splits}")
        # 在当前GPU上处理分配的数据
        _process_splits_on_gpu(args, assigned_splits)
    
    dist.destroy_process_group()

def _process_splits_on_gpu(args, splits):
    """
    在指定GPU上处理数据分割
    """
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"GPU {args.gpu_id}: Using device: {device}")
    
    # 加载模型和分词器
    print(f"GPU {args.gpu_id}: Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,  trust_remote_code=True , local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map={"": device} if device.type == 'cuda' else None,
        trust_remote_code=True,
        local_files_only=True,

    )
    model = model.to(device)
    model.eval()
    
    # 处理分配给当前GPU的数据分割
    for split in splits:
        print(f"GPU {args.gpu_id}: Processing {split} split...")
        # 处理逻辑与原来相同
        _process_single_split(model, tokenizer, args, split, device)
    
    # 清理
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

def load_fnspid_data(data_path, split):
    """加载FNSPID数据集"""
    file_path = os.path.join(data_path, f"{split}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 

def extract_trend_from_news(news_text):
    """
    从生成的新闻文本中提取趋势词(Rising/Falling)
    """
    # 定义趋势关键词
    trend_keywords = ['Rising', 'Falling']
    
    # 直接匹配单个词的情况
    if news_text.strip() in trend_keywords:
        return news_text.strip()
    
    # 尝试从复杂文本中提取趋势词
    for keyword in trend_keywords:
        if keyword in news_text:
            return keyword
    
    # 如果找不到趋势词，默认返回Stable
    return 'Rising'


STEP_LINE_PATTERN = re.compile(
    r"(?:Step|Day)\s*(\d)\s*:.*?strength:\s*([A-Za-z]+)\s*,\s*trend:\s*([A-Za-z]+)",
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
        step_idx = int(match.group(1)) - 1
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

def _process_single_split(model, tokenizer, args, split, device):
    """
    处理单个数据分割
    """
    # 加载原始数据
    original_data = load_fnspid_data(args.data_path, split)
    if args.test_limit is not None:
        original_data = original_data[:args.test_limit]
        print(f"GPU {args.gpu_id}: 限制处理前 {args.test_limit} 条记录用于测试")
    few_shots= args.few_shots
    # 构建prompt
    prompts = []
    for item in original_data:
        historical_data = item.get('historical_data', '')
        news = item.get('news', '')
        company_name=extract_company_names(news)
        mean_value = sum([float(x.strip()) for x in historical_data.split(',')])/5
        prompt = args.prompt_template.format(historical_data=historical_data, news=news,company_name=company_name,few_shots=few_shots,mean_value=mean_value)
        prompts.append(prompt)

    print(f"GPU {args.gpu_id}: Generating texts and extracting embeddings for {split}...")
    generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device
    )
    
    # 创建新数据集（包含原始数据和生成的文本）
    new_data = []
    for i, (original_item, generated_text) in enumerate(zip(original_data, generated_texts)):
        new_item = original_item.copy()
        new_item['news'] = generated_text
        strengths, trends, global_trend = extract_structured_trend_fields(generated_text)
        new_item['step_strengths'] = strengths
        new_item['step_trends'] = trends
        new_item['global_trend'] = global_trend
        # new_item['news'] = extract_trend_from_news(generated_text)
        new_data.append(new_item)
    
    # 保存新数据集（每个GPU保存自己的部分）
    output_file = os.path.join(args.output_path, f"{split}_gpu{args.gpu_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"GPU {args.gpu_id}: Saved generated {split} data to {output_file}")
    
    # 保存嵌入（每个GPU保存自己的部分）
    if hidden_states is not None:
        split_file = os.path.join(args.output_path, f'{split}_news_gpu{args.gpu_id}.pt')
        embeddings_dict = {
            f'{split}_news': hidden_states,
            f'{split}_masks': attention_masks,
            f'{split}_texts': generated_texts
        }
        torch.save(embeddings_dict, split_file)
        print(f"GPU {args.gpu_id}: Saved {split} embeddings to {split_file}")


STEP_LINE_PATTERN = re.compile(
    r"(?:Step|Day)\s*(\d)\s*:.*?strength:\s*([A-Za-z]+)\s*,\s*trend:\s*([A-Za-z]+)",
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
    "falling": -1,
}


def extract_structured_trend_fields(news_text, num_steps=5):
    """解析生成文本并返回数值化的强度、趋势以及全局趋势"""
    strengths = [0] * num_steps  # 0 表示未解析到有效强度
    trends = [0] * num_steps    # 0 表示未解析到有效趋势
    for match in STEP_LINE_PATTERN.finditer(news_text):
        step_idx = int(match.group(1)) - 1
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

def _merge_distributed_results(output_path, splits, gpu_ids):
    for split in splits:
        merged_data = []
        # 遍历所有可能的GPU ID，而不是连续查找
        for gpu_id in gpu_ids:
            partial_file = os.path.join(output_path, f"{split}_gpu{gpu_id}.json")
            if os.path.exists(partial_file):
                with open(partial_file, 'r', encoding='utf-8') as f:
                    partial_data = json.load(f)
                    merged_data.extend(partial_data)
                # 处理完后删除临时文件
                os.remove(partial_file)
        
        # 保存合并后的数据
        output_file = os.path.join(output_path, f"{split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"Merged {split} data: {len(merged_data)} samples")
        # 删除临时文件
        gpu_index = 0
        partial_file = os.path.join(output_path, f"{split}_gpu{gpu_index}.json")
        while os.path.exists(partial_file):
            os.remove(partial_file)
            gpu_index += 1
            partial_file = os.path.join(output_path, f"{split}_gpu{gpu_index}.json")
    
    # 合并嵌入文件
    all_embeddings = {}
    for split in splits:
        gpu_index = 0
        partial_file = os.path.join(output_path, f'{split}_news_gpu{gpu_index}.pt')
        
        split_embeddings = []
        split_masks = []
        split_texts = []
        
        while os.path.exists(partial_file):
            partial_embeddings = torch.load(partial_file)
            if f'{split}_news' in partial_embeddings:
                split_embeddings.append(partial_embeddings[f'{split}_news'])
                split_masks.append(partial_embeddings[f'{split}_masks'])
                split_texts.extend(partial_embeddings[f'{split}_texts'])
            gpu_index += 1
            partial_file = os.path.join(output_path, f'{split}_news_gpu{gpu_index}.pt')
        
        if split_embeddings:
            # 合并张量
            all_embeddings[f'{split}_news'] = torch.cat(split_embeddings, dim=0)
            all_embeddings[f'{split}_masks'] = torch.cat(split_masks, dim=0)
            all_embeddings[f'{split}_texts'] = split_texts
    
    # 保存所有嵌入到一个.pt文件
    embeddings_file = os.path.join(output_path, 'all_embeddings.pt')
    torch.save(all_embeddings, embeddings_file)
    print(f"Saved all embeddings to {embeddings_file}")
    
    # 为每个分割保存单独的嵌入文件
    for split in splits:
        if f'{split}_news' in all_embeddings:
            split_file = os.path.join(output_path, f'{split}_news.pt')
            torch.save(all_embeddings[f'{split}_news'], split_file)
            print(f"Saved {split} embeddings to {split_file}")
    
    # 删除临时嵌入文件
    for split in splits:
        gpu_index = 0
        partial_file = os.path.join(output_path, f'{split}_news_gpu{gpu_index}.pt')
        while os.path.exists(partial_file):
            os.remove(partial_file)
            gpu_index += 1
            partial_file = os.path.join(output_path, f'{split}_news_gpu{gpu_index}.pt')
def generate_text_and_extract_hidden_states(model, tokenizer, prompts, batch_size=10, max_new_tokens=512, device='cuda'):
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
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
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
            for j, input_ids_seq in enumerate(input_ids):
                output_ids = generated_ids[j][len(input_ids_seq):]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())
            
            # 提取生成文本的隐藏状态（只提取新生成的部分）
            for j, (full_ids, input_ids_seq) in enumerate(zip(generated_ids, input_ids)):
    # 只取新生成的部分
                new_ids = full_ids[len(input_ids_seq):]
    # 为了避免太长，我们只取前512个token
                if len(new_ids) > 512:
                    new_ids = new_ids[:512]
                
                # 为新生成的部分创建attention mask
                new_attention_mask = torch.ones_like(new_ids)
                
                # 添加batch维度
                new_ids_batch = new_ids.unsqueeze(0)
                new_attention_mask_batch = new_attention_mask.unsqueeze(0)
                
                # 获取隐藏状态
                outputs = model(
                    input_ids=new_ids_batch,
                    attention_mask=new_attention_mask_batch,
                    output_hidden_states=True
                )
                
                # 提取最后一层隐藏状态
                last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
                
                # 保存到CPU以节省GPU内存
                all_hidden_states.append(last_hidden.cpu())
                all_attention_masks.append(new_attention_mask_batch.cpu())
    
        if all_hidden_states:
            # 对于隐藏状态，我们需要处理不同序列长度的情况
            # 方法2: 使用填充到最大序列长度
            max_seq_len = max(h.size(1) for h in all_hidden_states)
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


def _process_splits_on_single_gpu(args):
    """
    单GPU处理逻辑（原有代码）
    """
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和分词器
    print("Loading Qwen3-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True,
            local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map={"": device} if device.type == 'cuda' else None,
        trust_remote_code=True,
        local_files_only=True,
      
    )
    model = model.to(device)
    model.eval()
    
    # 存储所有分割的数据和嵌入
    all_embeddings = {}
    
    # 处理每个数据集分割
    for split in args.splits:
        _process_single_split(model, tokenizer, args, split, device)
        # print(f"Processing {split} split...")
        
        # # 加载原始数据
        # original_data = load_fnspid_data(args.data_path, split)
        # if args.test_limit is not None:
        #     original_data = original_data[:args.test_limit]
        #     print(f"限制处理前 {args.test_limit} 条记录用于测试")
        
        # # 构建prompt
        # prompts = []
        # for item in original_data:
        #     historical_data = item.get('historical_data', '')
        #     news = item.get('news', '')
        #     company_name = extract_company_names(news)
        #     prompt = args.prompt_template.format(historical_data=historical_data, news=news)
        #     prompts.append(prompt)

        # print(f"Generating texts and extracting embeddings for {split}...")
        # generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states(
        #     model, tokenizer, prompts, args.batch_size, args.max_new_tokens, device
        # )
        
        # # 创建新数据集（包含原始数据和生成的文本）
        # new_data = []
        # for i, (original_item, generated_text) in enumerate(zip(original_data, generated_texts)):
        #     new_item = original_item.copy()
        #     new_item['news'] = generated_text
        #     new_data.append(new_item)
        
        # # 保存新数据集
        # output_file = os.path.join(args.output_path, f"{split}.json")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(new_data, f, ensure_ascii=False, indent=2)
        # print(f"Saved generated {split} data to {output_file}")
        
        # # 保存嵌入
        # if hidden_states is not None:
        #     all_embeddings[f'{split}_news'] = hidden_states
        #     all_embeddings[f'{split}_masks'] = attention_masks
        #     all_embeddings[f'{split}_texts'] = generated_texts
        #     print(f"Generated embeddings for {split}: {hidden_states.shape}")
        # else:
        #     print(f"Failed to generate embeddings for {split}")
    
    # 保存所有嵌入到一个.pt文件
    embeddings_file = os.path.join(args.output_path, 'all_embeddings.pt')
    torch.save(all_embeddings, embeddings_file)
    print(f"Saved all embeddings to {embeddings_file}")
    
    # 为每个分割保存单独的嵌入文件
    for split in args.splits:
        if f'{split}_news' in all_embeddings:
            split_file = os.path.join(args.output_path, f'{split}_news.pt')
            torch.save(all_embeddings[f'{split}_news'], split_file)
            print(f"Saved {split} embeddings to {split_file}")
    
    # 清理
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
def main():
    parser = argparse.ArgumentParser(description='Generate Qwen texts and extract embeddings for FNSPID dataset')
    parser.add_argument('--data-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_camf', 
                       help='Path to FNSPID dataset')
    parser.add_argument('--output-path', type=str, default='/home/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/ver_gen_new_total', 
                       help='Output path for generated texts and embeddings')
    parser.add_argument('--model-path', type=str, 
                       default='/home/llh/MMTSF/MMTSF_LIB/pretrain_model/EmbeddingModel/Qwen3-4B',
                       help='Path to Qwen model')
    parser.add_argument('--splits', nargs='+', default=['train', 'vali', 'test'], 
                       help='Dataset splits to process')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for generation and encoding')
    parser.add_argument('--gpu-id', type=int, default=7,
                       help='GPU ID to use')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum new tokens to generate')
    # parser.add_argument('--prompt-template', type=str, 
    #                default="You are a financial analyst focusing on {company_name} company. Analyze the company's stock price {historical_data} last weekand news {news} to provide a analyse trend and its strength,The trend strength should be selected from these five words:Emerging,Moderate,Significant,Prominent,Dominant, and the trend direction should be chosen from the three words: Rising, Stable, and Falling.Your output should follow the format below:Day 1: Analysis,strength trend;Day 2: Analysis, strength trend;Day 3: Analysis, strength trend;Day 4: Analysis, strength trend;Day 5: Analysis, strength trend.here is two few_shots:{few_shots}",
    #                help='Prompt template for text generation')
    parser.add_argument('--prompt-template', type=str, 
                   default="You are a financial analyst focusing on {company_name} company. Analyze the company's daily stock price {historical_data} (one day's data corresponds to one Step), the historical mean value {mean_value} over the last week, and the relevant news {news}. For each day (Step1-Step5), determine an analysis sentence, a trend direction, and a trend strength, then conclude the global trend.\
                            Follow these core rules strictly:\
                            1. Trend direction for Step1-Step5:\
                            - For each Step i, compare the corresponding daily value hisdata_i in {historical_data} with {mean_value}.\
                            - If hisdata_i > {mean_value}, the trend MUST be Rising.\
                            - If hisdata_i < {mean_value}, the trend MUST be Falling.\
                            - If hisdata_i = {mean_value} completely, the trend MUST be Rising.\
                            - The trend field may ONLY be exactly one of these two words: Rising or Falling.\
                            2. Trend strength for Step1-Step5:\
                            - Determine strength based on the magnitude of hisdata_i - {mean_value}.\
                            - The strength field MUST be EXACTLY one of the following words (no other words allowed):\
                                Emerging, Moderate, Significant, Prominent, Dominant.\
                            3. Global trend:\
                            - Infer a predicted future mean value from {historical_data}, {mean_value}, and {news}.\
                            - If predicted future mean > {mean_value}, output Global trend: Rising.\
                            - If predicted future mean < {mean_value}, output Global trend: Falling.\
                            - The global trend may ONLY be Rising or Falling.\
                            OUTPUT FORMAT (MUST be followed exactly):\
                            - You MUST output EXACTLY SIX LINES, no more and no fewer.\
                            - Lines 1-5 MUST each follow this template (keep all colons, commas, spaces, and order exactly):\
                            Step1: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
                            Step2: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
                            Step3: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
                            Step4: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
                            Step5: analysis: <one English sentence>, strength: <one of Emerging/Moderate/Significant/Prominent/Dominant>, trend: <Rising or Falling>\
                            - Line 6 MUST be EXACTLY one of these two lines:\
                            Global trend: Rising\
                            Global trend: Falling\
                            ADDITIONAL FORMATTING CONSTRAINTS (ALL are mandatory):\
                            - Do NOT add any extra text before, between, or after these six lines.\
                            - Do NOT add bullet points, numbering, Markdown, quotes, explanations, or headings.\
                            - Do NOT insert any line breaks inside a single Step line; each Step (Step1-Step5) MUST be contained on exactly one line.\
                            - Do NOT change the labels Step1, Step2, Step3, Step4, Step5, Global trend.\
                            - Do NOT change the sub-labels analysis:, strength:, trend: and do NOT change their order.\
                            - Do NOT use any words for strength other than: Emerging, Moderate, Significant, Prominent, Dominant.\
                            - Do NOT use any words for trend other than: Rising, Falling.\
                            - Do NOT use semicolons (;) anywhere in the output.\
                            - Do NOT add additional fields or extra punctuation beyond commas, colons, and the final newline at the end of each line.\
                            - Do NOT repeat, paraphrase, or otherwise mention this prompt in your output.\
                            - Do NOT write your reasoning process,you may think step by step privately, but only output the final answer.Just output like the few_shot's.",
                   help='Prompt template for text generation2')
    parser.add_argument('--test-limit', type=int, default=None,
                   help='限制处理的样本数量，用于测试 (默认: None,处理所有样本)')
    parser.add_argument('--gpus', type=str, help='Comma separated GPU ids for parallel processing')
    parser.add_argument('--few-shots', type=str, default="src/generate_qwen_embedding/few_shots_new_pro.txt",
                   help=' few_shots')
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
  # 处理GPU参数
    if args.gpus:
        gpu_ids = _parse_gpu_list(args.gpus)
    else:
        gpu_ids = [args.gpu_id]
    args.few_shots = open(args.few_shots, 'r', encoding='utf-8').read()    
    if len(gpu_ids) > 1:
        # 设置分布式环境变量
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29501')  # 使用不同端口避免冲突
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in gpu_ids)
        
        # 准备参数
        base_args = vars(args)
        base_args['gpu_ids'] = gpu_ids
        
        # 数据分割列表
        data_splits = args.splits
        
        # 启动分布式处理
        mp.spawn(
            _distributed_embedding_worker,
            nprocs=len(gpu_ids),
            args=(base_args, data_splits),
            join=True,
        )

        _merge_distributed_results(args.output_path, args.splits,gpu_ids)

    else:
    # 设置设备
        _process_splits_on_single_gpu(args)

if __name__ == "__main__":
    main()





# Your output should follow the format below:Day 1: Analysis,strength trend;Day 2: Analysis, strength trend;Day 3: Analysis, strength trend;Day 4: Analysis, strength trend;Day 5: Analysis, strength trend.
# parser.add_argument('--prompt-template', type=str, 
#                    default="You are a financial analyst focusing on {company_name}. Analyze the company's stock price {historical_data} last weekand news {news}to provide a analyse trend and its strength,The trend strength should be selected from these five words:Emerging,Moderate,Significant,Prominent,Dominant, and the trend direction should be chosen from the three words: Rising, Stable, and Falling.Your output should follow the format below:Day 1: Analysis,strength trend;Day 2: Analysis, strength trend;Day 3: Analysis, strength trend;Day 4: Analysis, strength trend;Day 5: Analysis, strength trend.",
#                    help='Prompt template for text generation')


# def _process_single_split(model, tokenizer, args, split, device):
#     """
#     处理单个数据分割
#     """
#     # 加载原始数据
#     original_data = load_fnspid_data(args['data_path'], split)
#     if args.get('test_limit') is not None:
#         original_data = original_data[:args['test_limit']]
#         print(f"限制处理前 {args['test_limit']} 条记录用于测试")
    
#     # 构建prompt
#     prompts = []
#     for item in original_data:
#         historical_data = item.get('historical_data', '')
#         news = item.get('news', '')
#         company_name = extract_company_names(news)        
#         prompt = args['prompt_template'].format(historical_data=historical_data, news=news)
#         prompts.append(prompt)

#     print(f"Generating texts and extracting embeddings for {split}...")
#     generated_texts, hidden_states, attention_masks = generate_text_and_extract_hidden_states(
#         model, tokenizer, prompts, args['batch_size'], args['max_new_tokens'], device
#     )
    
#     # 创建新数据集（包含原始数据和生成的文本）
#     new_data = []
#     for i, (original_item, generated_text) in enumerate(zip(original_data, generated_texts)):
#         new_item = original_item.copy()
#         new_item['news'] = generated_text
#         new_data.append(new_item)
    
#     # 保存新数据集
#     output_file = os.path.join(args['output_path'], f"{split}.json")
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(new_data, f, ensure_ascii=False, indent=2)
#     print(f"Saved generated {split} data to {output_file}")
    
#     # 保存嵌入
#     if hidden_states is not None:
#         # 为每个分割保存单独的嵌入文件
#         split_file = os.path.join(args['output_path'], f'{split}_news.pt')
#         embeddings_dict = {
#             f'{split}_news': hidden_states,
#             f'{split}_masks': attention_masks,
#             f'{split}_texts': generated_texts
#         }
#         torch.save(embeddings_dict, split_file)
#         print(f"Saved {split} embeddings to {split_file}")

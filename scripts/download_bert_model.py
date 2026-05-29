#!/usr/bin/env python3
"""
下载BERT模型到本地指定路径
用法: python download_bert_model.py [--model-name bert-base-uncased] [--output-dir /ssd/hf_home/models]
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import BertModel, BertTokenizer, BertConfig
    from transformers.file_utils import TRANSFORMERS_CACHE
except ImportError:
    print("错误: 请先安装transformers库")
    print("pip install transformers")
    sys.exit(1)


def download_bert_model(model_name='bert-base-uncased', output_dir='/ssd/hf_home/models'):
    """
    下载BERT模型到指定目录
    
    Args:
        model_name: BERT模型名称（如 'bert-base-uncased'）
        output_dir: 输出目录路径
    """
    print(f"开始下载BERT模型: {model_name}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n模型将保存到: {output_path}")
    
    try:
        # 下载模型和tokenizer
        print("\n正在下载模型...")
        model = BertModel.from_pretrained(model_name)
        print("✓ 模型下载完成")
        
        print("\n正在下载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer下载完成")
        
        print("\n正在下载config...")
        config = BertConfig.from_pretrained(model_name)
        print("✓ Config下载完成")
        
        # 保存到指定路径
        print(f"\n正在保存到 {output_path}...")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        config.save_pretrained(str(output_path))
        
        print(f"\n✓✓✓ 成功！BERT模型已保存到: {output_path}")
        print(f"\n使用方法:")
        print(f"在配置文件中设置:")
        print(f"  bert_model_path: {output_path}")
        print(f"\n或者在代码中使用:")
        print(f"  model = BertModel.from_pretrained('{output_path}', local_files_only=True)")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 如果网络受限，可以使用代理:")
        print("   export http_proxy=your_proxy")
        print("   export https_proxy=your_proxy")
        print("3. 或者手动从Hugging Face下载模型文件")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='下载BERT模型到本地')
    parser.add_argument('--model-name', default='bert-base-uncased',
                       help='BERT模型名称 (默认: bert-base-uncased)')
    parser.add_argument('--output-dir', default='/public/home/maoyaoxin/llh/MMTSF/hf_home',
                       help='输出目录 (默认: /ssd/hf_home/models)')
    args = parser.parse_args()
    
    download_bert_model(args.model_name, args.output_dir)


if __name__ == "__main__":
    main()


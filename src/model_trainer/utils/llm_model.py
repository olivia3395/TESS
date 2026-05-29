from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Tuple
import torch.nn as nn
from transformers import GemmaConfig



class LLMInitializer:
    @staticmethod
    def init_llm_model(config: dict) -> Tuple[nn.Module, nn.Module]:
        """
        大模型初始化入口函数
        参数:
            config: 包含以下键的配置字典
                - llm_model: 模型标识符 (llama-7b/llama-8b等)
                - [model_type]_path: 对应的本地模型路径
        返回:
            tuple: (初始化的模型, tokenizer)
        """
        model_type = config["llm_model"].lower()
        MODEL_INIT_MAP = {
            "llama-7b": LLMInitializer._init_llama_7b,
            "llama-8b": LLMInitializer._init_llama_8b,
            "deepseek-7b": LLMInitializer._init_ds_llama_7b,
            "deepseek-r1-8b": LLMInitializer._init_ds_r1_llama_8b,
            "gemma": LLMInitializer._init_gemma
        }
        
        if model_type not in MODEL_INIT_MAP:
            raise ValueError(f"不支持的模型类型: {model_type}. 可选: {list(MODEL_INIT_MAP.keys())}")
            
        return MODEL_INIT_MAP[model_type](config)

    @staticmethod
    def _init_llama_7b(config: dict) -> Tuple[nn.Module, nn.Module]:
        return LLMInitializer._generic_init(config, "llama-7b")

    @staticmethod
    def _init_llama_8b(config: dict) -> Tuple[nn.Module, nn.Module]:
        return LLMInitializer._generic_init(config, "llama-8b")

    @staticmethod
    def _init_ds_llama_7b(config: dict) -> Tuple[nn.Module, nn.Module]:
        return LLMInitializer._generic_init(config, "deepseek-7b")

    @staticmethod
    def _init_ds_r1_llama_8b(config: dict) -> Tuple[nn.Module, nn.Module]:
        return LLMInitializer._generic_init(config, "deepseek-r1-8b")

    @staticmethod
    def _init_gemma(config: dict) -> Tuple[nn.Module, nn.Module]:
        model_key = 'gemma'
        
        path_key = f"{model_key.replace('-', '_')}_path"
        model_path = config[path_key]
        model_config = GemmaConfig.from_pretrained(model_path)
        model_config.output_attentions = True
        model_config.output_hidden_states = True

    @staticmethod
    def _generic_init(config: dict, model_key: str) -> Tuple[nn.Module, nn.Module]:
        """通用初始化逻辑"""
        import ipdb;ipdb.set_trace()
        # 转换模型标识符格式（llama-7b -> llama_7b_path）
        path_key = f"{model_key.replace('-', '_')}_path"
        import ipdb; ipdb.set_trace()
    
        model_path = config[path_key]
        
        if not model_path:
            raise ValueError(f"配置中缺少{path_key}路径，请检查config设置")

        try:
            # 加载模型配置
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            model_config.output_attentions = True
            model_config.output_hidden_states = True

            # 加载模型和tokenizer
            model = AutoModel.from_pretrained(
                model_path,
                config=model_config,
                trust_remote_code=True,
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            # 验证模型完整性
            if hasattr(model, "layers"):
                assert len(model.layers) > 0, "模型层数异常"
            elif hasattr(model, "transformer"):
                assert len(model.transformer.h) > 0, "Transformer层数异常"

            return model, tokenizer

        except OSError as e:
            raise RuntimeError(f"本地模型加载失败，请检查路径: {model_path}") from e
        except Exception as e:
            raise RuntimeError(f"模型初始化异常: {str(e)}") from e

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_trainer.layers.Embed import DataEmbedding
from model_trainer.layers.Causal import TempDisentangler
from model_trainer.layers.StandardNorm import Normalize

class MultiModal_Baseline_Token_Level(nn.Module):
    def __init__(self, configs):
        super(MultiModal_Baseline_Token_Level, self).__init__()
        ## 在configs/model里面找到这个模型的yaml, 可以修改配置
        ## configs/dataset/FNSPID.yaml. 以及index.yaml里面可以找到数据集相关的配置
        ## 运行的时候 在MMTSF_LIB这个目录,  PYTHONPATH=src python src/model_trainer/main.py --model MultiModal_Baseline_Token_Level --dataset FNSPID --gpu 0 就能运行起来, 即只需要输入数据集, 模型名称, GPU就可以, 当GPU指定多个的时候,例如 0,1,2,3 会用多GPU训练
        ## 这里采取的融合方式是token level的, 相比于之前embedding level的, 可解释性更强一些 
        ## chat_template可以补充, 在configs/model/MultiModal_Baseline_Token_Level 其实就是prompt,可以控制LLM的生成 
        self.pred_len = configs["pred_len"]
        self.seq_len = configs["seq_len"]
        self.embedding_size = configs["embedding_size"]
        self.hid_dim = configs['embedding_size']
        self.mm_emb_dim = configs['embedding_size']
        self.dropout = configs['dropout'] 

        self.sudden_drop = configs['sudden_drop']
        self.fuse_drop = configs['fuse_drop']
        self.depth = configs['depth']

        self.e_layers = configs["e_layers"]
        self.enc_in = configs["enc_in"]
        self.embed = configs["embed"]


        self.enc_embedding = DataEmbedding(self.enc_in, self.hid_dim, self.embed, self.dropout)
      
        t_kernels = [2**i for i in range(int(math.log2(self.seq_len//2)))]

        self.temporal = TempDisentangler(
            input_dims=self.hid_dim,
            output_dims=self.hid_dim * 2,
            kernels=t_kernels,
            length=self.seq_len,
            hidden_dims=self.hid_dim,
            depth=self.depth,
            dropout=self.dropout
        )



        # Flatten the h_ts
        self.time_to_mm = nn.Linear(self.hid_dim, self.mm_emb_dim)

        # === Text encoder ===
        self.text_model_path = configs.get('text_model_path') or configs.get('llama_8b_path')
        if not self.text_model_path:
            raise ValueError("配置缺少 text_model_path 或 llama_8b_path，无法初始化文本编码器")
        self.text_max_length = int(configs.get('text_max_length', 256))
        self.text_finetune = bool(configs.get('text_finetune', False))
        self.text_dtype = configs.get('text_model_dtype', 'float16')
        self.text_use_chat_template = bool(configs.get('text_chat_template', True))
        self.text_enable_thinking = bool(configs.get('text_enable_thinking', False))
        self.text_generation_prompt = bool(configs.get('text_add_generation_prompt', False))
        self.use_llm_encoder = bool(configs.get('use_llm_encoder', True))
        self._init_text_encoder()

        self.text_proj = nn.Linear(self.text_hidden_dim, self.mm_emb_dim)

        # === Cross-modal fusion ===
        self.cross_attn_heads = int(configs.get('cross_heads', 8))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.mm_emb_dim,
            num_heads=self.cross_attn_heads,
            dropout=self.fuse_drop,
            batch_first=True,
        )
        self.cross_norm1 = nn.LayerNorm(self.mm_emb_dim)
        self.cross_norm2 = nn.LayerNorm(self.mm_emb_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.mm_emb_dim, self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.fuse_drop),
        )

        
        # Decoder for prediction
       
        self.decoder_mlp = nn.Sequential(
                nn.Linear(self.mm_emb_dim , 256),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256,512),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, self.pred_len)
            )

        self.mi_regulization = nn.CrossEntropyLoss()

        if configs['beta1'] is None:
            configs['beta1'] = 0.2
        if configs['beta2'] is None:
            configs['beta2'] = 0.2
        self.beta1 = configs['beta1']
        self.beta2 = configs['beta2']
        self.normalize_layers = Normalize(1, affine=False)
        self.last_attention = None
        self.last_attention_scores = None


    def _init_text_encoder(self) -> None:
        config = AutoConfig.from_pretrained(self.text_model_path, trust_remote_code=True)
        self.text_hidden_dim = config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if pad_token is None:
                pad_token = "<pad>"
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'right'

        text_dtype = torch.float32
        if isinstance(self.text_dtype, str):
            if self.text_dtype.lower() in {'float16', 'fp16', 'half'}:
                text_dtype = torch.float16
            elif self.text_dtype.lower() in {'bfloat16', 'bf16'}:
                text_dtype = torch.bfloat16
        elif isinstance(self.text_dtype, torch.dtype):
            text_dtype = self.text_dtype

        self.text_model = None
        self.token_embedding = None

        if self.use_llm_encoder:
            self.text_model = AutoModelForCausalLM.from_pretrained(
                self.text_model_path,
                torch_dtype=text_dtype,
                trust_remote_code=True,
            )
            if hasattr(self.text_model, "resize_token_embeddings"):
                embedding_matrix = self.text_model.get_input_embeddings()
                if embedding_matrix is not None and embedding_matrix.num_embeddings != len(self.tokenizer):
                    self.text_model.resize_token_embeddings(len(self.tokenizer))
            if not self.text_finetune:
                for param in self.text_model.parameters():
                    param.requires_grad = False
                self.text_model.eval()
        else:
            with torch.no_grad():
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.text_model_path,
                        torch_dtype=text_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map={'': 'cpu'},
                    )
                except (TypeError, ValueError):
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.text_model_path,
                        torch_dtype=text_dtype,
                        trust_remote_code=True,
                    )
                    base_model = base_model.to(torch.device('cpu'))
                embedding_layer = base_model.get_input_embeddings()
                weight = embedding_layer.weight.detach().clone().float()
                del base_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.token_embedding = nn.Embedding.from_pretrained(
                weight,
                freeze=not self.text_finetune,
            )
            if self.text_finetune:
                self.token_embedding.weight.requires_grad = True

   
    def _build_chat_inputs(self, texts: Sequence[str]) -> Sequence[str]:
        if not self.text_use_chat_template:
            return list(texts)
        formatted: list[str] = []
        for text in texts:
            messages = [{"role": "user", "content": text}]
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=self.text_generation_prompt,
                    enable_thinking=self.text_enable_thinking,
                )
            except AttributeError:
                chat_text = text
            formatted.append(chat_text)
        return formatted

    def _encode_text(
        self,
        news: Sequence[str],
        device: torch.device,
        precomputed_hidden: torch.Tensor | None = None,
        precomputed_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if precomputed_hidden is not None:
            token_embeddings = precomputed_hidden.to(device)
            token_embeddings = token_embeddings.to(self.text_proj.weight.dtype)
            if precomputed_mask is not None:
                key_padding_mask = (precomputed_mask == 0).bool().to(device)
            else:
                key_padding_mask = torch.zeros(token_embeddings.size(0), token_embeddings.size(1), dtype=torch.bool, device=device)
            return token_embeddings, key_padding_mask

        if not isinstance(news, (list, tuple)):
            raise TypeError("news 输入应为字符串列表")
        inputs = self._build_chat_inputs(news)
        batch = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt',
        )
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)

        if self.use_llm_encoder:
            model_device = next(self.text_model.parameters()).device
            if model_device != device:
                self.text_model.to(device)

            forward_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'output_hidden_states': True,
                'use_cache': False,
            }
            if self.text_finetune:
                outputs = self.text_model(**forward_kwargs)
            else:
                with torch.no_grad():
                    outputs = self.text_model(**forward_kwargs)
            token_embeddings = outputs.hidden_states[-1]
        else:
            if self.token_embedding.weight.device != device:
                self.token_embedding = self.token_embedding.to(device)
            token_embeddings = self.token_embedding(input_ids)

        token_embeddings = token_embeddings.to(self.text_proj.weight.dtype)
        key_padding_mask = (attention_mask == 0).bool()
        return token_embeddings, key_padding_mask

    def forward(self,x_enc,news,news_hidden=None,news_mask=None,flag='train'):
        """
        时序编码->解耦  获得时序的初始表征 
        """
        """
        x_enc: 原始的时间序列数据 B L Channel_Size 
        x_text:预提取的文本表征
        """

        x_enc = x_enc.unsqueeze(-1)
        self.batch_size = x_enc.size(0)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc) 
        h_env, h_ts = self.temporal(enc_out) 
        
        
       

        ## Dynamic Context Learning 
        time_seq = self.time_to_mm(h_ts)

        text_tokens, key_padding_mask = self._encode_text(news, device=x_enc.device, precomputed_hidden=news_hidden, precomputed_mask=news_mask)
        text_seq = self.text_proj(text_tokens)

        attn_output, attn_weights = self.cross_attn(
            query=time_seq,
            key=text_seq,
            value=text_seq,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        attn_output = self.cross_norm1(attn_output + time_seq)
        attn_output = self.cross_norm2(attn_output + self.cross_ffn(attn_output))

        self.last_attention = attn_weights.detach()
        self.last_attention_scores = None

        fused_summary = attn_output.mean(dim=1)

        dec_out = self.decoder_mlp(fused_summary).unsqueeze(-1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)
        
        return self.dec_out
    
    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss_cons = F.mse_loss(outputs, batch_y)
 

        
        return loss_cons

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_llm_encoder and self.text_model is not None and not self.text_finetune:
            self.text_model.eval()
        return self
    
    def get_model_embeddings(self,*args, **kwargs):
        raise NotImplementedError("暂不支持导出中间嵌入")


        
     
        

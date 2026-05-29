import copy
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model_trainer.layers.Embed import DataEmbedding
from model_trainer.layers.Causal import TempDisentangler
from model_trainer.layers.StandardNorm import Normalize


class MultiModal_Baseline_SelfAttention(nn.Module):
    def __init__(self, configs):
        super().__init__()

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

        t_kernels = [2 ** i for i in range(int(math.log2(self.seq_len // 2)))]

        self.temporal = TempDisentangler(
            input_dims=self.hid_dim,
            output_dims=self.hid_dim * 2,
            kernels=t_kernels,
            length=self.seq_len,
            hidden_dims=self.hid_dim,
            depth=self.depth,
            dropout=self.dropout,
        )

        self.time_to_mm = nn.Linear(self.hid_dim, self.mm_emb_dim)

        self.text_layernorm = nn.LayerNorm(self.mm_emb_dim)
        self.time_layernorm = nn.LayerNorm(self.mm_emb_dim)
        self.text_position_embed = nn.Embedding(self.text_max_length, self.mm_emb_dim)
        self.time_position_embed = nn.Embedding(self.seq_len, self.mm_emb_dim)

        # text encoder options
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

        pre_attn_layers = int(configs.get('pre_attn_layers', 1))
        pre_attn_heads = int(configs.get('pre_attn_heads', 4))
        if pre_attn_heads < 1:
            pre_attn_heads = 1
        if self.mm_emb_dim % pre_attn_heads != 0:
            pre_attn_heads = 1
        if pre_attn_layers > 0:
            text_pre_layer = nn.TransformerEncoderLayer(
                d_model=self.mm_emb_dim,
                nhead=pre_attn_heads,
                dim_feedforward=int(configs.get('pre_attn_ffn_dim', self.mm_emb_dim * 2)),
                dropout=self.fuse_drop,
                batch_first=True,
            )
            time_pre_layer = copy.deepcopy(text_pre_layer)
            self.text_pre_encoder = nn.TransformerEncoder(text_pre_layer, num_layers=pre_attn_layers)
            self.time_pre_encoder = nn.TransformerEncoder(time_pre_layer, num_layers=pre_attn_layers)
        else:
            self.text_pre_encoder = None
            self.time_pre_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.mm_emb_dim,
            nhead=int(configs.get('self_attn_heads', 8)),
            dim_feedforward=int(configs.get('self_attn_ffn_dim', self.mm_emb_dim * 4)),
            dropout=self.fuse_drop,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(configs.get('self_attn_layers', 2)),
        )

        self.post_fusion_ffn = nn.Sequential(
            nn.Linear(self.mm_emb_dim, self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.fuse_drop),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.mm_emb_dim * 2, self.mm_emb_dim),
            nn.Sigmoid(),
        )

        self.text_token_type = nn.Parameter(torch.randn(1, 1, self.mm_emb_dim))
        self.time_token_type = nn.Parameter(torch.randn(1, 1, self.mm_emb_dim))

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.mm_emb_dim, 256),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.pred_len),
        )

        self.mi_regulization = nn.CrossEntropyLoss()

        if configs['beta1'] is None:
            configs['beta1'] = 0.2
        if configs['beta2'] is None:
            configs['beta2'] = 0.2
        self.beta1 = configs['beta1']
        self.beta2 = configs['beta2']
        self.normalize_layers = Normalize(1, affine=False)

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
                embedding_layer = self.text_model.get_input_embeddings()
                if embedding_layer is not None and embedding_layer.num_embeddings != len(self.tokenizer):
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
            self.token_embedding = nn.Embedding.from_pretrained(weight, freeze=not self.text_finetune)
            if self.text_finetune:
                self.token_embedding.weight.requires_grad = True

    def _build_chat_inputs(self, texts: Sequence[str]) -> Sequence[str]:
        if not self.text_use_chat_template:
            return list(texts)
        formatted = []
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
        precomputed_hidden: Optional[torch.Tensor] = None,
        precomputed_mask: Optional[torch.Tensor] = None,
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

    def forward(self, x_enc, news, news_hidden=None, news_mask=None, flag='train'):
        x_enc = x_enc.unsqueeze(-1)
        batch_size = x_enc.size(0)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc)
        _, h_ts = self.temporal(enc_out)

        time_seq = self.time_to_mm(h_ts)

        text_tokens, key_padding_mask = self._encode_text(news, device=x_enc.device, precomputed_hidden=news_hidden, precomputed_mask=news_mask)
        text_seq = self.text_proj(text_tokens)

        text_len = text_seq.size(1)
        time_len = time_seq.size(1)

        text_positions = torch.arange(text_len, device=x_enc.device).unsqueeze(0)
        time_positions = torch.arange(time_len, device=x_enc.device).unsqueeze(0)

        if text_len > self.text_max_length:
            raise ValueError(f"text sequence length {text_len} exceeds configured max length {self.text_max_length}")
        if time_len > self.seq_len:
            raise ValueError(f"time sequence length {time_len} exceeds configured seq_len {self.seq_len}")

        text_seq = text_seq + self.text_position_embed(text_positions)
        time_seq = time_seq + self.time_position_embed(time_positions)

        text_seq = self.text_layernorm(text_seq + self.text_token_type.expand(batch_size, text_len, -1))
        time_seq = self.time_layernorm(time_seq + self.time_token_type.expand(batch_size, time_len, -1))

        if self.text_pre_encoder is not None:
            text_seq = self.text_pre_encoder(text_seq, src_key_padding_mask=key_padding_mask)
        if self.time_pre_encoder is not None:
            time_seq = self.time_pre_encoder(time_seq)

        concat_seq = torch.cat([text_seq, time_seq], dim=1)

        if key_padding_mask is not None:
            pad_text = key_padding_mask
        else:
            pad_text = torch.zeros(text_seq.size()[:-1], dtype=torch.bool, device=x_enc.device)
        pad_time = torch.zeros(time_seq.size()[:-1], dtype=torch.bool, device=x_enc.device)
        src_key_padding_mask = torch.cat([pad_text, pad_time], dim=1)

        encoder_out = self.transformer_encoder(concat_seq, src_key_padding_mask=src_key_padding_mask)

        time_out = encoder_out[:, text_len:, :]
        gate = self.fusion_gate(torch.cat([time_out, time_seq], dim=-1))
        time_out = gate * time_out + (1 - gate) * time_seq
        time_out = self.post_fusion_ffn(time_out)
        fused_summary = time_out.mean(dim=1)

        dec_out = self.decoder_mlp(fused_summary).unsqueeze(-1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)

        return self.dec_out

    def calculate_loss(self, batch_y):
        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)
        loss_cons = F.mse_loss(outputs, batch_y)
        return loss_cons

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_llm_encoder and self.text_model is not None and not self.text_finetune:
            self.text_model.eval()
        return self

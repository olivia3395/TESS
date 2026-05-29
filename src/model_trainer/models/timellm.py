from math import sqrt

import torch
import torch.nn as nn

try:
    from transformers import (
        LlamaConfig,
        LlamaModel,
        LlamaTokenizer,
        GPT2Config,
        GPT2Model,
        GPT2Tokenizer,
        BertConfig,
        BertModel,
        BertTokenizer,
    )
    import transformers
    TRANS_AVAILABLE = True
except Exception:  # transformers not installed or import failed
    TRANS_AVAILABLE = False
    transformers = None
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize

if TRANS_AVAILABLE:
    transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TimeLLM(nn.Module):

    def __init__(self, config):
        super(TimeLLM, self).__init__()
 
        self.pred_len = config["pred_len"]
        self.seq_len = config["seq_len"]
        self.d_ff = config["d_ff"]
        self.top_k = config["top_k"]
        self.d_llm = config["llm_dim"]
        self.patch_len = config["patch_len"]
        self.stride = config["stride"]
        self.llm_model_type = config["llm_model"]
        # Handle llm_layers from config (can be list or int)
        llm_layers_config = config["llm_layers"]
        if isinstance(llm_layers_config, list):
            self.llm_layers = llm_layers_config[0] if llm_layers_config else 32
        else:
            self.llm_layers = llm_layers_config
        # Try to load pretrained LLM; fallback to offline lightweight encoder if unavailable
        self.offline_mode = False
        try:
            if not TRANS_AVAILABLE:
                raise ImportError("transformers not available")
            if self.llm_model_type == 'llama':
                model_name = config["llm_model_name"] if "llm_model_name" in config else "huggyllama/llama-7b"
                print(f"Loading LLaMA model from Hugging Face: {model_name}")
                self.llama_config = LlamaConfig.from_pretrained(model_name)
                self.llama_config.num_hidden_layers = config.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                self.llm_model = LlamaModel.from_pretrained(model_name, trust_remote_code=True, config=self.llama_config)
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            elif self.llm_model_type == 'GPT2':
                model_name = config["llm_model_name"] if "llm_model_name" in config else "gpt2"
                print(f"Loading GPT2 model from Hugging Face: {model_name}")
                self.gpt2_config = GPT2Config.from_pretrained(model_name)
                self.gpt2_config.num_hidden_layers = self.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                self.llm_model = GPT2Model.from_pretrained(model_name, trust_remote_code=True, config=self.gpt2_config)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, trust_remote_code=True)
            elif self.llm_model_type == 'BERT':
                model_name = config["llm_model_name"] if "llm_model_name" in config else "google-bert/bert-base-uncased"
                print(f"Loading BERT model from Hugging Face: {model_name}")
                self.bert_config = BertConfig.from_pretrained(model_name)
                self.bert_config.num_hidden_layers = self.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                self.llm_model = BertModel.from_pretrained(model_name, trust_remote_code=True, config=self.bert_config)
                self.tokenizer = BertTokenizer.from_pretrained(model_name, trust_remote_code=True)
            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
        except Exception as e:
            print(f"[TimeLLM] Offline fallback due to: {e}")
            self.offline_mode = True
            # Minimal tokenizer and embeddings
            self.vocab_size = 8192
            self.simple_tokenizer = SimpleTokenizer(self.vocab_size)
            self.word_embed_layer = nn.Embedding(self.vocab_size, self.d_llm)
            self.word_embeddings = self.word_embed_layer.weight
            # Lightweight encoder in place of LLM
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_llm, nhead=min(4, max(1, self.d_llm // 64)), batch_first=True)
            self.offline_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.offline_proj = nn.Linear(self.d_llm, self.d_ff)
        self.prompt_domain = config["prompt_domain"]
        if self.prompt_domain:
            self.description = config["content"]
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(config["dropout"])

        self.patch_embedding = PatchEmbedding(
            config["d_model"], self.patch_len, self.stride, config["dropout"])

        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # Handle n_heads from config (can be list or int)
        n_heads_config = config["n_heads"]
        if isinstance(n_heads_config, list):
            n_heads = n_heads_config[0] if n_heads_config else 8
        else:
            n_heads = n_heads_config
        self.reprogramming_layer = ReprogrammingLayer(config["d_model"], n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums


        self.output_projection = FlattenHead(1, self.head_nf, self.pred_len,
                                                head_dropout=config["dropout"])
  

        self.normalize_layers = Normalize(1, affine=False)

    def forward(self, x_enc, news=None, flag='train'):
        """
        Forward pass of TimeLLM
        
        Args:
            x_enc: Input time series data (batch_size, seq_len) or (batch_size, seq_len, 1)
            events: Optional list of event dictionaries for each sample in batch
            flag: 'train' or 'test' mode
        
        Returns:
            dec_out: Forecast output (batch_size, pred_len)
        """
        dec_out = self.forecast(x_enc, news)
        return dec_out


    def forecast(self, x_enc, news=None):

      
        x_enc = x_enc.unsqueeze(-1)
         
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]

      
        deterministic_backup = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        medians = torch.median(x_enc, dim=1).values
        torch.use_deterministic_algorithms(deterministic_backup)
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
    
        prompt = []
        
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())

            # Build base prompt with statistics
            prompt_parts = [
                f"<|start_prompt|>Dataset description: {self.description}",
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; ",
                "Input statistics: ",
                f"min value {min_values_str}, ",
                f"max value {max_values_str}, ",
                f"median value {median_values_str}, ",
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, ",
                f"top {self.top_k} lags are : {lags_values_str}"
            ]
            # 使用逐样本的 news 文本（列表中对应 b 的一条）
            news_text = ""
            if news is not None:
                if isinstance(news, (list, tuple)) and len(news) > b:
                    nb = news[b]
                    if isinstance(nb, str):
                        news_text = nb
                    elif isinstance(nb, (list, tuple)):
                        news_text = " ".join(str(x) for x in nb)
                    else:
                        news_text = ""
                elif isinstance(news, str):
                    news_text = news
            prompt_parts.append(f" Related events: {news_text}")
            # Add events to prompt if provided
            # if events is not None and b < len(events) and events[b] is not None:
            #     event_texts = []
            #     # Handle both dict and list of events
            #     if isinstance(events[b], dict):
            #         for event_key, event_text in events[b].items():
            #             if event_text:  # Skip empty events
            #                 event_texts.append(event_text)
            #     elif isinstance(events[b], list):
            #         event_texts = [e for e in events[b] if e]
                
            #     if event_texts:
            #         events_str = " ".join(event_texts)
            #         prompt_parts.append(f" Related events: {events_str}")
            
            prompt_parts.append("<|end_prompt|>")
            prompt_ = "".join(prompt_parts)
            
            prompt.append(prompt_)

        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        if not self.offline_mode:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))  # (batch, prompt_token, dim)
        else:
            # Simple whitespace tokenize and embed
            prompt_ids = self.simple_tokenizer(prompt)
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=x_enc.device)
            prompt_embeddings = self.word_embed_layer(prompt_ids)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings,enc_out], dim=1)
        if not self.offline_mode:
            dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
            dec_out = dec_out[:, :, :self.d_ff]
        else:
            dec_out = self.offline_encoder(llama_enc_out)
            dec_out = self.offline_proj(dec_out)

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        self.dec_out = dec_out.squeeze()
        return self.dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    def calculate_loss(self,batch_y):
        loss_func = nn.MSELoss()
        return loss_func(self.dec_out,batch_y)


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        ## Source Embedding = WTE + Linear 
        ## Target Embedding = x patch embedding 
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        ## 再过一遍线性层 映射到融合的隐空间 
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        ## 两者相乘 算注意力 
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        ## 两者相乘 
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class SimpleTokenizer:
    """A minimal whitespace tokenizer for offline mode.

    It hashes tokens into a fixed vocabulary and pads/truncates to a
    constant length per sample for easy batching.
    """
    def __init__(self, vocab_size: int = 8192, max_len: int = 64):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ids_batch = []
        for t in texts:
            tokens = t.split()
            ids = [(hash(tok) % (self.vocab_size - 2)) + 2 for tok in tokens]
            ids = ids[: self.max_len]
            if len(ids) < self.max_len:
                ids += [0] * (self.max_len - len(ids))
            ids_batch.append(ids)
        return ids_batch

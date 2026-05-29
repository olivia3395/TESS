from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from peft import LoraConfig,get_peft_model  # type: ignore
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
        self.top_k = 5
        self.d_llm = config["llm_dim"]
        self.patch_len = config["patch_len"]
        self.stride = config["stride"]
        self.llm_model_type = config["llm_model"]
        self.llm_layers = config["llm_layers"]
        self.lora_config = LoraConfig(
            r=12,
            lora_alpha = 32,
            target_modules = ["q_proj","v_proj","k_proj","o_proj"],
            lora_dropout = 0.05,
            bias="none"
        )
        self.llm_model=get_peft_model(self.llm_model,self.lora_config)
        self.llm_model.print_trainable_parameters()
        self.prompt_domain = config["prompt_domain"]
        if self.prompt_domain:
            self.description = config["content"]
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(config["dropout"])

        self.patch_embedding = PatchEmbedding(
            config["d_model"], self.patch_len, self.stride, config["dropout"])

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(config["d_model"], config["n_heads"], self.d_ff, self.d_llm)

        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums


        self.output_projection = FlattenHead(1, self.head_nf, self.pred_len,
                                                head_dropout=config["dropout"])
  

        self.normalize_layers = Normalize(1, affine=False)

    def forward(self, x_enc,x_text):

        dec_out = self.forecast(x_enc,x_text)
        return dec_out


    def forecast(self, x_enc,x_text):


        x_enc = x_enc.unsqueeze(-1)
         
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
    
        prompt = []
        
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings,enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out.squeeze()
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    def calculate_loss(self):
        pass


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

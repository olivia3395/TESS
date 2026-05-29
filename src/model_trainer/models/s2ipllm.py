#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 

class Prompt(nn.Module):
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        print(self.pool_size)
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape),requires_grad=False)
                print('zero initialized key')
                
            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')
            
            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape),requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')

            elif prompt_key_init == 'text_prototype':
                self.text_prototype_linear = nn.Linear(50257, pool_size)
                
          
        





        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_key:   #if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            
            if self.prompt_key_init == 'text_prototype':
                prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
            
            else:
                prompt_key = self.prompt
            
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C   self.prompt_key
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
                
            batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(2) # B, top_k, 1, length, C

            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_key'] = prompt_key  # prompt_key

        return out
    
class S2IPLLM(nn.Module):
    
    def __init__(self, configs):
        super(S2IPLLM, self).__init__()
        self.configs = configs
        self.is_ln = configs['ln']
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.patch_size = configs['patch_size']
        self.stride = configs['stride']
        self.d_ff = 768
        self.patch_num = (configs['seq_len'] - configs['patch_size']) // configs['stride'] + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.llm_layers = configs["llm_layers"]

        local_model_path = "/home/work/GPT2"
        self.gpt2_config = GPT2Config.from_pretrained(local_model_path)

        self.gpt2_config.num_hidden_layers = self.llm_layers 
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True
        self.gpt2 = GPT2Model.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )
        self.gpt2.h = self.gpt2.h[:configs['gpt_layers']]
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=False
            )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        


        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name  or 'wpe' in name:   #or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False

        self.in_layer = nn.Linear(configs['patch_size']*3, configs['embedding_size'])
        self.out_layer = nn.Linear(int(configs['embedding_size'] / 3 * (self.patch_num+configs['prompt_length'])) , configs['pred_len'])
        self.prompt_pool = Prompt(
            length=1,
            embed_dim=768,
            embedding_key='mean',
            prompt_init='uniform',
            prompt_pool=False,
            prompt_key=True,
            pool_size=self.configs['pool_size'],
            top_k=self.configs['prompt_length'],
            batchwise_prompt=False,
            # prompt_key_init=self.configs['prompt_init'],
            wte=self.gpt2.wte.weight
        )
        for layer in (self.gpt2, self.in_layer, self.out_layer):       
            layer.cuda()
            layer.train()


    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        dec_out = dec_out.squeeze(-1)
        return dec_out[:, -self.pred_len:]  # [B, L, D]
        
        

   

    def forecast(self, x_enc):
        

        
         
        B, L, M = x_enc.shape
            
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
 
        x = rearrange(x_enc, 'b l m -> (b m) l') 


        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs['trend_length'], center=True).mean().fillna(method='bfill').fillna(method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs['seasonal_length']).transform('mean').fillna(method='bfill').fillna(method='ffill') 
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined
                
            

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c = 3)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        pre_prompted_embedding = self.in_layer(x.float())




            
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

               

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*M*3, -1))
            
            
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss

        outputs = outputs * stdev[:,:,:M]
        outputs = outputs + means[:,:,:M]

        return outputs

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.mlp import MLP

class AutoTimes(nn.Module):
    def __init__(self, configs):
        super(AutoTimes, self).__init__()
        self.token_len = configs['token_len']
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
        self.tokenizer = GPT2Tokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=False
            )
        self.hidden_dim_of_gpt2 = 768
        # self.mix = configs.mix_embeds

        # if self.mix:
        #     self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        # if configs['mlp_hidden_layers'] == 0:
        #     if not configs['use_multi_gpu'] or (configs['use_multi_gpu'] and configs['local_rank'] == 0):
        #         print("use linear as tokenizer and detokenizer")
        #     self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
        #     self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        # else:
        #     if not configs['use_multi_gpu'] or (configs['use_multi_gpu'] and configs['local_rank'] == 0):
        #         print("use mlp as tokenizer and detokenizer")
        self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2, 
                        configs['mlp_hidden_dim'], configs['mlp_hidden_layers'], 
                        configs['dropout'], configs['mlp_activation'])
        self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                        configs['mlp_hidden_dim'], configs['mlp_hidden_layers'],
                        configs['dropout'], configs['mlp_activation']) 
    
    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        times_embeds = self.encoder(fold_out)
        # if self.mix:
        #     times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
        #     x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        #     times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        outputs = self.gpt2(
            inputs_embeds=times_embeds).last_hidden_state
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        x_enc = self.forecast(x_enc)
        x_enc = x_enc.squeeze(-1)
        return x_enc
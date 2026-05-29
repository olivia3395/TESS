import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x embedding_size x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.padding = configs['stride']
        self.patch_len = configs['patch_len']
        self.stride = configs['stride']
        ## 短序列应该测其他模型. 
        

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs['embedding_size'], self.patch_len, self.stride,configs['dropout'])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout'],
                                      output_attention=False), configs['embedding_size'], configs['n_heads']),
                    configs['embedding_size'],
                    configs['embedding_size'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs['embedding_size']), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs['embedding_size'] * \
                       int((configs['seq_len'] - self.patch_len) / self.stride + 2)
  
        self.head = FlattenHead(configs['enc_in'], self.head_nf, configs['pred_len'],
                                head_dropout=configs['dropout'])
        

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        x_enc = x_enc.unsqueeze(-1)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x embedding_size]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x embedding_size]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x embedding_size]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x embedding_size x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze()
        return self.dec_out

  

    def forward(self, x_enc):

        dec_out = self.forecast(x_enc)
        return dec_out
       


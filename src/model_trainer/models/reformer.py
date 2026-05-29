import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding


class Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Reformer, self).__init__()
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']

        self.enc_embedding = DataEmbedding(configs['enc_in'], configs['embedding_size'], configs['embed'], configs['freq'],
                                        configs['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs['embedding_size'], configs['n_heads'],
                                bucket_size=bucket_size, n_hashes=n_hashes),
                    configs['embedding_size'],
                    configs['embedding_size'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['embedding_size'])
        )

        self.projection = nn.Linear(
            configs['embedding_size'], configs['c_out'], bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        # add placeholder
        # x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        # if x_mark_enc is not None:
        #     x_mark_enc = torch.cat(
        #         [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        return dec_out  # [B, L, D]



    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        dec_out = dec_out.squeeze(-1)
        self.dec_out = dec_out[:, -self.pred_len:]
        return self.dec_out

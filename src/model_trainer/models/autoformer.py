'''
自回归无法复现
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Autoformer, self).__init__()
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']
        self.pred_len = configs['pred_len']

        # Decomp
        kernel_size = configs['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs['enc_in'], configs['embedding_size'], configs['embed'], configs['freq'],
                                                configs['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs['factor'], attention_dropout=configs['dropout'],
                                    output_attention=False),
                        configs['embedding_size'], configs['n_heads']),
                    configs['embedding_size'],
                    configs['embedding_size'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=my_Layernorm(configs['embedding_size'])
        )
        # Decoder
        self.dec_embedding = DataEmbedding_wo_pos(configs['dec_in'], configs['embedding_size'], configs['embed'], configs['freq'],
                                                configs['dropout'])
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs['factor'], attention_dropout=configs['dropout'],
                                    output_attention=False),
                        configs['embedding_size'], configs['n_heads']),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs['factor'], attention_dropout=configs['dropout'],
                                    output_attention=False),
                        configs['embedding_size'], configs['n_heads']),
                    configs['embedding_size'],
                    configs['c_out'],
                    configs['embedding_size'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=my_Layernorm(configs['embedding_size']),
            projection=nn.Linear(configs['embedding_size'], configs['c_out'], bias=True)
        )

    def forecast(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len,
        #                      x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = torch.cat(
        #     [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        seasonal_init = seasonal_init[:, -self.label_len:, :]
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        import ipdb;ipdb.set_trace()
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out


    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        self.dec_out = dec_out.squeeze(-1)
        return self.dec_out
    
    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss = F.mse_loss(outputs, batch_y)

        return loss

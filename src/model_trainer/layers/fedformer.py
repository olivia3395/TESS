import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']
        self.pred_len = configs['pred_len']

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs['moving_avg'])
        self.enc_embedding = DataEmbedding(configs['enc_in'], configs['embedding_size'], configs['embed'], configs['freq'],
                                        configs['dropout'])
        self.dec_embedding = DataEmbedding(configs['dec_in'], configs['embedding_size'], configs['embed'], configs['freq'],
                                        configs['dropout'])

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs['embedding_size'], L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs['embedding_size'], L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs['embedding_size'],
                                                out_channels=configs['embedding_size'],
                                                seq_len_q=self.seq_len // 2 + self.pred_len,
                                                seq_len_kv=self.seq_len,
                                                modes=self.modes,
                                                ich=configs['embedding_size'],
                                                base='legendre',
                                                activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs['embedding_size'],
                                        out_channels=configs['embedding_size'],
                                        n_heads=configs['n_heads'],
                                        seq_len=self.seq_len,
                                        modes=self.modes,
                                        mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs['embedding_size'],
                                        out_channels=configs['embedding_size'],
                                        n_heads=configs['n_heads'],
                                        seq_len=self.seq_len // 2 + self.pred_len,
                                        modes=self.modes,
                                        mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs['embedding_size'],
                                                    out_channels=configs['embedding_size'],
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len,
                                                    modes=self.modes,
                                                    mode_select_method=self.mode_select,
                                                    num_heads=configs['n_heads'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs['embedding_size'], configs['n_heads']),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out



    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        self.dec_out = dec_out.unsqueeze(-1)
        return self.dec_out
    
    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss = F.mse_loss(outputs, batch_y)

        return loss


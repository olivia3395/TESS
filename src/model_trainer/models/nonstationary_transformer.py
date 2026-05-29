import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Nonstationary_Transformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Nonstationary_Transformer, self).__init__()
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.label_len = configs['label_len']

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=configs['enc_in'], 
            d_model=configs['embedding_size'], 
            embed_type=configs['embed'], 
            dropout=configs['dropout']
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False, 
                            configs['factor'], 
                            attention_dropout=configs['dropout'],
                            output_attention=False
                        ), 
                        configs['embedding_size'], 
                        configs['n_heads']
                    ),
                    configs['embedding_size'],
                    configs['embedding_size'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['embedding_size'])
        )
        
        # Decoder
        self.dec_embedding = DataEmbedding(
            c_in=configs['dec_in'], 
            d_model=configs['embedding_size'], 
            embed_type=configs['embed'], 
            dropout=configs['dropout']
        )
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(
                            True, 
                            configs['factor'], 
                            attention_dropout=configs['dropout'],
                            output_attention=False
                        ),
                        configs['embedding_size'], 
                        configs['n_heads']
                    ),
                    AttentionLayer(
                        DSAttention(
                            False, 
                            configs['factor'], 
                            attention_dropout=configs['dropout'],
                            output_attention=False
                        ),
                        configs['embedding_size'], 
                        configs['n_heads']
                    ),
                    configs['embedding_size'],
                    configs['embedding_size'],
                    dropout=configs['dropout'],
                    activation=configs['activation'],
                )
                for l in range(configs['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['embedding_size']),
            projection=nn.Linear(configs['embedding_size'], configs['c_out'], bias=True)
        )
        self.tau_learner = Projector(
            enc_in=configs['enc_in'], 
            seq_len=configs['seq_len'], 
            hidden_dims=configs['p_hidden_dims'],
            hidden_layers=configs['p_hidden_layers'], 
            output_dim=1
        )
        
        self.delta_learner = Projector(
            enc_in=configs['enc_in'], 
            seq_len=configs['seq_len'],
            hidden_dims=configs['p_hidden_dims'], 
            hidden_layers=configs['p_hidden_layers'],
            output_dim=configs['seq_len']
        )

    def forecast(self, x_enc, x_mark_enc=None,x_mark_dec=None):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        # x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
        #                       dim=1).to(x_enc.device).clone()
        x_dec_new = x_enc[:, -self.label_len:, :].to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out


    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        dec_out = dec_out.squeeze(-1)
        self.dec_out = dec_out[:, -self.pred_len:]
        return self.dec_out
        

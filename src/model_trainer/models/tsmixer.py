import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs['seq_len'], configs['embedding_size']),
            nn.ReLU(),
            nn.Linear(configs['embedding_size'], configs['seq_len']),
            nn.Dropout(configs['dropout'])
        )

        self.channel = nn.Sequential(
            nn.Linear(configs['enc_in'], configs['embedding_size']),
            nn.ReLU(),
            nn.Linear(configs['embedding_size'], configs['enc_in']),
            nn.Dropout(configs['dropout'])
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class TSMixer(nn.Module):
    def __init__(self, configs):
        super(TSMixer, self).__init__()
        self.layer = configs['e_layers']
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs['e_layers'])])
        self.pred_len = configs['pred_len']
        self.projection = nn.Linear(configs['seq_len'], configs['pred_len'])

    def forecast(self, x_enc):

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return enc_out

    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc)
        dec_out = dec_out.squeeze(-1)
        self.dec_out = dec_out[:, -self.pred_len:]
        return self.dec_out 


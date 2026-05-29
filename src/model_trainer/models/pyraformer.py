import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder
import torch.nn.functional as F

class Pyraformer(nn.Module):
    """ 
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
       
        self.pred_len = configs['pred_len']
        self.embedding_size = configs['embedding_size']
        # self.window_size = [4,4] 这是bitcoin上的设置，在electricity和原来的为[4,4]
        self.window_size = [2,2]

        self.inner_size = configs['inner_size']
    
       
        self.encoder = Encoder(configs, self.window_size, self.inner_size)

        
        self.projection = nn.Linear(
            (len(self.window_size)+1)*self.embedding_size, self.pred_len )
        
    # def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
    #     dec_out = self.projection(enc_out).view(
    #         enc_out.size(0), self.pred_len, -1)
    #     return dec_out
    
    def forecast(self, x_enc):
        # Normalization
        x_enc = x_enc.unsqueeze(-1)
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        
        dec_out = dec_out * std_enc + mean_enc
        self.dec_out = dec_out.squeeze()
        return self.dec_out

   

    def forward(self, x_enc):
   
        dec_out = self.forecast(x_enc)
        return dec_out  # [B, L, D]

    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss_main = F.mse_loss(outputs, batch_y)

       
        return loss_main
    


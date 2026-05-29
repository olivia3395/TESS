import torch
import torch.nn as nn
import torch.nn.functional as F
from model_trainer.layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from model_trainer.layers.Embed import DataEmbedding
from model_trainer.layers.MultiModal import CrossModalAttention, CrossModalTransformer
from model_trainer.layers.Causal import TempDisentangler
from model_trainer.layers.Causal import EnvEmbedding
from model_trainer.layers.StandardNorm import Normalize
# from utils.llm_model import LLMInitializer
import math

class CMA(nn.Module):
    def __init__(self, configs):
        super(CMA, self).__init__()
    
        self.pred_len = configs["pred_len"]
        self.seq_len = configs["seq_len"]
        def _resolve(value):
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return value[0]
            return value

        self.embedding_size = _resolve(configs["embedding_size"])
        self.hid_dim = self.embedding_size
        self.mm_emb_dim = self.embedding_size
        self.dropout = float(_resolve(configs['dropout']))
        self.meta_drop = float(_resolve(configs['meta_drop']))
        self.sudden_drop = float(_resolve(configs['sudden_drop']))
        self.fuse_drop = float(_resolve(configs['fuse_drop']))
        self.depth = int(_resolve(configs['depth']))
        self.num_envs = int(_resolve(configs['num_envs']))
        self.e_layers = int(_resolve(configs["e_layers"]))
        self.enc_in = int(_resolve(configs["enc_in"]))
        self.embed = configs["embed"]
        self.freq = configs['freq'] or 'h'

        self.enc_embedding = DataEmbedding(self.enc_in, self.hid_dim, self.embed, freq=self.freq, dropout=self.dropout)
      
        t_kernels = [2**i for i in range(int(math.log2(self.seq_len//2)))]

        self.temporal = TempDisentangler(
            input_dims=self.hid_dim,
            output_dims=self.hid_dim * 2,
            kernels=t_kernels,
            length=self.seq_len,
            hidden_dims=self.hid_dim,
            depth=self.depth,
            dropout=self.dropout
        )

    
        self.codebook_dim = int(_resolve(configs['codebook_dim']))
        self.codebook = EnvEmbedding(self.num_envs, self.codebook_dim)
        self.env_to_codebook = nn.Linear(self.hid_dim, self.codebook_dim)
        self.codebook_to_env = nn.Linear(self.codebook_dim, self.hid_dim)

        # Flatten the h_entity
        self.mlp_flatten = nn.Sequential(
                nn.Linear(self.seq_len * self.hid_dim, self.hid_dim),  # 输入是 seq_len * hid_dim，输出是 hid_dim
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
        self.mlp_flatten_2 = nn.Sequential(
                nn.Linear(self.seq_len * self.hid_dim, self.hid_dim),  # 输入是 seq_len * hid_dim，输出是 hid_dim
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
        
        ##
        
        self.time_to_mm = nn.Linear(self.hid_dim, self.mm_emb_dim)
        self.text_emb_dim = int(_resolve(configs['text_emb_dim']))
        self.meta_fc = nn.Sequential(
            nn.Linear(self.text_emb_dim , self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.meta_drop)
        )
        self.dynamic_fc = nn.Sequential(
            nn.Linear(self.text_emb_dim , self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.sudden_drop)
        )
        # Multi-modal attention
        self.dynamic_fused_layer = CrossModalAttention(embed_dim=self.embedding_size,dropout=self.fuse_drop)
        self.static_fused_layer = CrossModalAttention(embed_dim=self.embedding_size,dropout=self.fuse_drop)
        
        # Decoder for prediction
       
        self.decoder_mlp = nn.Sequential(
                nn.Linear(self.hid_dim, 256),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256,512),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, self.pred_len)
            )


        self.beta1 = float(_resolve(configs["beta1"]))
        self.beta2 = float(_resolve(configs["beta2"]))
        self.normalize_layers = Normalize(1, affine=False)

   
    def forward(self,x_enc,news_feat,flag='train'):
        """
        时序编码->解耦  获得时序的初始表征 
        """
        """
        x_enc: 原始的时间序列数据 B L Channel_Size 
        x_text:预提取的文本表征
        """

        x_enc = x_enc.unsqueeze(-1)
        self.batch_size = x_enc.size(0)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc) 
        h_env, h_entity = self.temporal(enc_out) 
        

        
        
       

        ## Dynamic Context Learning 
        flatten_h_entity =  h_entity.view(self.batch_size, -1)
        h_entity = self.mlp_flatten_2(flatten_h_entity)
        news_embed = self.dynamic_fc(news_feat)
        h_entity = self.dynamic_fused_layer(h_entity,news_embed)
        h_final_repr = h_entity
        
        dec_out = self.decoder_mlp(h_final_repr).unsqueeze(-1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)
        
        #画数据分布图的时候用的
        # return self.dec_out,h_env,h_entity,env_ind
        return self.dec_out
    
    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss_cons = F.mse_loss(outputs, batch_y)
 

        
        return loss_cons
    
    def get_model_embeddings(self,batch_x,meta_domain,news,ct1,flag):
        dec_out,h_env, h_entity, env_ind = self.forward(batch_x,meta_domain,news,ct1,flag)  
        return {
            'h_env': h_env,
            'h_entity': h_entity,
            'env_ind': env_ind
        }


        
     
        



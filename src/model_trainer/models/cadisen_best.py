import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.Embed import DataEmbedding
from layers.MultiModal import CrossModalAttention,CrossModalTransformer
from layers.Causal import TempDisentangler
from layers.Causal import EnvEmbedding
from layers.StandardNorm import Normalize
# from utils.llm_model import LLMInitializer
import math
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

class CaDisen_best(nn.Module):
    def __init__(self, configs):
        super(CaDisen_best, self).__init__()
    
        self.pred_len = configs["pred_len"]
        self.seq_len = configs["seq_len"]
        self.embedding_size = configs["embedding_size"]
        self.hid_dim = configs['embedding_size']
        self.mm_emb_dim = configs['embedding_size']
        self.dropout = configs['dropout'] 
        self.meta_drop = configs['meta_drop']
        self.sudden_drop = configs['sudden_drop']
        self.fuse_drop = configs['fuse_drop']
        self.depth = configs['depth']
        self.num_envs = configs['num_envs']
        self.e_layers = configs["e_layers"]
        self.enc_in = configs["enc_in"]
        self.embed = configs["embed"]


        self.enc_embedding = DataEmbedding(self.enc_in, self.hid_dim, self.embed, self.dropout)
      
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

    
        self.codebook_dim = configs['codebook_dim']
        self.codebook = EnvEmbedding(self.num_envs,self.codebook_dim)
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
        self.text_emb_dim = configs['text_emb_dim']
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
                nn.Linear(self.hid_dim*2  , 256),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256,512),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, self.pred_len)
            )

        self.mi_regulization = nn.CrossEntropyLoss()

        self.beta1 = configs["beta1"]
        self.beta2 = configs["beta2"]
        self.normalize_layers = Normalize(1, affine=False)

   
    def forward(self,x_enc,meta_feat,news_feat,flag='train'):
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
        

        
        ## 文本做分层匹配,以做干预 
        meta_embed = self.meta_fc(meta_feat)
        env_in = self.env_to_codebook(meta_embed)
        if flag == "train":
            env_output,env_q,env_ind = self.codebook.straight_through(env_in)
        elif flag =="test":
            env_output, env_q, env_ind = self.codebook.straight_through_test(env_in)
        meta_embed = self.codebook_to_env(env_output)
        
        flatten_h_env = h_env.view(self.batch_size,-1)
        h_env = self.mlp_flatten(flatten_h_env)
        h_env = self.static_fused_layer(h_env,meta_embed)
        
       

        ## Dynamic Context Learning 
        flatten_h_entity =  h_entity.view(self.batch_size, -1)
        h_entity = self.mlp_flatten_2(flatten_h_entity)
        news_embed = self.dynamic_fc(news_feat)
        h_entity = self.dynamic_fused_layer(h_entity,news_embed)
        h_final_repr = torch.cat((h_env,h_entity),dim=1)
        
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


        
     
        



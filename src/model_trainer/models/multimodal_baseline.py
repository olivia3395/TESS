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
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
"""
关于怎么做实验: 
Multi Modal Baseline: 
1. 有分析的版本, 通过qwen3 embedding model进行编码
2. 去掉分析的版本, 尝试两种编码方式 qwen3 embedding model/ qwen3 llm model的词表层编码(mean pooling)
3. 用qwen3 llm model生成过程的最后一层隐藏层进行编码  

Multi Modal Baseline Token Level: 
1. 用qwen3 llm model的词表层进行编码 即 get_input_embedding  这个已经尝试过了
2. 用qwen3 llm 生成过程中的最后一层隐藏层进行编码, 这个需要重新跑一遍生成过程?
以上两个模型还需要测试没有使用文本,纯时序的版本, 效果怎么样  
每个版本的实验 都需要用清晰的自然语言记录,参数 结果 
需要搜索超参数,我在configs中已经配置了超参数的空间
同时记录训练每个epoch的在训练集上的loss变化,在验证集上,测试集上的mse变化 可以根据这个判断模型是否收敛,是否过拟合,画几个曲线图 
并且分成两组样本进行事后归因:
(1) 加入文本后, MSE下降的测试样本
(2) 加入文本后, MSE上升的测试样本 
对(1) 样本进行token归因, 观察这些样本往往都有哪些token, 是否因为强度和趋势分析是正确的
对(2) 样本进行token归因 , 观察这些样本的文本有哪些共性, 是否是因为强度和趋势分析是错误的, 梯度近似的算法, 分析哪些文本token在预测中起到了作用。

还需要分析, 为什么加入分析之后的效果更好了?  我猜测背景信息中, 可以让模型学到相似的文本的作用 

性能的曲线图: 
case study section:分析gen1和gen2 
坍缩的曲线: 
特征归因token分析: 
文本和文本之间的关系:  A和B同样是一种背景, 然后时序是相似的, 所以背景相当于起到了一种连接的作用 


"""

class MultiModal_Baseline(nn.Module):
    def __init__(self, configs):
        super(MultiModal_Baseline, self).__init__()
    
        self.pred_len = configs["pred_len"]
        self.seq_len = configs["seq_len"]
        self.embedding_size = configs["embedding_size"]
        self.hid_dim = configs['embedding_size']
        self.mm_emb_dim = configs['embedding_size']
        self.dropout = configs['dropout'] 

        self.sudden_drop = configs['sudden_drop']
        self.fuse_drop = configs['fuse_drop']
        self.depth = configs['depth']

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



        # Flatten the h_ts
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

        self.dynamic_fc = nn.Sequential(
            nn.Linear(self.text_emb_dim , self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.sudden_drop)
        )
        # Multi-modal attention
        self.dynamic_fused_layer = CrossModalAttention(embed_dim=self.embedding_size,dropout=self.fuse_drop)

        
        # Decoder for prediction
       
        self.decoder_mlp = nn.Sequential(
                nn.Linear(self.hid_dim , 256),
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
        self.last_attention = None
        self.last_attention_scores = None

   
    def forward(self,x_enc,news_feat,flag='train'):
        """

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
        h_env, h_ts = self.temporal(enc_out) 
        
        
       

        ## Dynamic Context Learning 
        flatten_h_ts =  h_ts.view(self.batch_size, -1)
        h_ts = self.mlp_flatten_2(flatten_h_ts)
        #dropout 


        
    
        news_embed = self.dynamic_fc(news_feat)#

        h_ts, attn_probs = self.dynamic_fused_layer(h_ts, news_embed, return_scores=True)#
        self.last_attention = attn_probs.detach()
        scores = getattr(self.dynamic_fused_layer, "last_attention_scores", None)
        self.last_attention_scores = scores.detach() if isinstance(scores, torch.Tensor) else None
    
        dec_out = self.decoder_mlp(h_ts).unsqueeze(-1)#

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)
        
        return self.dec_out
    
    def calculate_loss(self,batch_y):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss_cons = F.mse_loss(outputs, batch_y)
 

        
        return loss_cons
    
    def get_model_embeddings(self,batch_x,meta_domain,news,ct1,flag):
        dec_out,h_env, h_ts, env_ind = self.forward(batch_x,meta_domain,news,ct1,flag)  
        return {
            'h_env': h_env,
            'h_ts': h_ts,
            'env_ind': env_ind
        }


        
     
        


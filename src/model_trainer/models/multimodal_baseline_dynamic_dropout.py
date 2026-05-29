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
Dynamic Dropout MultiModal Baseline:

基于原始MultiModal_Baseline，添加动态dropout机制：
1. 动态dropout选择层：学习每个token每个维度的dropout概率
2. 训练时使用离散dropout（0/1），评估时使用连续加权
3. Word embedding loss：监督dropout后的embeddings接近理想embeddings
4. 支持token级别文本输入 [B, L, text_emb_dim]

实验目标：探究文本中哪些token对时间序列预测最重要，通过学习性dropout机制实现。
"""

class MultiModal_Baseline_Dynamic_Dropout(nn.Module):
    def __init__(self, configs):
        super(MultiModal_Baseline_Dynamic_Dropout, self).__init__()

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
                nn.Linear(self.seq_len * self.hid_dim, self.hid_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
        self.mlp_flatten_2 = nn.Sequential(
                nn.Linear(self.seq_len * self.hid_dim, self.hid_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )

        self.time_to_mm = nn.Linear(self.hid_dim, self.mm_emb_dim)
        self.text_emb_dim = configs['text_emb_dim']

        self.dynamic_fc = nn.Sequential(
            nn.Linear(self.text_emb_dim , self.mm_emb_dim),
            nn.PReLU(),
            nn.Dropout(self.sudden_drop)
        )
        # Multi-modal attention
        self.dynamic_fused_layer = CrossModalAttention(embed_dim=self.embedding_size,dropout=self.fuse_drop)

        # Dynamic dropout selection layer - outputs [B, L, D] mask for each token dimension
        self.dynamic_dropout_selection_layer = nn.Sequential(
            nn.Linear(self.text_emb_dim, self.text_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.text_emb_dim // 2, self.text_emb_dim),
            nn.Sigmoid()  # [0,1] probabilities for each dimension
        )

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
        self.beta3 = configs.get("beta3", 0.1)  # word embedding loss weight
        self.beta4 = configs.get("beta4", 0.05)  # reconstruction loss weight

        self.normalize_layers = Normalize(1, affine=False)
        self.last_attention = None
        self.last_attention_scores = None


    def forward(self,x_enc,news_feat,flag='train', attention_mask=None):
        """
        x_enc: 原始的时间序列数据 B L Channel_Size
        news_feat: token级别文本embeddings [B, L, text_emb_dim]
        attention_mask: [B, L] - 1 for real tokens, 0 for padding
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

        # Dynamic dropout for token embeddings
        # news_feat: [B, L, text_emb_dim] - token-level embeddings
        word_embedding = news_feat

        # 保存 attention_mask 供 calculate_loss 使用
        self.attention_mask = attention_mask

        # Compute dropout mask [B, L, D]
        dropout_mask = self.dynamic_dropout_selection_layer(word_embedding)

        # Apply attention mask to avoid dropout on padding tokens
        if attention_mask is not None:
            # attention_mask: [B, L] - 1 for real tokens, 0 for padding
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            dropout_mask = dropout_mask * attention_mask_expanded  # Don't dropout padding tokens

        # Training vs Evaluation logic
        if flag == "train":
            # Training: Use Bernoulli sampling for discrete dropout (0 or 1)
            discrete_mask = torch.bernoulli(dropout_mask)
            dropout_word_embeddings = word_embedding * discrete_mask
        else:
            # Evaluation: Use continuous mask for soft weighting
            dropout_word_embeddings = word_embedding * dropout_mask

        # Pool to get sentence-level representation [B, D]
        # Use attention_mask for weighted pooling if available
        if attention_mask is not None:
            # Mask out padding tokens for proper averaging
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            masked_embeddings = dropout_word_embeddings * attention_mask_expanded
            h_text = masked_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_text = dropout_word_embeddings.mean(dim=1)

        # Save intermediate results for loss computation
        self.dropout_word_embeddings = dropout_word_embeddings
        self.dropout_mask = dropout_mask
        self.original_word_embeddings = word_embedding

        news_embed = self.dynamic_fc(h_text)

        h_ts, attn_probs = self.dynamic_fused_layer(h_ts, news_embed, return_scores=True)
        self.last_attention = attn_probs.detach()
        scores = getattr(self.dynamic_fused_layer, "last_attention_scores", None)
        self.last_attention_scores = scores.detach() if isinstance(scores, torch.Tensor) else None

        dec_out = self.decoder_mlp(h_ts).unsqueeze(-1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)

        return self.dec_out

    def calculate_loss(self, batch_y, gt_embeddings=None, text_quality_gt=None, mode='train'):

        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        loss_cons = F.mse_loss(outputs, batch_y)

        # 评估时只返回时序MSE，不包含其他损失项
        if mode != 'train':
            return loss_cons

        total_loss = loss_cons

        # Word embedding loss: dropout后的embeddings vs GT embeddings（仅在训练时计算）
        if gt_embeddings is not None and hasattr(self, 'dropout_word_embeddings'):
            # 确保维度匹配
            if gt_embeddings.shape != self.dropout_word_embeddings.shape:
                raise ValueError(f"GT embeddings shape {gt_embeddings.shape} 与 dropout_word_embeddings shape {self.dropout_word_embeddings.shape} 不匹配")
            
            # 如果有 attention_mask，只对有效 token 计算损失
            if hasattr(self, 'attention_mask') and self.attention_mask is not None:
                # attention_mask: [B, L] - 1 for real tokens, 0 for padding
                attention_mask_expanded = self.attention_mask.unsqueeze(-1).float()  # [B, L, 1]
                # 只对有效 token 计算 MSE
                valid_gt = gt_embeddings * attention_mask_expanded
                valid_dropout = self.dropout_word_embeddings * attention_mask_expanded
                # 计算每个样本的有效 token 数量
                valid_counts = attention_mask_expanded.sum(dim=1).sum(dim=1).clamp(min=1)  # [B]
                # 计算每个样本的 MSE，然后平均
                per_sample_mse = ((valid_gt - valid_dropout) ** 2).sum(dim=(1, 2)) / valid_counts  # [B]
                loss_word_emb = per_sample_mse.mean()
            else:
                # 如果没有 attention_mask，直接计算 MSE
                loss_word_emb = F.mse_loss(self.dropout_word_embeddings, gt_embeddings)
            # 控制word embedding loss的权重，默认权重0.1
            total_loss += self.beta3 * loss_word_emb

        return total_loss

    def get_model_embeddings(self,batch_x,meta_domain,news,ct1,flag):
        dec_out = self.forward(batch_x,meta_domain,news,ct1,flag)
        return {
            'dec_out': dec_out
        }

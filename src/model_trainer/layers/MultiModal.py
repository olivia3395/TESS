import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

        # LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, h_time, h_text):
        """
        Cross-modal attention mechanism optimized for 2D inputs.
        :param h_time: Tensor of shape [batch_size, seq_len, embed_dim] (Time-series data)
        :param h_text: Tensor of shape [batch_size, seq_len, embed_dim] (Text data)
        :return: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = h_time.shape

        # 1. Projection: Convert both time-series and text to Queries, Keys, Values
        Q = self.q_proj(h_time)  # [batch_size, seq_len, embed_dim]
        K = self.k_proj(h_text)  # [batch_size, seq_len, embed_dim]
        V = self.v_proj(h_text)  # [batch_size, seq_len, embed_dim]

        # 2. Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]

        # 3. Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, n_heads, seq_len, seq_len]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, seq_len, seq_len]
        attn_probs = self.dropout_layer(attn_probs)

        # 4. Apply attention to values
        attn_output = torch.matmul(attn_probs, V)  # [batch_size, n_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # [batch_size, seq_len, embed_dim]

        # 5. Add & Normalize: Skip connection + Layer Normalization
        output = self.layer_norm(attn_output + h_time)  # [batch_size, seq_len, embed_dim]

        # 6. Output projection
        output = self.out_proj(output)  # [batch_size, seq_len, embed_dim]

        return output
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.dropout = dropout
        self.last_attention = None
        self.last_attention_scores = None

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Attention output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, h_time, h_text, return_scores: bool = False):
        """
        Cross-modal attention mechanism optimized for 2D inputs.
        :param h_time: Tensor of shape [batch_size,seq_len, mm_emb_dim] (Time-series data)
        :param h_text: Tensor of shape [batch_size, seq_len,mm_emb_dim] (Text data)
        :return: Tensor of shape [batch_size, mm_emb_dim]
        """
        # 1. Projection: Convert both time-series and text to Queries, Keys, Values
        Q = self.q_proj(h_time)  # [batch_size, mm_emb_dim]
        K = self.k_proj(h_text)    # [batch_size, mm_emb_dim]
        V = self.v_proj(h_text)    # [batch_size, mm_emb_dim]

        # 2. Scaled Dot-Product Attention (Q, K, V)
        # Use batch matrix multiplication to avoid unsqueezing tensors
        # Q * K^T: Compute attention scores directly
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / (self.embed_dim ** 0.5)  # [batch_size, 1]
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [batch_size, 1]
        self.last_attention_scores = attn_scores.detach()
        self.last_attention = attn_probs.detach()

        # 3. Attention output: Weighted sum of V
        attn_output = attn_probs * V  # [batch_size, mm_emb_dim]

        # 4. Add & Normalize: Skip connection + Layer Normalization
        output = self.layer_norm(attn_output + h_time)

        # 5. Output projection (Optional, depending on your use case)
        output = self.out_proj(output)  # [batch_size, mm_emb_dim]

        if return_scores:
            return output, attn_probs
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossModalAttention(nn.Module):
#     def __init__(self, hid_dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.num_heads = num_heads
#         self.head_dim = hid_dim // num_heads

#         # 定义 Query、Key、Value 的线性变换
#         self.q_linear = nn.Linear(hid_dim, hid_dim)
#         self.k_linear = nn.Linear(hid_dim, hid_dim)
#         self.v_linear = nn.Linear(hid_dim, hid_dim)

#         # 输出层和层归一化
#         self.fc_out = nn.Linear(hid_dim, hid_dim)
#         self.layer_norm = nn.LayerNorm(hid_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, query, key, value):
#         # 输入形状: [batch_size, seq_len, hid_dim]
#         batch_size = query.shape[0]

#         # 残差连接保留原始输入
#         residual = query

#         # 线性变换获取 Q, K, V
#         Q = self.q_linear(query)
#         K = self.k_linear(key)
#         V = self.v_linear(value)

#         # 切分为多头注意力
#         Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         # 计算注意力分数
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
#         attention = F.softmax(energy, dim=-1)
#         x = torch.matmul(self.dropout(attention), V)

#         # 合并多头并投影
#         x = x.permute(0, 2, 1, 3).contiguous()
#         x = x.view(batch_size, -1, self.hid_dim)
#         x = self.fc_out(x)

#         # 残差连接和层归一化
#         x = self.layer_norm(x + residual)
#         return x
    
# class CrossModalFusion(nn.Module):
#     def __init__(self, hid_dim, num_heads=4):
#         super().__init__()
#         # 定义双向交叉注意力
#         self.attn_entity2text = CrossModalAttention(hid_dim, num_heads)
#         self.attn_text2entity = CrossModalAttention(hid_dim, num_heads)

#         # 最终融合层
#         self.fc = nn.Linear(2 * hid_dim, hid_dim)

#     def forward(self, h_entity, h_text):
#         # 时序模态作为 Query，文本模态作为 Key/Value
#         out_entity = self.attn_entity2text(h_entity, h_text, h_text)

#         # 文本模态作为 Query，时序模态作为 Key/Value
#         out_text = self.attn_text2entity(h_text, h_entity, h_entity)

#         # 拼接并融合
#         fused = torch.cat([out_entity, out_text], dim=-1)
#         fused = self.fc(fused)  # [batch_size, seq_len, hid_dim]

#         return fused    


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrossModalTransformer(nn.Module):
    def __init__(self, hid_dim, nhead, num_layers):
        super(CrossModalTransformer, self).__init__()
        # Transformer 编码器层
        encoder_layer = TransformerEncoderLayer(d_model=hid_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=nhead)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, h_entity, h_text):
        # 输入形状: [bs, horizon, hid_dim]
        # 转换为 Transformer 的输入形状: [horizon, bs, hid_dim]
        h_entity = h_entity.permute(1, 0, 2)
        h_text = h_text.permute(1, 0, 2)
        
        # 通过 Transformer 编码器
        h_entity = self.transformer_encoder(h_entity)
        h_text = self.transformer_encoder(h_text)
        
        # 交叉注意力融合
        h_cross, _ = self.cross_attention(h_entity, h_text, h_text)
        
        # 转换为原始形状: [bs, horizon, hid_dim]
        h_cross = h_cross.permute(1, 0, 2)
        
        # 全局池化: [bs, horizon, hid_dim] -> [bs, hid_dim]
        h_cross = self.global_pool(h_cross.permute(0, 2, 1)).squeeze(-1)
        
        return h_cross

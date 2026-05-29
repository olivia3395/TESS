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
Conditional Quality-Aware MultiModal Baseline Model

核心思想：保留原有注意力机制，保守增强质量感知
关键策略：基础权重(注意力) × 质量分数(专家指导)

新增功能：
1. 保留原有的CrossModalAttention机制：基础权重学习
2. 条件质量评估器：外部标签指导的质量分数计算
3. 保守权重融合：基础权重 × 质量分数
4. 联合优化：预测损失 + 质量评估损失

工作原理：
- 第一步：用原始注意力机制计算基础权重（保持原有学习能力）
- 第二步：用条件质量评估计算质量分数（专家标签指导）
- 第三步：两个权重相乘得到最终权重（保守增强）
- 最终：既保留原有效果，又加入质量感知
"""

class MultiModal_Baseline_QualityAware(nn.Module):
    def __init__(self, configs):
        super(MultiModal_Baseline_QualityAware, self).__init__()

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

        # 🔴 新增：条件质量评估器
        # 输入：时序特征 + 文本特征，输出：上下文相关的质量分数
        self.conditional_quality_net = nn.Sequential(
            nn.Linear(self.hid_dim + self.mm_emb_dim, 128),  # 时序dim + 文本dim
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),  # 输出质量分数logits
        )

        # 用于保存时序特征，供质量评估使用
        self.temporal_feature = None

        # 质量损失权重，可配置
        self.quality_loss_weight = getattr(configs, 'quality_loss_weight', 0.1)

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


    def forward(self, x_enc, news_feat, text_quality_gt=None, flag='train'):
        """
        Conditional Quality-Aware Forward Pass

        Args:
            x_enc: 时间序列数据 [B, L]
            news_feat: 文本嵌入特征 [B, text_emb_dim] (已编码)
            text_quality_gt: 外部质量标签 [B] (0/1) - 训练时提供
            flag: 'train' 或 'test'
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

        # 文本特征映射
        news_embed = self.dynamic_fc(news_feat)  # [B, mm_emb_dim]

        # 🟢 第一步：使用原来的注意力机制计算基础权重
        # 保持原有模型的文本权重学习能力
        h_ts_base, attn_probs_base = self.dynamic_fused_layer(h_ts, news_embed, return_scores=True)
        base_attention_weights = attn_probs_base  # [B, 1] - 基础注意力权重

        # 🔴 第二步：条件质量评估（增强）
        # 将时序特征和文本特征拼接，作为质量评估的条件输入
        combined_features = torch.cat([h_ts, news_embed], dim=-1)  # [B, hid_dim + mm_emb_dim]
        quality_logits = self.conditional_quality_net(combined_features)  # [B, 1]
        quality_confidence = torch.sigmoid(quality_logits).squeeze(-1)  # [B] - 0~1质量分数

        # 🔵 第三步：保守的权重融合
        # 结合基础注意力和质量评估：base_weight * quality_score
        # 这样既保留了原有的学习能力，又加入了质量感知
        final_weight = base_attention_weights.squeeze(-1) * quality_confidence  # [B]
        final_weight = final_weight.unsqueeze(-1)  # [B, 1]

        # 应用最终权重
        weighted_news_embed = news_embed * final_weight  # [B, mm_emb_dim]

        # 跨模态融合（使用调整后的权重）
        h_ts, attn_probs = self.dynamic_fused_layer(h_ts, weighted_news_embed, return_scores=True)
        self.last_attention = attn_probs.detach()
        scores = getattr(self.dynamic_fused_layer, "last_attention_scores", None)
        self.last_attention_scores = scores.detach() if isinstance(scores, torch.Tensor) else None

        # 解码预测
        dec_out = self.decoder_mlp(h_ts).unsqueeze(-1)

        # 逆标准化
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        self.dec_out = dec_out.squeeze(-1)

        # 保存质量评估结果
        self.quality_logits = quality_logits
        self.quality_confidence = quality_confidence

        return self.dec_out


    def calculate_loss(self, batch_y, text_quality_gt=None):
        """
        智能损失计算：自动区分训练和评估阶段

        - 训练时：如果提供了text_quality_gt，返回联合损失用于优化
        - 评估时：只返回预测损失用于计算MSE等指标

        Args:
            batch_y: 目标预测值 [B, pred_len]
            text_quality_gt: 外部质量标签 [B] (0/1) - 可选，训练时提供
        """
        outputs = self.dec_out[:, -self.pred_len:]
        batch_y = batch_y[:, -self.pred_len:].to(outputs.device)

        # 预测损失 - 始终计算，用于评估指标
        loss_pred = F.mse_loss(outputs, batch_y)

        # 🔴 质量评估损失：仅当提供外部标签时包含
        if text_quality_gt is not None and hasattr(self, 'quality_logits'):
            loss_quality = F.binary_cross_entropy_with_logits(
                self.quality_logits.squeeze(-1),  # [B]
                text_quality_gt.float()           # [B] - 外部标签
            )
            # 联合损失：用于训练优化
            total_loss = loss_pred + self.quality_loss_weight * loss_quality
            return total_loss

        # 🔵 纯预测损失：用于评估指标计算
        return loss_pred


    def get_model_embeddings(self, batch_x, news_feat, flag='train'):
        """兼容性方法"""
        dec_out = self.forward(batch_x, news_feat, flag)
        return {
            'h_env': None,  # 暂时不支持
            'h_ts': None,   # 暂时不支持
            'env_ind': None
        }


    def get_quality_info(self):
        """获取条件质量评估相关信息，用于分析"""
        return {
            'quality_logits': getattr(self, 'quality_logits', None),
            'quality_confidence': getattr(self, 'quality_confidence', None),
            'last_attention': self.last_attention,
            'last_attention_scores': self.last_attention_scores
        }

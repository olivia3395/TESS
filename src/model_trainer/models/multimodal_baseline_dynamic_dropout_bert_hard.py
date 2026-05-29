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
import os
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
"""
Dynamic Dropout MultiModal Baseline with BERT Transformer (Hard Classification):

基于MultiModal_Baseline_Dynamic_Dropout_BERT，统一使用硬分类逻辑：
1. 动态dropout选择层：学习每个token每个维度的dropout概率
2. 训练和评估时统一使用离散dropout（0/1）硬分类
3. Word embedding loss：监督dropout后的embeddings接近理想embeddings
4. 支持token级别文本输入 [B, L, text_emb_dim]
5. 使用BERT Transformer进行上下文编码和CLS提取，替代简单平均池化

实验目标：通过BERT Transformer增强文本表示能力，利用上下文信息提升预测性能。
统一使用硬分类以确保训练和评估时的一致性。
"""


class BertTransformerEncoder(nn.Module):
    """
    BERT Transformer编码器，用于将token embeddings转换为句子级别embedding。
    流程: Token Embedding (4096) -> 投影 (768) -> 添加CLS -> BERT -> 提取CLS -> 投影 (4096)
    """
    def __init__(self, qwen_dim=4096, bert_dim=768, max_length=24, bert_model_name='bert-base-uncased', bert_model_path=None):
        super(BertTransformerEncoder, self).__init__()
        
        # 如果没有指定路径，尝试使用项目默认路径
        if bert_model_path is None:
            default_path = '/ssd/hf_home/models/bert-base-uncased'
            if os.path.exists(default_path):
                bert_model_path = default_path
        
        self.qwen_dim = qwen_dim
        self.bert_dim = bert_dim
        self.max_length = max_length
        
        # 投影层：Qwen 4096 -> BERT 768
        self.projection = nn.Linear(qwen_dim, bert_dim)
        
        # 加载BERT模型
        # 优先使用本地路径，如果提供路径则使用路径，否则尝试使用模型名称（可能从Hugging Face Hub下载）
        if bert_model_path is not None and os.path.exists(bert_model_path):
            # 使用本地路径加载
            self.bert_model = BertModel.from_pretrained(bert_model_path, local_files_only=True)
        else:
            # 尝试从Hugging Face Hub加载
            # 如果网络有问题，transformers会自动使用本地缓存（如果存在）
            try:
                self.bert_model = BertModel.from_pretrained(bert_model_name)
            except Exception as e:
                # 如果下载失败，给出明确的错误提示
                error_msg = (
                    f"无法加载BERT模型 '{bert_model_name}'。\n"
                    f"网络连接失败，请检查网络或预先下载模型。\n"
                    f"解决方案：\n"
                    f"1. 检查网络连接\n"
                    f"2. 预先下载模型到本地，然后在配置文件中设置 bert_model_path 参数\n"
                    f"3. 或者设置环境变量 TRANSFORMERS_CACHE 指向本地缓存目录"
                )
                raise RuntimeError(error_msg) from e
        
        # 冻结BERT参数（不做MLM预训练）
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # CLS token embedding（可学习的，用于序列开头）
        cls_token_init = torch.randn(1, 1, bert_dim) * 0.02  # 小随机初始化
        self.cls_token_embedding = nn.Parameter(cls_token_init)
        
        # 最终投影层：BERT 768 -> 4096
        self.final_projection = nn.Linear(bert_dim, qwen_dim)
        
    def forward(self, word_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_embeddings: [batch_size, seq_len, qwen_dim] token embeddings
            attention_mask: [batch_size, seq_len] attention mask (1 for real tokens, 0 for padding)
        Returns:
            cls_embeddings: [batch_size, qwen_dim] CLS token embeddings after projection to 4096
        """
        batch_size, seq_len, _ = word_embeddings.shape
        
        # 1. 投影到BERT维度
        bert_embeddings = self.projection(word_embeddings)  # [B, L, 768]
        
        # 2. 添加CLS token在序列开头
        cls_tokens = self.cls_token_embedding.expand(batch_size, -1, -1)  # [B, 1, 768]
        sequence_embeddings = torch.cat([cls_tokens, bert_embeddings], dim=1)  # [B, L+1, 768]
        
        # 3. 扩展attention_mask，在开头添加CLS token的位置（CLS总是有效的）
        # attention_mask格式: 1=真实token, 0=padding token
        cls_attention_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([cls_attention_mask, attention_mask], dim=1)  # [B, L+1]
        
        # 4. 通过BERT模型（使用inputs_embeds参数）
        # BERT会自动添加位置编码和LayerNorm，并处理attention_mask屏蔽padding tokens
        bert_outputs = self.bert_model(
            inputs_embeds=sequence_embeddings,
            attention_mask=extended_attention_mask,  # BERT内部会自动转换为attention bias来屏蔽padding
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        bert_output = bert_outputs.last_hidden_state  # [B, L+1, 768]
        
        # 5. 提取CLS token（第一个位置的embedding）
        cls_embedding = bert_output[:, 0, :]  # [B, 768]
        
        # 6. 投影回4096维
        final_embedding = self.final_projection(cls_embedding)  # [B, 4096]
        
        return final_embedding


class MultiModal_Baseline_Dynamic_Dropout_BERT_Hard(nn.Module):
    def __init__(self, configs):
        super(MultiModal_Baseline_Dynamic_Dropout_BERT_Hard, self).__init__()

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

        # BERT Transformer for pooling token embeddings to sentence-level representation
        # max_seq_len硬编码为24
        self.bert_transformer = BertTransformerEncoder(
            qwen_dim=self.text_emb_dim,  # 4096
            bert_dim=768,
            max_length=24,  # 硬编码
            bert_model_name=configs.get('bert_model_name', 'bert-base-uncased'),
            bert_model_path=configs.get('bert_model_path', None)  # 可选的本地模型路径
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

        # Unified hard classification logic for both training and evaluation
        # Use Bernoulli sampling for discrete dropout (0 or 1)
        discrete_mask = torch.bernoulli(dropout_mask)
        dropout_word_embeddings = word_embedding * discrete_mask

        # 使用BERT Transformer替代平均池化，获取句子级别表示 [B, D]
        # 替换原来的平均池化操作（第159-167行）
        if attention_mask is not None:
            # 使用BERT Transformer进行上下文编码和CLS提取
            h_text = self.bert_transformer(dropout_word_embeddings, attention_mask)  # [B, 4096]
        else:
            # 如果没有attention_mask，仍然使用BERT Transformer（BERT内部会处理）
            # 创建一个全1的mask
            dummy_mask = torch.ones(dropout_word_embeddings.shape[:2], 
                                  device=dropout_word_embeddings.device, 
                                  dtype=torch.long)
            h_text = self.bert_transformer(dropout_word_embeddings, dummy_mask)  # [B, 4096]

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
            
            total_loss += self.beta3 * loss_word_emb

        return total_loss

    def get_model_embeddings(self,batch_x,meta_domain,news,ct1,flag):
        dec_out = self.forward(batch_x,meta_domain,news,ct1,flag)
        return {
            'dec_out': dec_out
        }


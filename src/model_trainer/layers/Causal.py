import math  # 导入math库，提供数学常数和函数
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from typing import List  # 从typing库导入List类型，用于类型注解

import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
import torch.nn.functional as F  # 导入PyTorch中用于功能性操作的函数
import torch.fft as fft  # 导入PyTorch的傅里叶变换模块
from einops import reduce, rearrange  # 导入einops库，用于张量重排和降维等操作
from torch import Tensor  # 从PyTorch导入Tensor类型
from torch.nn import Parameter  # 导入Parameter类，用于定义可学习的参数


class TempDisentangler(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims, depth, dropout):
        super().__init__()

        component_dims = output_dims // 2
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims

        self.kernels = kernels

        self.EnvEncoder = nn.ModuleList(
            [nn.Conv1d(input_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.Ent_time = SelfAttention(num_heads=4, in_dim=input_dims, hid_dim= hidden_dims, dropout=dropout)

        #### frequency settings
        self.length = length
        self.num_freqs = (self.length // 2) + 1

        self.Ent_freq_weight = nn.Parameter(torch.empty((self.num_freqs, hidden_dims, hidden_dims), dtype=torch.cfloat))
        self.Ent_freq_bias = nn.Parameter(torch.empty((self.num_freqs, hidden_dims), dtype=torch.cfloat))
        self.reset_parameters()

        self.Ent_dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.Ent_freq_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.Ent_freq_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.Ent_freq_bias, -bound, bound)

    def forward(self, x):  # x: B x T x input_dims

        env_rep = []
        x = x.transpose(1,2)
        for idx, mod in enumerate(self.EnvEncoder):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            env_rep.append(out.transpose(1, 2))  # b t d
        env_rep = reduce(
            rearrange(env_rep, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1,2)

        entity_time = self.Ent_time(x)
        input_freq = fft.rfft(x, dim=1)
        output_freq = torch.zeros(x.size(0), x.size(1) // 2 + 1, self.hidden_dims, device=x.device, dtype=torch.cfloat)
        output_freq[:, :self.num_freqs] = torch.einsum('bti,tio->bto', input_freq[:, :self.num_freqs], self.Ent_freq_weight) + self.Ent_freq_bias
        entity_freq = fft.irfft(output_freq, n=x.size(1), dim = 1)

        entity_rep = torch.add(entity_time, entity_freq)
        entity_rep = self.Ent_dropout(entity_rep)
        return env_rep, entity_rep



class EnvEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # 初始化代码本（codebook）层，K为环境的数量，D为环境向量的维度
        self.embedding = nn.Embedding(K, D)  
        # 初始化权重为均匀分布的值
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, x): # 输入x: [b, l, h_d]，即batch大小b，序列长度l，特征维度h_d
    
        x_ = x.contiguous()  # 确保x在内存中是连续的
        # 使用vq函数将输入x映射到codebook中，得到环境的潜在表示
        latents = vq(x_, self.embedding.weight)
        return latents  # 返回潜在表示

    def straight_through(self, z_e_x):  # x: [b, h_d]，即批次大小b，潜在向量维度h_d
    
        '''
        z_e_x: 代表环境的潜在向量
        '''
  
        #z_e_x.shape:torch.Size([32, 140])
        z_e_x_ = z_e_x.contiguous()  # 确保z_e_x是连续存储的
        # 使用vq_st函数进行straight-through估计，得到离散化的潜在向量和对应的索引
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())  # z_q_x_: 离散化的潜在表示,torch.Size([32, 140]);indices: 对应的索引,torch.Size([32])
        z_q_x = z_q_x_.contiguous()  # 保证z_q_x是连续的

        # 从embedding表中选择与索引对应的向量
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices) # torch.Size([32, 140])
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)  # 恢复形状
        z_q_x_bar = z_q_x_bar_.contiguous()  # 保证是连续存储

        return z_q_x, z_q_x_bar, indices  # 返回离散化的潜在向量、量化的向量以及索引

    def straight_through_test(self, z_e_x):  # 当进行soft index的测试时
        '''
        这个函数执行soft索引操作，即不进行硬选择
        '''

        inputs = z_e_x.contiguous()  # 确保输入是连续存储的
        codebook = self.embedding.weight.detach()  # 从codebook中获取环境向量，detach表示不参与梯度计算

        with torch.no_grad():  # 不需要计算梯度
            embedding_size = codebook.size(1)  # 获取embedding的维度
            inputs_flatten = inputs.view(-1, embedding_size)  # 将输入展平
            codebook_sqr = torch.sum(codebook ** 2, dim=1)  # 计算codebook中每个向量的平方和
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)  # 计算输入向量的平方和
            # 计算输入向量与codebook中所有向量的欧几里得距离
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            # 使用softmax计算距离的概率分布（得到索引）
            indices = torch.softmax(distances, dim=1)  
            # 计算soft索引对应的环境向量
            codes_flatten = torch.mm(indices, codebook)
            codes = codes_flatten.view_as(inputs)  # 恢复到原来的形状

            return codes.contiguous(), None, indices  # 返回soft索引对应的环境向量和对应的索引


import torch
from torch.autograd import Function
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):

        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
 
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
 
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class LayerNorm(nn.Module):
    def __init__(self, hid_dim, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hid_dim))
        self.bias = nn.Parameter(torch.zeros(hid_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_heads, in_dim, hid_dim, dropout):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(hid_dim / num_heads)
        self.hid_dim = hid_dim

        self.query = nn.Linear(in_dim, self.hid_dim)
        self.key = nn.Linear(in_dim, self.hid_dim)
        self.value = nn.Linear(in_dim, self.hid_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hid_dim, hid_dim)
        self.LayerNorm = LayerNorm(hid_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hid_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states





import os
import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from layers.MultiModal import CrossModalAttention

import time as Time
import ipdb


torch.pi = math.pi

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)




class diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config['timesteps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.c = config['c']
        self.embedding_dim = config['embedding_size']
        self.negsample_step = config['negsample_step']
        self.device = config['device']
        if config['beta_sche'] == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif config['beta_sche'] == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x1_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x1_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        

            
        # Multimodal Fusion:
        self.unet = nn.Linear(self.embedding_dim*2,
                         self.embedding_dim, bias=False)
        init(self.unet)
        self.cross_attn = CrossModalAttention(self.embedding_dim,dropout=0.1)
       




    def q_sample(self, x1_start, t, noise=None):
        # print(self.betas)

        if noise is None:
            noise = torch.randn_like(x1_start)
            # noise = torch.randn_like(x1_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x1_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x1_start.shape
        )
        return sqrt_alphas_cumprod_t * x1_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x1_start, x2_start, t, noise=None, loss_type="l2"):
        #

        if noise is None:
            noise_x = torch.randn_like(x1_start)

        x_noisy = self.q_sample(x1_start=x1_start, t=t, noise=noise_x)  ##id

        t_emb = self.get_timestep_embedding(t, self.embedding_dim).to(self.device)
        diff_i = torch.cat((x_noisy,t_emb),dim=1)
        diff_i = self.unet(diff_i)

        predicted_x = self.cross_attn(diff_i,x2_start)
        
        

        #
        if loss_type == 'l1':
            loss = F.l1_loss(x1_start, predicted_x)
        elif loss_type == 'l2':
            loss_x = F.mse_loss(x1_start, predicted_x)

            loss = loss_x 
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x1_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x1_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x1_t.shape) * x1_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x1_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, x1_t, x2_t, t, t_index):

        t = t.to(self.device)

        
        t_emb = self.get_timestep_embedding(t, self.embedding_dim).to(self.device)

        diff_i = torch.cat((x1_t,t_emb),dim=1)
        diff_i = self.unet(diff_i)
        x1_start = self.cross_attn(diff_i,x2_t)


        model_mean_x = (
                extract(self.posterior_mean_coef1, t, x1_t.shape) * x1_start +
                extract(self.posterior_mean_coef2, t, x1_t.shape) * x1_t
        )


        if t_index == 0:
            return model_mean_x
        else:

            posterior_variance_t_x = extract(self.posterior_variance, t, x1_t.shape)
            noise_x = torch.randn_like(x1_t)

            return model_mean_x + torch.sqrt(posterior_variance_t_x) * noise_x

    @torch.no_grad()
    def sample(self, x1_start, x2_start):

        noise_x = torch.randn_like(x1_start)

        t = torch.tensor([self.timesteps-1] * x1_start.shape[0]).to(x1_start.device)
        x1_noisy = self.q_sample(x1_start=x1_start, t=t, noise=noise_x) ## x_noisy x1_start加噪
        
        x1_t = x1_noisy
        x2_t = x2_start

    
   
        
        for n in reversed(range(0, self.timesteps)):
            x1_t = self.p_sample(x1_t,x2_t,
                              torch.full((x1_t.shape[0],), n, dtype=torch.long), n)
    
        return x1_t       
                
            

       

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


    def selfAttention(self, features):

        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)


        attn = q.mul(self.embedding_dim ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v 

        y = features.mean(dim=-2) 

        return y
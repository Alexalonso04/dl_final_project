import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional
import logging


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms"""
    @abstractmethod
    def forward(self, x: torch.Tensor, rel_pos=None, attn_mask=None) -> torch.Tensor:
        pass

class StandardAttention(BaseAttention):
    '''Karpathy's Causal Attention'''
    def __init__(self, config):
        super().__init__()
        self.use_rope = config.use_rope
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.rotary = Rotary(self.n_embd // self.n_head) if self.use_rope else None


    def forward(self, x, rel_pos=None, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q = F.rms_norm(q, (q.size(-1),)).to(v.dtype)
        k = F.rms_norm(k, (k.size(-1),)).to(v.dtype)

        if self.use_rope:
            q = self.rotary(q)
            k = self.rotary(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class DifferentialAttention(BaseAttention):
    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_rope = config.use_rope
        # For DIFF transformer, use half the number of heads since each has 2 components
        self.n_head = config.n_head // 2
        self.n_embd = config.n_embd
        # Each head gets twice the dimension since we have half the heads
        self.head_dim = config.n_embd // (config.n_head // 2) 
        self.layer_idx = layer_idx

        self.rotary = Rotary((self.n_embd // self.n_head) // 2) if self.use_rope else None
        
        # key, query, value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        
        # Layer-dependent lambda initialization
        self.lambda_init = config.diff_lambda_init or 0.8 - 0.6 * math.exp(-0.3 * (self.layer_idx+1 - 1))
        
        # Initialize lambda parameters for reparameterization
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))


    def forward(self, x):
        B, T, C = x.size()

        # Compute lambda scalar 
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).to(x.dtype)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).to(x.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        # Compute the query, key, and value projections
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # Split heads and reshape
        q = q.view(B, T, self.n_head, 2, self.head_dim // 2).transpose(1, 2)  # (B, nh, T, 2, hs/2)
        k = k.view(B, T, self.n_head, 2, self.head_dim // 2).transpose(1, 2)  # (B, nh, T, 2, hs/2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)          # (B, nh, T, hs)
        
        # Split queries and keys for differential attention
        q1 = q[..., 0, :]  # (B, nh, T, hs/2)
        q2 = q[..., 1, :]  # (B, nh, T, hs/2)
        k1 = k[..., 0, :]  # (B, nh, T, hs/2)
        k2 = k[..., 1, :]  # (B, nh, T, hs/2)

        # # Apply RMS norm to queries and keys
        q1 = F.rms_norm(q1, (q1.size(-1),)).to(v.dtype)
        q2 = F.rms_norm(q2, (q2.size(-1),)).to(v.dtype)
        k1 = F.rms_norm(k1, (k1.size(-1),)).to(v.dtype)
        k2 = F.rms_norm(k2, (k2.size(-1),)).to(v.dtype)

        if self.use_rope:
            q1 = self.rotary(q1)
            k1 = self.rotary(k1)
            q2 = self.rotary(q2)
            k2 = self.rotary(k2)
        
        with torch.amp.autocast('cuda', enabled=False):
            # Compute attention with flash attention
            y = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
            y = y - lambda_full * F.scaled_dot_product_attention(q2, k2, v, is_causal=True)
        
         # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B,T,C)  # (B, T, nh, hs)
        y = self.out_proj(y)
        
        # Scale output
        y = y * (1 - self.lambda_init)
        
        return y

class SelectiveAttention(BaseAttention):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, rel_pos=None, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # TODO: Old imp. is too slow to run on 1xH100 in any reasonable amount of time. ~1 step / 10k+ ms
        # scaled_dot_product_attention is here as a placeholder.
        print("NOPE.avi")

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

def create_attention(attention_type: str, config, layer_idx: Optional[int] = None) -> BaseAttention:
    """Factory function to create attention mechanisms"""
    if attention_type == 'base':
        return StandardAttention(config)
    elif attention_type == 'differential':
        return DifferentialAttention(config, layer_idx)
    elif attention_type == 'selective':
        return SelectiveAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
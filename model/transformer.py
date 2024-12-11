import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import logging
from typing import Optional, Tuple
from model.attention import create_attention
from model.optimizers import Muon

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_relu_square = config.use_relu_square
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() if self.use_relu_square else self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.model_type == 'differential':
            self.attn = create_attention(config.model_type, config, layer_idx)
            self.head_dim = config.n_embd // config.n_head // 2
        else:
            self.attn = create_attention(config.model_type, config)

    def forward(self, x):
        x = x + self.dropout(self.attn(norm(x)))
        x = x + self.dropout(self.mlp(norm(x)))
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.model.vocab_size is not None
        assert config.model.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.model.vocab_size, config.model.n_embd),
            wpe = nn.Embedding(config.model.block_size, config.model.n_embd),
            drop = nn.Dropout(config.model.dropout),
            h = nn.ModuleList([Block(config.model, layer_idx) for layer_idx in range(config.model.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.model.n_embd, config.model.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        logging.debug(f"Number of parameters: {n_params/1e6:.1f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = std * (2.0 * self.config.model.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.model.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.model.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if self.config.model.clamp_logits:
            logits = 30 * torch.tanh(logits / 30) # @Grad62304977
            logits = logits.float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        
        if hasattr(self.config, 'optimizer') and self.config.optimizer.type == 'muon':
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"
            
            # 1. Embedding optimizer
            embedding_opt = torch.optim.Adam(
                [self.transformer.wte.weight],
                lr=0.6,
                betas=(0.8, 0.95),
                fused=use_fused,
                weight_decay = weight_decay
            )
            
            # 2. LM head optimizer
            lm_head_opt = torch.optim.Adam(
                [self.lm_head.weight],
                lr=0.008,
                betas=(0.8, 0.95),
                fused=use_fused,
                weight_decay = weight_decay
            )
            
            # 3. Split transformer block parameters
            muon_params = []
            other_params = []
            
            for name, param in self.transformer.h.named_parameters():
                if param.dim() == 2:  # Only 2D tensors go to Muon
                    muon_params.append(param)
                else:
                    other_params.append(param)
            
            # Create Muon optimizer for 2D params
            muon_opt = Muon(
                muon_params,
                lr=0.05,
                momentum=self.config.optimizer.momentum,
                nesterov=self.config.optimizer.nesterov,
                ns_steps=self.config.optimizer.backend_steps
            )
            
            # Create Adam for remaining transformer params (biases etc)
            if other_params:  # Only create if we have params to optimize
                other_opt = torch.optim.Adam(
                    other_params,
                    lr=0.04, 
                    betas=(0.8, 0.95),
                    fused=use_fused,
                    weight_decay= weight_decay
                )
                return [embedding_opt, lm_head_opt, muon_opt, other_opt]
                
            return [embedding_opt, lm_head_opt, muon_opt]
        
        else:
            # Original AdamW logic
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            optim_groups = [
                {'params': [p for p in param_dict.values() if p.dim() >= 2], 'weight_decay': weight_decay},
                {'params': [p for p in param_dict.values() if p.dim() < 2], 'weight_decay': 0.0}
            ]
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused, weight_decay=weight_decay)
            return optimizer
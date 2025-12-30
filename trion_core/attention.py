import torch
import torch.nn as nn
import math
from .layers import BitGhostLinear

# CUDA Engine Import
try:
    from .engine import attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class GhostAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.attn_threshold = getattr(config, 'attn_threshold', -0.1)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = BitGhostLinear(self.d_model, self.d_model)
        self.k_proj = BitGhostLinear(self.d_model, self.d_model)
        self.v_proj = BitGhostLinear(self.d_model, self.d_model)
        self.o_proj = BitGhostLinear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if CUDA_AVAILABLE and x.is_cuda and not self.training:
            target_dtype = torch.float16
        else:
            target_dtype = x.dtype
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous().to(target_dtype)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous().to(target_dtype)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous().to(target_dtype)

        if CUDA_AVAILABLE and x.is_cuda and not self.training:
            out = torch.empty_like(q)
            L = torch.empty((B, self.n_head, T), dtype=torch.float32, device=x.device)
            mask_tensor = torch.empty(0, dtype=target_dtype, device=x.device)
            attention_cuda.forward(q, k, v, out, L, mask_tensor, self.scale, True, self.attn_threshold)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            if out.dtype != x.dtype: out = out.to(x.dtype)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.relu(scores - self.attn_threshold)
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-6
            attn_weights = attn_weights / attn_sum
            out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)

        return self.o_proj(out)

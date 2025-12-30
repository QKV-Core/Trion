import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .quantization import quantize_158bit
except ImportError:
    # Yedek (Dosya bulunamazsa diye)
    def quantize_158bit(x):
        scale = x.abs().mean().clamp(min=1e-5)
        return (x / scale).round().clamp(-1, 1) * scale

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class BitGhostLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w_quant = quantize_158bit(self.weight)
        return F.linear(x, w_quant, self.bias)

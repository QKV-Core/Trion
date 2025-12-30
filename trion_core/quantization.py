import torch

class BitNetQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        scale = x.abs().mean().clamp(min=1e-5)
        x_quant = (x / scale).round().clamp(-1, 1)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize_158bit(x):
    return BitNetQuantizer.apply(x)

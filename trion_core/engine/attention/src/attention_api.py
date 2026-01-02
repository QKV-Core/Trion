import torch
# Doğrudan derlenmiş modülden çekiyoruz
from ...src import attention_cuda

class GhostAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, mask=None, causal=True, scale=None, dropout_p=0.0, attn_threshold=0.0):
        # Güvenlik Kontrolleri
        if torch.isnan(Q).any() or torch.isnan(K).any() or torch.isnan(V).any():
             raise ValueError("Kernel received NaN Input!")

        if scale is None:
            scale = 1.0 / (Q.size(-1) ** 0.5)
            
        # Contiguous Zorunluluğu
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
            
        B, H, T, D = Q.shape
        Out = torch.empty_like(Q)
        L = torch.empty((B, H, T), dtype=torch.float32, device=Q.device)
        
        if mask is None:
            mask_tensor = torch.empty(0, device=Q.device, dtype=Q.dtype)
        else:
            mask_tensor = mask.contiguous()

        # C++ KERNEL ÇAĞRISI (THRESHOLD EKLENDİ)
        # Not: C++ tarafındaki wrapper, 'has_mask' boolean'ını mask.numel() ile kendi çözer.
        attention_cuda.forward(
            Q, K, V, Out, L, mask_tensor, 
            scale, 
            causal, 
            attn_threshold # <--- İŞTE BU EKSİKTİ
        )
        
        # Backward için kaydet
        ctx.save_for_backward(Q, K, V, Out, L, mask_tensor)
        ctx.scale = scale
        ctx.causal = causal
        ctx.attn_threshold = attn_threshold
        
        return Out

    @staticmethod
    def backward(ctx, dO):
        # Backward şu an Ghost Modunda aktif değil (Eğitim yok)
        # Ancak API hatası vermemesi için boş gradientler döndürelim.
        Q, K, V, Out, L, Mask = ctx.saved_tensors
        return torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V), None, None, None, None, None

def ghost_attention(q, k, v, mask=None, causal=True, scale=None, dropout_p=0.0, attn_threshold=0.0):
    return GhostAttentionFunction.apply(q, k, v, mask, causal, scale, dropout_p, attn_threshold)
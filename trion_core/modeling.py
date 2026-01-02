import torch
import torch.nn as nn
import torch.nn.functional as F
from .config.model_config import GhostConfig


# ------------------------------------------------------------
# STANDARD LINEAR (FLOAT / REFERANS)
# ------------------------------------------------------------
class BitGhostLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ------------------------------------------------------------
# SIGN + MASK LINEAR (TERNARY Q-ONLY)
# ------------------------------------------------------------
class SignMaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        w = torch.empty(out_features, in_features)
        nn.init.uniform_(w, -1.0, 1.0)
        self.weight = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        sign = torch.sign(self.weight)
        mask = (self.weight != 0).to(x.dtype)
        return F.linear(x, sign * mask, self.bias)


# ------------------------------------------------------------
# ATTENTION (Q-ONLY + STEP-3B + STEP-3E + HEAD STATS)
# ------------------------------------------------------------
class GhostAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head

        self.q_proj = SignMaskLinear(self.d_model, self.d_model)
        self.k_proj = BitGhostLinear(self.d_model, self.d_model)
        self.v_proj = BitGhostLinear(self.d_model, self.d_model)
        self.o_proj = BitGhostLinear(self.d_model, self.d_model)

        self.register_buffer(
            "head_keep_ratio",
            torch.linspace(0.35, 0.15, self.n_head),
            persistent=False
        )

        self.register_buffer("min_keep_heads", torch.tensor(2), persistent=False)
        self.register_buffer("max_keep_heads", torch.tensor(6), persistent=False)

        # 🔍 HEAD ENTROPY STATS (OBSERVATION ONLY)
        self.last_head_stats = None

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Q RMS
        q = q / (q.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)

        # -------------------------------------------------
        # STEP-3B: Q CHANNEL SPARSITY (ALIAS SAFE)
        # -------------------------------------------------
        q_abs = q.abs()
        keep_counts = (self.head_keep_ratio * self.head_dim).long().clamp(min=1)

        q_mask = torch.zeros_like(q)

        for h in range(self.n_head):
            k_keep = keep_counts[h].item()
            thresh = torch.topk(
                q_abs[:, h], k_keep, dim=-1
            ).values[..., -1].unsqueeze(-1)
            q_mask[:, h] = (q_abs[:, h] >= thresh)

        q = q * q_mask

        # K RMS
        k = k / (k.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)

        # Attention
        attn_logits = q @ k.transpose(-2, -1)

        causal = torch.tril(
            torch.ones(T, T, device=x.device)
        ).view(1, 1, T, T)

        attn_logits = attn_logits.masked_fill(causal == 0, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)

        # -------------------------------------------------
        # STEP-3E: ENTROPY-AWARE HEAD GATING
        # -------------------------------------------------
        head_energy = (attn ** 2).mean(dim=(2, 3))          # [B, H]
        head_prob = F.softmax(head_energy, dim=1)
        head_entropy = -(head_prob * (head_prob + 1e-9).log()).sum(dim=1)  # [B]

        max_ent = torch.log(torch.tensor(float(self.n_head), device=x.device))
        ent_norm = (head_entropy / max_ent).clamp(0, 1)

        keep_h = (
            self.min_keep_heads +
            ent_norm * (self.max_keep_heads - self.min_keep_heads)
        ).round().long()

        head_mask = torch.zeros(B, self.n_head, device=x.device)
        for b in range(B):
            k_h = keep_h[b].item()
            thresh = torch.topk(head_energy[b], k_h).values[-1]
            head_mask[b] = (head_energy[b] >= thresh).float()

        head_mask = head_mask.view(B, self.n_head, 1, 1)

        # 🔍 SAVE HEAD STATS (NO GRAPH)
        with torch.no_grad():
            self.last_head_stats = {
                "head_entropy_mean": head_entropy.mean().item(),
                "head_entropy_std": head_entropy.std().item(),
                "active_head_count": head_mask.sum(dim=1).mean().item(),
            }

        attn = attn * head_mask
        v = v * head_mask

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ------------------------------------------------------------
# MLP
# ------------------------------------------------------------
class GhostMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.d_model
        self.fc1 = BitGhostLinear(config.d_model, hidden)
        self.fc2 = BitGhostLinear(hidden, config.d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------------------------------------
# BLOCK
# ------------------------------------------------------------
class GhostBlock(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = GhostAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = GhostMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ------------------------------------------------------------
# MODEL (CLEAN — NO LOGIT MASKING)
# ------------------------------------------------------------
class GhostModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_seq_len + 2, config.d_model
        )

        self.layers = nn.ModuleList(
            [GhostBlock(config, i) for i in range(config.n_layer)]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = BitGhostLinear(
            config.d_model, config.vocab_size, bias=False
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device) + 2

        x = self.token_embedding(input_ids) + self.position_embedding(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output_head(x)

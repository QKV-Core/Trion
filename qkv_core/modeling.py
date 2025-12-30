import torch
import torch.nn as nn
from .layers import BitGhostLinear, RMSNorm
from .attention import GhostAttention
from .config.model_config import GhostConfig

class GhostMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_dim = 4 * config.d_model
        self.fc1 = BitGhostLinear(config.d_model, hidden_dim)
        self.fc2 = BitGhostLinear(hidden_dim, config.d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GhostBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = GhostAttention(config) 
        self.ln2 = RMSNorm(config.d_model)
        self.mlp = GhostMLP(config, layer_idx)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GhostModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([GhostBlock(config, i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for layer in self.layers: x = layer(x)
        return self.output_head(self.norm(x))

QKVModel = GhostModel
QKVConfig = GhostConfig

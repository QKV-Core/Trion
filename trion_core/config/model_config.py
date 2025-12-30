from dataclasses import dataclass, asdict
import json
import os

@dataclass
class GhostConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 4
    max_seq_len: int = 256
    ternary_threshold: float = 0.5
    attn_threshold: float = -0.1
    seed: int = 1337
    dropout_p: float = 0.0

    @classmethod
    def load(cls, path):
        if not os.path.exists(path): return cls()
        with open(path, 'r') as f: return cls(**json.load(f))

def get_shakespeare_config():
    return GhostConfig(vocab_size=65, d_model=256, n_layer=6, attn_threshold=-0.1)

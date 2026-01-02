# config/model_config.py
from dataclasses import dataclass
import json
import os

@dataclass
class GhostConfig:
    # ------------------------
    # CORE GEOMETRY
    # ------------------------
    vocab_size: int = 50272          # OPT-125M default
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    max_seq_len: int = 2048

    # ============================
    # GHOST CORE PARAMETERS
    # ============================
    qk_keep: float = 0.6          # 🔥 TERNARY CHAMPION
    qk_mode: str = "ternary"      # "float" | "ternary"
    energy_fix: bool = True       # std energy clamp

    # Runtime
    brain_path: str = "trion_brain_opt_ghost.pt"
    tokenizer_name: str = "facebook/opt-125m"

    # Future hooks (BitNet 1.58 / 1.3)
    bit_width: float = 1.58       # informational (yet)
    
    # ------------------------
    # GHOST CORE PARAMETERS
    # ------------------------
    ternary_threshold: float = 0.7   # τ : silence threshold
    attn_threshold: float = -0.1     # Phase-2 attention cutoff
    dropout_p: float = 0.0           # kept for future, inactive now
    seed: int = 1337


    # ------------------------
    # RUNTIME / EXPERIMENT FLAGS
    # ------------------------
    use_ternary: bool = True
    bias_free: bool = True
    embedding_mode: str = "dense"    # "dense" | "ghost"
    output_head_mode: str = "float"  # "float" | "ternary"

    # ------------------------
    # IO
    # ------------------------
    brain_path: str = "trion_brain_opt_ghost.pt"
    tokenizer_name: str = "facebook/opt-125m"

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

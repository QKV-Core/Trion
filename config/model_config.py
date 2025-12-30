from dataclasses import dataclass, asdict
import json
import os

@dataclass
class GhostConfig:
    """
    QKVCore Motorunun Merkezi Konfigürasyonu.
    Tüm hiperparametreler buradan yönetilir.
    """
    # --- Model Mimarisi ---
    vocab_size: int = 65       # Shakespeare için 65, ASCII için 256, GPT için 50257
    d_model: int = 256         # Model genişliği (Embedding boyutu)
    n_layer: int = 6           # Derinlik (Blok sayısı)
    n_head: int = 4            # Attention kafa sayısı
    max_seq_len: int = 256     # Context Window (Ne kadar geriyi hatırlıyor)
    
    # --- Ghost Engine Ayarları (Kritik) ---
    ternary_threshold: float = 0.5  # Ağırlıkların %50'si 0 olsun (Sparsity)
    attn_threshold: float = -0.1    # Shifted ReLU eşiği (Negatif olması şart!)
    
    # --- Diğer ---
    seed: int = 1337           # Deterministik sonuçlar için
    dropout_p: float = 0.0     # Ghost zaten sparse, dropout genelde 0 kalır

    def save(self, path):
        """Konfigürasyonu JSON olarak kaydeder."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, path):
        """JSON dosyasından konfigürasyonu yükler."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# --- Hazır Reçeteler (Presets) ---

def get_tiny_config():
    """Hızlı test ve debug için minimalist ayarlar."""
    return GhostConfig(
        d_model=64,
        n_layer=2,
        n_head=2,
        max_seq_len=64
    )

def get_shakespeare_config():
    """GTX 1050 üzerinde Shakespeare eğitimi için optimize edilmiş ayarlar."""
    return GhostConfig(
        vocab_size=65,
        d_model=256,
        n_layer=6,
        n_head=4,
        max_seq_len=256,
        attn_threshold=-0.1
    )

def get_base_config():
    """Daha güçlü donanımlar için standart ayarlar."""
    return GhostConfig(
        d_model=512,
        n_layer=8,
        n_head=8,
        max_seq_len=512
    )
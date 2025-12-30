import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostEmbedding(nn.Module):
    """
    Ghost Embedding:
    Standart 'Lookup Table' yerine, Seed-Based Deterministik Ternary Vektörler.
    
    Mantık:
    - Kelimelerin vektörleri eğitilmez (Frozen).
    - Her kelime, kendi ID'sine bağlı bir seed ile oluşturulmuş sparse bir imzadır.
    - Token 42 -> Seed(Global + 42) -> Ternary Vector.
    """
    def __init__(self, num_embeddings, embedding_dim, sparsity=0.5, seed=42):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sparsity = sparsity
        self.seed = seed
        
        # 1. Deterministik Oluşturucu
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        # 2. Sanal Ağırlıklar (Latent Weights)
        w_raw = torch.randn(num_embeddings, embedding_dim, generator=g)
        
        # 3. Sparsity Mask (Gürültü Kesici)
        # Belirli oranda 0 atıyoruz. (Sessizlik)
        threshold = torch.quantile(torch.abs(w_raw), sparsity)
        mask = (torch.abs(w_raw) > threshold).float()
        
        # 4. Sign Extraction
        signs = torch.sign(w_raw)
        
        # 5. Final Ternary Weights {-1, 0, +1}
        w_ternary = signs * mask
        
        # 6. Dondur (Eğitim Yok)
        self.weight = nn.Parameter(w_ternary, requires_grad=False)

    def forward(self, input_ids):
        # Standart lookup, ama ternary weight ile.
        return F.embedding(input_ids, self.weight)

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, sparsity={self.sparsity}'
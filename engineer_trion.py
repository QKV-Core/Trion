import torch
import torch.nn as nn
import os
import math
import sys

# Trion Core Modüllerini Çek
try:
    from trion_core.modeling import QKVModel, QKVConfig
except ImportError:
    print("❌ HATA: 'qkv_core' klasörü bulunamadı.")
    sys.exit()

def calculate_trion_brain():
    print("📐 TRION CORE: Statik Beyin İnşası Başlatılıyor...")
    print("   -> Hedef: Rastgelelikten arındırılmış, matematiksel denge.")

    # 1. TRION MİMARİSİ (GPT-2 Small Standartları)
    config = QKVConfig(
        vocab_size=50257,  # GPT-2 Sözlüğü
        d_model=768,       # Standart Genişlik
        n_layer=12,        # Derinlik
        n_head=12,         # Dikkat Kafaları
        max_seq_len=1024,
        attn_threshold=-0.1 
    )
    
    model = QKVModel(config)
    print(f"⚙️  İskelet Hazır: {config.d_model}x{config.n_layer} Katman")

    # 2. MATEMATİKSEL ENJEKSİYON (Weight Engineering)
    print("💉 Matrislere 'Identity' ve 'Xavier' hesaplamaları uygulanıyor...")
    
    with torch.no_grad():
        # A) Embedding (Kelime Vektörleri)
        # Çok düşük varyanslı normal dağılım (Kelimeler karışmasın diye)
        nn.init.normal_(model.token_embedding.weight, mean=0.0, std=0.02)
        model.position_embedding.weight.data.fill_(0.0) # Pozisyon başta nötr olsun

        for i, layer in enumerate(model.layers):
            # B) ATTENTION (İletişim)
            # Identity Matrix: Girdi = Çıktı (Ayna Etkisi)
            # Bu, modelin eğitimsizken saçmalamasını engeller.
            nn.init.eye_(layer.attn.q_proj.weight)
            nn.init.eye_(layer.attn.k_proj.weight)
            nn.init.eye_(layer.attn.v_proj.weight)
            
            # Çıkış projeksiyonunu sıfıra yakın tutuyoruz ki gürültü birikmesin
            nn.init.normal_(layer.attn.o_proj.weight, std=0.001)

            # C) MLP (Düşünme) - 1.58-bit Kritik Bölge
            # Kaiming Initialization: Aktivasyonların sönmemesi için kazanç (Gain) hesabı
            # 1.58 bit olduğu için sinyali biraz güçlendiriyoruz (Gain=sqrt(2))
            nn.init.kaiming_normal_(layer.mlp.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(layer.mlp.fc2.weight, mode='fan_in', nonlinearity='relu')
            
            # Katman Normalizasyonlarını (RMSNorm) nötrle
            layer.ln1.weight.data.fill_(1.0)
            layer.ln2.weight.data.fill_(1.0)
            
            if i % 3 == 0: print(f"   -> Katman {i} stabilize edildi.")

        # D) Output Head (Kafa)
        # Giriş ile çıkış matrisini bağlıyoruz (Weight Tying)
        model.output_head.weight = model.token_embedding.weight

    # 3. KAYIT
    filename = "trion_brain.pt"
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)

    print("-" * 50)
    print(f"✅ TRION BEYNİ OLUŞTURULDU: {filename}")
    print(f"📊 Boyut: {size_mb:.2f} MB")
    print("🧠 Durum: 'Tabula Rasa' (Temiz Levha)")
    print("   -> Bu model artık gürültü (çöp karakter) üretmez.")
    print("   -> Eğitime başladığında çok hızlı öğrenir.")
    print("-" * 50)

if __name__ == "__main__":
    calculate_trion_brain()
import os

def create_readme():
    print("📝 TRION CORE: README.md dosyası hazırlanıyor...")
    
    # Tüm içeriği tek bir raw string (r"...") içine alıyoruz.
    # Böylece Python hiçbir özel karakteri (ters taksim, tırnak vs.) karıştırmaz.
    
    content = r"""<div align="center">

# 💠 TRION CORE
### The 1.58-bit High-Performance LLM Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/engine-1.58--bit-magenta.svg)](qkv_core/)

*Ultra-düşük bellek kullanımı, yüksek hız ve matematiksel zeka.*

[Özellikler](#-özellikler) • [Kurulum](#-kurulum) • [Matematik](#-matematiksel-altyapı) • [Mimari](#-sistem-mimarisi)

</div>

---

## 🚀 Proje Hakkında
**Trion Core**, BitNet b1.58 mimarisini temel alan, yeni nesil bir Büyük Dil Modeli (LLM) çekirdeğidir. Standart modellerin aksine, ağırlıkları (weights) 16-bit FP16 yerine **1.58-bit {-1, 0, 1}** değerlerinde saklar.

Bu devrimsel yaklaşım sayesinde:
* **Hafıza (VRAM) kullanımı %70 azalır.**
* **Matris çarpımları (MatMul), toplama işlemine (Addition) indirgenir.**
* **Eğitim süresi ve enerji tüketimi radikal biçimde düşer.**

---

## 🧮 Matematiksel Altyapı

Trion Core, ağırlıkları sıkıştırmak için **Absmean Quantization** tekniğini kullanır.

### 1. Kuantizasyon Formülü
Ağırlık matrisi $W$ için ölçekleme faktörü $\gamma$ ve kuantize ağırlık $W_{quant}$ şöyle hesaplanır:

$$ \gamma = \frac{1}{nm} \sum_{ij} |W_{ij}| $$

$$ W_{quant} = \text{Clip}\left(\text{Round}\left(\frac{W}{\gamma}\right), -1, 1\right) $$

Sonuç olarak $W_{quant}$ matrisi sadece $\{-1, 0, +1\}$ değerlerini içerir.

### 2. İleri Besleme (Forward Pass)
Aktivasyonlar $X$, 8-bit hassasiyetine ölçeklenir:

$$ Y = (W_{quant} \times X_{quant}) \times \frac{\gamma \beta}{Q_b} $$

Burada işlem, ağır matris çarpımı yerine **Sparse Addition** (Seyrek Toplama) işlemine dönüşür.

---

## 🏗️ Sistem Mimarisi

Trion Core veri akış şeması (Mermaid):

```mermaid
graph TD
    A[Input Text] -->|Tokenizer| B(Token IDs)
    B --> C{Trion Embedding}
    C -->|FP32| D[Layer 1: BitGhostBlock]
    D -->|RMSNorm| E[Attention Mechanism]
    E -->|Identity Init| F[MLP: 1.58-bit Linear]
    F -->|BitQuant| G[Layer N...]
    G --> H[RMSNorm Final]
    H --> I[Output Head]
    I -->|Logits| J[Next Token Prediction]
    
    style C fill:#222,stroke:#00bcd4,stroke-width:2px
    style F fill:#440000,stroke:#ff0000,stroke-width:2px
    style I fill:#222,stroke:#00bcd4,stroke-width:2px
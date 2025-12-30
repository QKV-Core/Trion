<div align="center">

# 💠 TRION CORE
### The 1.58-bit High-Performance LLM Engine

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)](https://www.python.org/)
[![Engine](https://img.shields.io/badge/engine-1.58--bit-magenta.svg)](qkv_core/)

*Ultra-low memory footprint, high inference speed, and mathematical precision.*

[Features](#-features) • [Installation](#-installation) • [Mathematics](#-mathematics) • [Architecture](#-architecture)

</div>

---

## 🚀 About the Project
**Trion Core** is a next-generation Large Language Model (LLM) engine based on the BitNet b1.58 architecture. Unlike standard models, it stores weights in **1.58-bit {-1, 0, 1}** precision instead of 16-bit FP16.

This revolutionary approach enables:
* **70% Reduction** in VRAM/Memory usage.
* **Matrix Multiplications (MatMul)** are simplified into **Additions**.
* **Significantly lower** training time and energy consumption.

---

## 🧮 Mathematics

Trion Core utilizes **Absmean Quantization** to compress weights into ternary values.

### 1. Quantization Formula
For a weight matrix $W$, the scaling factor $\gamma$ and quantized weights $W_{quant}$ are calculated as:

$$\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|$$

$$W_{quant} = \text{Clip}\left(\text{Round}\left(\frac{W}{\gamma}\right), -1, 1\right)$$

The resulting $W_{quant}$ matrix contains only $\\{-1, 0, +1\\}$.

### 2. Forward Pass
Activations $X$ are scaled to 8-bit precision, transforming the operation into:

$$Y = (W_{quant} \times X_{quant}) \times \frac{\gamma \beta}{Q_b}$$

Heavy matrix multiplications are replaced by **Sparse Additions**, dramatically boosting performance on consumer hardware like the GTX 1050.

---

## Roadmap
- v2.0: Trainable ternary weights (STE)
- Activation quantization
- KV-cache optimized inference
- Larger-scale dataset experiments

## 🏗️ Architecture

System data flow visualized (GitHub Mermaid integration):

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
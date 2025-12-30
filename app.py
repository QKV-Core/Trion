import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
# Transformers kütüphanesinden gerekli parçalar
try:
    from transformers import GPT2Tokenizer
except ImportError:
    st.error("Lütfen 'pip install transformers' komutunu çalıştırın.")
    st.stop()

# --- IMPORTLAR ---
try:
    from qkv_core.modeling import QKVModel, QKVConfig
    try:
        from qkv_core.engine import attention_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
except ImportError as e:
    st.error(f"⚠️ Kritik Hata: {e}")
    st.stop()

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="QKV Core (GPT-2 Edition)", page_icon="🧠", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    .stTextInput > div > div > input { background-color: #2d2d2d; color: #fff; border: 1px solid #444; }
    section[data-testid="stSidebar"] { background-color: #1e1e1e; }
    .stButton > button { background-color: #0d47a1; color: white; border: none; width: 100%; }
    .stButton > button:hover { background-color: #1565c0; }
</style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- KAYNAK YÜKLEYİCİ ---
@st.cache_resource
def load_resources():
    # 1. GPT-2 TOKENIZER
    try:
        # Hata vermemesi için model_max_length'i çok yüksek tutuyoruz
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 1000000 # Limit Bypass
    except Exception as e:
        st.error(f"Tokenizer hatası: {e}")
        st.stop()
    
    # 2. MODEL YAPILANDIRMASI (GPT-2 Small)
    config = QKVConfig(
        vocab_size=50257,  
        d_model=768,       
        n_layer=12,        
        n_head=12,         
        max_seq_len=1024,
        attn_threshold=-0.1 
    )
    model = QKVModel(config).to(device)
    
    status = "🆕 BOŞ (GPT-2 Bekleniyor)"
    
    # 3. BEYİN YÜKLEME
    if os.path.exists("ghost_brain.pt"):
        try:
            state_dict = torch.load("ghost_brain.pt", map_location=device)
            model.load_state_dict(state_dict)
            status = "🧠 GPT-2 1.58b (Aktif)"
        except Exception as e:
            status = "⚠️ Model Uyumsuz"
            st.warning("Model dosyası uyumsuz, lütfen convert işlemini tekrar yapın.")
            
    return tokenizer, model, status

tokenizer, model, status_msg = load_resources()

# --- YAN MENÜ ---
st.sidebar.title("🎛️ QKV (GPT-2 CORE)")
st.sidebar.caption(f"Durum: {status_msg}")
st.sidebar.metric("Motor", "1.58-BIT HYBRID" if CUDA_AVAILABLE else "PYTHON MODE")

st.sidebar.header("🔪 Cerrahi Ayarlar")
current_thresh = st.sidebar.slider("Sparsity Threshold", -2.0, 2.0, -0.1, 0.1)
model.config.attn_threshold = current_thresh

st.sidebar.header("🌊 Üretim")
temp = st.sidebar.slider("Temperature", 0.1, 2.0, 0.6)
max_len = st.sidebar.slider("Uzunluk", 50, 500, 100)

# --- ANA EKRAN ---
tab_chat, tab_train = st.tabs(["💬 SOHBET", "🎓 İNCE AYAR (Fine-Tuning)"])

# ==========================================
# SEKME 1: SOHBET
# ==========================================
with tab_chat:
    st.subheader("🤖 QKV x GPT-2 Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Sistem hazır. Veri yükleyip eğitebilirsin."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Mesaj yaz..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            res_box = st.empty()
            full_res = ""
            
            model.eval()
            if CUDA_AVAILABLE and device == 'cuda': model.half()
            
            # Context Window Koruması
            try:
                tokens = tokenizer.encode(prompt)
                # Eğer girdi çok uzunsa son 100 tokenı al
                if len(tokens) > 100: tokens = tokens[-100:]
                idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    for i in range(max_len):
                        # GPT-2 limiti 1024, taşmamak için son 1000'i al
                        cond_idx = idx[:, -1000:] 
                        
                        logits = model(cond_idx)
                        next_logits = logits[:, -1, :]
                        
                        probs = torch.softmax(next_logits.float() / temp, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        idx = torch.cat((idx, next_token), dim=1)
                        
                        token_id = next_token.item()
                        word = tokenizer.decode([token_id])
                        
                        full_res += word
                        res_box.markdown(full_res + "▌")
                
                res_box.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
            except Exception as e:
                st.error(f"Üretim hatası: {e}")

# ==========================================
# SEKME 2: EĞİTİM (Düzeltildi)
# ==========================================
with tab_train:
    st.subheader("🎓 Lobotomi Rehabilitasyonu")
    st.info("GPT-2'yi 1.58-bit'e sıkıştırdığımız için şu an 'sarhoş' gibi olabilir. Ona metin okutarak zekasını geri kazandırabilirsin.")
    
    uploaded_file = st.file_uploader("Eğitim Verisi (.txt)", type=["txt"])
    
    col1, col2 = st.columns(2)
    steps = col1.number_input("Adım Sayısı", 10, 5000, 100)
    lr_input = col2.number_input("Öğrenme Hızı", 1e-6, 1e-3, 1e-5, format="%.6f")
    
    if st.button("🔥 EĞİTİMİ BAŞLAT"):
        if uploaded_file:
            try:
                text = uploaded_file.read().decode("utf-8")
                st.caption(f"Veri Okundu: {len(text)} karakter. İşleniyor...")
                
                # --- DÜZELTME: Tokenizer Limit Bypass ---
                # Hata veren kısım burasıydı. truncation=False diyerek
                # "Ne kadar uzun olursa olsun hepsini sayıya çevir" diyoruz.
                full_tokens = tokenizer.encode(text, truncation=False)
                
                st.caption(f"Token Sayısı: {len(full_tokens)} (Limit aşıldı ama devam ediliyor ✅)")
                
                model.train()
                model.float() # FP32 Hassasiyeti
                
                data = torch.tensor(full_tokens, dtype=torch.long)
                optimizer = optim.AdamW(model.parameters(), lr=lr_input)
                loss_fn = nn.CrossEntropyLoss()
                
                chart_loss = st.line_chart([])
                progress = st.progress(0)
                
                losses = []
                batch_size = 4 # Bellek dostu olması için küçük batch
                
                start_time = time.time()
                
                for i in range(steps):
                    # Veri setinden rastgele küçük bir parça (128 token) alıyoruz
                    # Bu sayede 4324 tokenlık veriyi parça parça yiyoruz.
                    ix = torch.randint(len(data) - 128, (batch_size,))
                    x_batch = torch.stack([data[j:j+128] for j in ix]).to(device)
                    y_batch = torch.stack([data[j+1:j+129] for j in ix]).to(device)
                    
                    logits = model(x_batch)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if i % 5 == 0:
                        chart_loss.line_chart(losses)
                        progress.progress((i+1)/steps)
                
                st.success(f"✅ İnce Ayar Tamamlandı! ({time.time() - start_time:.1f}s)")
                torch.save(model.state_dict(), "ghost_brain.pt")
                st.toast("Beyin Güncellendi!")
                
            except Exception as e:
                st.error(f"Eğitim Hatası: {e}")
                st.write("İpucu: Metin dosyası çok mu büyük? Daha küçük bir dosyayla dene.")
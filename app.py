import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# --- GEREKSİNİMLER ---
try:
    from transformers import GPT2Tokenizer
    from trion_core.modeling import QKVModel, QKVConfig
    # CUDA Kontrolü
    try:
        from trion_core.engine import attention_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
except ImportError as e:
    st.error(f"Sistem Hatası: {e}")
    st.info("Lütfen 'pip install transformers' kurduğundan emin ol.")
    st.stop()

# --- TRION ARAYÜZ AYARLARI ---
st.set_page_config(
    page_title="TRION CORE v1.0",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (Trion Teması: Cyan & Dark Grey) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #cfd8dc; }
    
    /* Yan Menü */
    section[data-testid="stSidebar"] { background-color: #171c26; border-right: 1px solid #2d3748; }
    
    /* Başlıklar */
    h1, h2, h3 { color: #00bcd4; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Inputlar */
    .stTextInput > div > div > input { 
        background-color: #232b3b; color: #00bcd4; border: 1px solid #455a64; 
    }
    
    /* Butonlar */
    .stButton > button { 
        background-color: #00bcd4; color: #000; font-weight: bold; border: none; 
        transition: 0.3s;
    }
    .stButton > button:hover { background-color: #00e5ff; box-shadow: 0 0 10px #00bcd4; }
    
    /* Progress Bar */
    .stProgress > div > div > div > div { background-color: #00bcd4; }
</style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- KAYNAK YÜKLEYİCİ ---
@st.cache_resource
def load_trion_system():
    # 1. SÖZLÜK (GPT-2)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 1000000 # Limit Kaldırma
    except:
        st.error("Tokenizer yüklenemedi. İnternet bağlantını kontrol et.")
        st.stop()
        
    # 2. TRION MODELİ
    config = QKVConfig(
        vocab_size=50257,
        d_model=768,
        n_layer=12,
        n_head=12,
        max_seq_len=1024,
        attn_threshold=-0.1 
    )
    model = QKVModel(config).to(device)
    
    status = "⚠️ ÇEKİRDEK BOŞ"
    
    # 3. TRION BEYNİNİ YÜKLE
    if os.path.exists("trion_brain.pt"):
        try:
            state_dict = torch.load("trion_brain.pt", map_location=device)
            model.load_state_dict(state_dict)
            status = "💠 TRION ONLINE"
        except Exception as e:
            status = f"❌ UYUMSUZ VERİ: {e}"
    
    return tokenizer, model, status

tokenizer, model, status_msg = load_trion_system()

# --- YAN MENÜ ---
st.sidebar.title("💠 TRION CORE")
st.sidebar.caption(f"Sistem Durumu: {status_msg}")
st.sidebar.markdown("---")

c1, c2 = st.sidebar.columns(2)
c1.metric("Birim", device.upper())
c2.metric("Mimari", "1.58-BIT")

st.sidebar.subheader("🎛️ İnce Ayar")
curr_thresh = st.sidebar.slider("Sparsity (Seyreklik)", -1.0, 1.0, -0.1, 0.05)
model.config.attn_threshold = curr_thresh

st.sidebar.subheader("🌊 Üretim")
temp = st.sidebar.slider("Temperature", 0.1, 2.0, 0.4) # Varsayılan: Düşük (Stabil)
max_len = st.sidebar.slider("Token Limiti", 50, 500, 150)

# --- ANA SEKMELER ---
tabs = st.tabs(["💬 TRION CHAT", "🎓 TRION LAB (Eğitim)"])

# ----------------- CHAT SEKMEKİ -----------------
with tabs[0]:
    st.subheader("Neural Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Trion Core sistemi hazır. Veri girişi bekleniyor."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Komut girin..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            res_box = st.empty()
            full_res = ""
            
            model.eval()
            if CUDA_AVAILABLE and device == 'cuda': model.half()
            
            try:
                # Token Limit Koruması
                tokens = tokenizer.encode(prompt)
                if len(tokens) > 200: tokens = tokens[-200:]
                
                idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    for i in range(max_len):
                        cond_idx = idx[:, -1000:]
                        logits = model(cond_idx)
                        next_logits = logits[:, -1, :]
                        
                        probs = torch.softmax(next_logits.float() / temp, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        idx = torch.cat((idx, next_token), dim=1)
                        
                        word = tokenizer.decode([next_token.item()])
                        full_res += word
                        res_box.markdown(full_res + "▌")
                
                res_box.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
                
            except Exception as e:
                st.error(f"İşlem Hatası: {e}")

# ----------------- EĞİTİM SEKMESİ -----------------
with tabs[1]:
    st.subheader("🎓 Trion Lab: Bilgi Yükleme")
    st.info("Trion Core, matematiksel olarak dengelenmiştir. Veri yüklediğinizde çok hızlı öğrenir.")
    
    uploaded_file = st.file_uploader("Metin Dosyası (.txt)", type=["txt"])
    
    c1, c2, c3 = st.columns(3)
    steps = c1.number_input("Adım (Steps)", 10, 10000, 200)
    batch = c2.number_input("Batch Size", 2, 32, 4)
    lr = c3.number_input("Öğrenme Hızı", 1e-6, 1e-2, 5e-5, format="%.6f")
    
    if st.button("💠 EĞİTİMİ BAŞLAT", type="primary"):
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.success(f"Veri Alındı: {len(text)} karakter. İşleniyor...")
            
            # Tokenizer Limit Bypass
            full_tokens = tokenizer.encode(text, truncation=False)
            data = torch.tensor(full_tokens, dtype=torch.long)
            
            model.train()
            model.float()
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            
            chart = st.line_chart([])
            prog = st.progress(0)
            
            losses = []
            start_t = time.time()
            
            try:
                for i in range(steps):
                    ix = torch.randint(len(data) - 128, (batch,))
                    x_b = torch.stack([data[j:j+128] for j in ix]).to(device)
                    y_b = torch.stack([data[j+1:j+129] for j in ix]).to(device)
                    
                    logits = model(x_b)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y_b.view(-1))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if i % 5 == 0:
                        chart.line_chart(losses)
                        prog.progress((i+1)/steps)
                
                st.balloons()
                st.success(f"✅ Eğitim Tamamlandı! ({time.time()-start_t:.1f}s)")
                torch.save(model.state_dict(), "trion_brain.pt")
                st.toast("Trion Beyni Güncellendi!")
                
            except Exception as e:
                st.error(f"Eğitim Hatası: {e}")
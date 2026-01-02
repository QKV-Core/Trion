import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# --- REQUISITOS ---
try:
    from transformers import GPT2Tokenizer
    from trion_core.modeling import  QKVModel, QKVConfig
    # CUDA Kontrolü
    try:
        from trion_core.engine import attention_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        CUDA_AVAILABLE = False
except ImportError as e:
    st.error(f"System error: {e}")
    st.info("Please make sure to run 'pip install transformers'.")
    st.stop()

# --- TRION CONFIGURACIÓN DE LA INTERFAZ ---
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
    # 1. diccionario (GPT-2)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 1000000 # Levantando el Límite
    except:
        st.error("Can't load tokenizer. Check Internet connection.")
        st.stop()
        
    # 2. TRION MODEL
    config = QKVConfig(
        vocab_size=50257,
        d_model=768,
        n_layer=12,
        n_head=12,
        max_seq_len=1024,
        attn_threshold=-0.1 
    )
    model = QKVModel(config).to(device)
    
    status = "⚠️ Empty kernel"
    
    # 3. TRION BEYNİNİ YÜKLE
    if os.path.exists("trion_brain.pt"):
        try:
            state_dict = torch.load("trion_brain.pt", map_location=device)
            model.load_state_dict(state_dict)
            status = "💠 TRION ONLINE"
        except Exception as e:
            status = f"❌ Incompatible data: {e}"
    
    return tokenizer, model, status

tokenizer, model, status_msg = load_trion_system()

# --- YAN MENÜ ---
st.sidebar.title("💠 TRION CORE")
st.sidebar.caption(f"System state: {status_msg}")
st.sidebar.markdown("---")

c1, c2 = st.sidebar.columns(2)
c1.metric("Device", device.upper())
c2.metric("Architecture", "1.58-BIT")

st.sidebar.subheader("🎛️ Tunning")
curr_thresh = st.sidebar.slider("Sparsity ", -1.0, 1.0, -0.1, 0.05)
model.config.attn_threshold = curr_thresh

st.sidebar.subheader("🌊 Production")
temp = st.sidebar.slider("Temp", 0.1, 2.0, 0.4) # Predeterminado: Bajo (estable)
max_len = st.sidebar.slider("Token Limit", 50, 500, 150)

# --- PESTAÑAS PRINCIPALES ---
tabs = st.tabs(["💬 TRION CHAT", "🎓 TRION LAB (Training)"])

# ----------------- PESTAÑA DE CHAT -----------------
with tabs[0]:
    st.subheader("Neural Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Trion Core system ready. Waiting for data entry.."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter command..."):
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
                st.error(f"Processing error: {e}")

# ----------------- EĞİTİM SEKMESİ -----------------
with tabs[1]:
    st.subheader("🎓 Trion Lab: Data load")
    st.info("Trion Core, math balanced. Fast learning on data load.")
    # Support for multiple document formats: .txt, .pdf, .docx, .epub, .mobi
    uploaded_file = st.file_uploader("Upload document (.txt, .pdf, .docx, .epub, .mobi)", type=["txt", "pdf", "docx", "epub", "mobi"])    
    c1, c2, c3 = st.columns(3)
    steps = c1.number_input("Steps", 10, 10000, 200)
    batch = c2.number_input("Batch Size", 2, 32, 4)
    lr = c3.number_input("Training speed", 1e-6, 1e-2, 5e-5, format="%.6f")    
    def extract_text_from_file(uploaded) -> str:
        import io
        name = uploaded.name.lower()
        raw = uploaded.read()
        # Plain text
        if name.endswith('.txt'):
            return raw.decode('utf-8', errors='ignore')

        # DOCX
        if name.endswith('.docx'):
            try:
                from docx import Document
            except Exception as e:
                raise RuntimeError("To read .docx install 'python-docx' (pip install python-docx)") from e
            doc = Document(io.BytesIO(raw))
            full = []
            for p in doc.paragraphs:
                if p.text:
                    full.append(p.text)
            return '\n'.join(full)

        # PDF
        if name.endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
            except Exception as e:
                raise RuntimeError("To read .pdf install 'PyPDF2' (pip install PyPDF2)") from e
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for pg in reader.pages:
                txt = pg.extract_text()
                if txt:
                    pages.append(txt)
            return '\n'.join(pages)

        # EPUB
        if name.endswith('.epub'):
            try:
                from ebooklib import epub
                from bs4 import BeautifulSoup
            except Exception as e:
                raise RuntimeError("To read .epub install 'ebooklib' and 'beautifulsoup4' (pip install ebooklib beautifulsoup4)") from e
            book = epub.read_epub(io.BytesIO(raw))
            parts = []
            from ebooklib import ITEM_DOCUMENT
            for item in book.get_items_of_type(ITEM_DOCUMENT):
                try:
                    content = item.get_content().decode('utf-8')
                except Exception:
                    try:
                        content = item.get_content().decode('latin-1')
                    except Exception:
                        continue
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n')
                if text:
                    parts.append(text)
            return '\n'.join(parts)

        # MOBI (best-effort)
        if name.endswith('.mobi'):
            # Try to use 'mobi' package if available
            try:
                from mobi import Mobi
            except Exception:
                raise RuntimeError("To read .mobi install 'mobi' (pip install mobi) or convert the file to .epub/.txt")
            buf = io.BytesIO(raw)
            m = Mobi(buf)
            try:
                m.parse()
            except Exception as e:
                raise RuntimeError("Failed to parse .mobi file") from e
            try:
                html = m.get_html()  # may be bytes or str
                if isinstance(html, bytes):
                    html = html.decode('utf-8', errors='ignore')
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    return soup.get_text(separator='\n')
                except Exception:
                    return html
            except Exception as e:
                raise RuntimeError("Failed to extract text from .mobi file") from e
        raise RuntimeError('Unsupported file type')

    if st.button("💠 Starting training", type="primary"):
        if uploaded_file:
            try:
                text = extract_text_from_file(uploaded_file)
            except Exception as e:
                st.error(f"Error extracting text: {e}")
                text = None

            if text is None:
                st.info("No text to train on. Upload a supported file or install the required libraries.")
            else:
                st.success(f"Data received: {len(text)} characters. Processing...")            
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
                st.success(f"✅ Training finished ({time.time()-start_t:.1f}s)")
                torch.save(model.state_dict(), "trion_brain.pt")
                st.toast("Trion have been updated.!")                
            except Exception as e:
                st.error(f"Error in training: {e}")
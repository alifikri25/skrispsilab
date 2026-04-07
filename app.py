import os
try:
    import cv2
except ImportError:
    import os
    os.system('pip uninstall -y opencv-python opencv-python-headless')
    os.system('pip install opencv-python-headless')

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Hematopoiesis AI | Pendeteksi Sel Darah", 
    page_icon="🩸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk memperindah UI
st.markdown("""
    <style>
    /* Gradient Background untuk Header */
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Menyembunyikan Streamlit Footer & Menu untuk kesan profesional */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style untuk Metric Box */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 10px 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 7px rgba(0,0,0,0.05);
    }
    
    /* Center the button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100) # Ikon Placeholder Medis
    st.markdown("## 🩸 Hematopoiesis AI")
    st.markdown("Aplikasi berbasis **Computer Vision** (YOLOv8) untuk analisis kuantitatif mikroskopik sel darah tepi.")
    st.markdown("---")
    st.markdown("**📖 Panduan Penggunaan:**")
    st.markdown("1. Pastikan gambar adalah hasil lensa mikroskop (perbesaran 400x - 1000x).")
    st.markdown("2. Upload sampel slide dengan format **JPG/PNG**.")
    st.markdown("3. Klik tombol inferensi untuk mendeteksi sel.")
    st.markdown("---")
    st.info("👨‍💻 *Dikembangkan untuk keperluan Skripsi.*")

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1 style="color:white; margin-bottom: 10px; font-weight:700;">🩺 Klasifikasi Sel Darah Tepi Otomatis</h1>
    <p style="font-size: 18px; font-weight:300;">Menggunakan teknologi Artificial Intelligence untuk mendeteksi Sel Darah Merah (RBC), Sel Darah Putih (WBC), dan Trombosit secara <i>real-time</i> dan akurat.</p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Model 'best.pt' tidak ditemukan! Pastikan model sudah diletakkan se-direktori dengan file app.py.")
else:
    # ================= MAIN CONTAINER =================
    st.markdown("### 📤 Upload Sampel Darah Anda")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], help="Upload gambar mikroskopik sel darah")

    if uploaded_file is not None:
        st.markdown("<hr/>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        
        # Bikin 2 kolom untuk layout perbandingan Asli vs Prediksi
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 🔬 Citra Mikroskopik Asli")
            st.image(image, use_container_width=True, caption="Gambar dari mikroskop yang diunggah")
            
            st.markdown("<br/>", unsafe_allow_html=True)
            # Tombol Prediksi dengan desain penuh (use_container_width=True)
            analyze_btn = st.button("🚀 Analisis Sel Darah Sekarang!", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 AI sedang memindai dan menghitung sel... mohon tunggu."):
                # Lakukan prediksi YOLO
                results = model.predict(image, conf=0.25)
                
                # Render Gambar dengan Bounding Box
                res_image = results[0].plot()
                res_image_rgb = res_image[..., ::-1] # Konversi BGR -> RGB untuk Pil
                
                with col2:
                    st.markdown("#### 🎯 Hasil Deteksi Bounding Box")
                    st.image(res_image_rgb, use_container_width=True, caption="Citra hasil inferensi YOLOv8")
            
            st.markdown("---")
            st.markdown("### 📊 Ringkasan Hitungan Sel (Cell Count)")
            
            boxes = results[0].boxes
            class_names = results[0].names
            
            # Hitung jumlah masig-masing kelas (0: Platelets, 1: RBC, 2: WBC asalnya, menyesuaikan training BCCD)
            counts = {"RBC": 0, "WBC": 0, "Platelets": 0}
            
            for cls_tensor in boxes.cls:
                cls_idx = int(cls_tensor.item())
                cat_name = class_names[cls_idx]
                if cat_name in counts:
                    counts[cat_name] += 1
                else:
                    counts[cat_name] = counts.get(cat_name, 0) + 1
            
            # Gunakan Streamlit Metrics API untuk UI Hitungan Sel yang Cantik
            met_col1, met_col2, met_col3 = st.columns(3)
            met_col1.metric(label="🔴 Jumlah Sel Darah Merah (RBC)", value=f"{counts.get('RBC', 0)}")
            met_col2.metric(label="⚪ Jumlah Sel Darah Putih (WBC)", value=f"{counts.get('WBC', 0)}")
            met_col3.metric(label="🦇 Jumlah Trombosit (Platelets)", value=f"{counts.get('Platelets', 0)}")
            
            st.success("✅ Analisis selesai. Perhitungan di atas menunjukkan estimasi otomatis dari AI pada area pandang mikroskop ini.")

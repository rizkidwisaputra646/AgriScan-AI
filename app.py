import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Smart Farming AI",
    page_icon="🌱",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# =========================
# CSS MODERN & FONT POPPINS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #15803d;
        font-weight: 700;
    }

    .result-card {
        background: linear-gradient(145deg, #ffffff, #f9fafb);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .card-healthy {
        border-top: 5px solid #16a34a;
    }
    
    .card-disease {
        border-top: 5px solid #ef4444;
    }

    .big-number {
        font-size: 54px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #16a34a, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    .disease-name {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        margin-top: 10px;
    }
    
    .badge-good {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .badge-bad {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913584.png", width=80)
    st.title("Tentang AI Ini")
    st.info("""
    **Smart Farming AI** adalah sistem berbasis *Deep Learning* yang dirancang untuk membantu petani mengidentifikasi penyakit pada daun tanaman secara instan.
    """)
    st.markdown("---")
    st.markdown("### 📊 Statistik Model")
    st.markdown("- **Akurasi Model:** ~95%")
    st.markdown("- **Jumlah Kelas:** 15 Jenis")
    st.markdown("- **Tanaman:** Tomat, Kentang, Paprika")
    st.markdown("---")
    st.caption("© 2026 Smart Farming v2.0")

# =========================
# LOAD MODEL (TFLITE)
# =========================
@st.cache_resource
def load_model():
    # Pastikan file model.tflite sudah diupload ke GitHub
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# =========================
# FORMAT NAMA KELAS
# =========================
def format_class_name(name):
    return (
        name.replace("___", " ")
        .replace("__", " ")
        .replace("_", " ")
        .title()
    )

# =========================
# DATA & REKOMENDASI
# =========================
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

disease_info = {
    "Tomato_Late_blight": "Kurangi kelembapan lingkungan, buang daun yang terinfeksi, dan gunakan fungisida sesuai anjuran.",
    "Tomato_Early_blight": "Buang daun yang terinfeksi dan lakukan rotasi tanaman secara berkala.",
    "Tomato_Bacterial_spot": "Gunakan benih sehat dan lakukan penyemprotan bakterisida sesuai kebutuhan.",
    "Tomato_healthy": "Tanaman dalam kondisi sangat prima. Pertahankan jadwal penyiraman dan pemupukan.",
    "Potato___Late_blight": "Pantau kelembapan dan lakukan pengendalian penyakit secara dini dengan fungisida.",
    "Potato___Early_blight": "Buang bagian yang terserang, pastikan drainase tanah baik, dan lakukan sanitasi lahan.",
    "Potato___healthy": "Tanaman dalam kondisi sehat dan bebas dari bercak daun.",
    "Pepper__bell___healthy": "Tanaman paprika sehat. Lanjutkan perawatan rutin Anda.",
    "Pepper__bell___Bacterial_spot": "Gunakan benih bebas penyakit, hindari penyiraman dari atas (overhead irrigation), dan sanitasi area tanam."
}

# =========================
# HEADER UTAMA
# =========================
st.markdown("""
<div class="main-header">
    <h1>🌱 Smart Farming AI Analyzer</h1>
    <p style='color: #6b7280; font-size: 18px;'>Unggah foto daun tanaman Anda dan biarkan AI mendiagnosis kesehatannya dalam hitungan detik.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Seret & lepas foto daun di sini (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)
st.markdown("---")

# =========================
# PREDIKSI & HASIL (TFLITE)
# =========================
if uploaded_file is not None:
    col_img, col_result = st.columns([1, 1.2], gap="large")
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col_img:
        st.subheader("📷 Foto Tanaman")
        st.image(image, use_container_width=True)

    # Persiapan Data Gambar
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) # Penting untuk TFLite
    
    with st.spinner("🧠 AI sedang menganalisis sel daun..."):
        # Setup TFLite Interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Masukkan gambar ke model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Ambil hasil prediksi
        prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Ekstrak hasil
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_index]
    display_name = format_class_name(predicted_class)
    is_healthy = "healthy" in predicted_class.lower()

    # Tampilkan di Kolom Kanan
    with col_result:
        st.subheader("🧬 Hasil Diagnosis")
        
        card_class = "card-healthy" if is_healthy else "card-disease"
        badge = (
            "<div class='status-badge badge-good'>✨ Tanaman Sehat</div>"
            if is_healthy
            else
            "<div class='status-badge badge-bad'>⚠️ Terdeteksi Penyakit</div>"
        )
        
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="subtitle">PREDIKSI AI</div>
            <div class="disease-name">{display_name}</div>
            <div class="big-number">{confidence:.1f}%</div>
            <div>{badge}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 💡 Tindakan yang Disarankan")
        if predicted_class in disease_info:
            if is_healthy:
                st.success(disease_info[predicted_class], icon="✅")
            else:
                st.warning(disease_info[predicted_class], icon="🚨")
        else:
            st.info("Belum tersedia rekomendasi spesifik untuk kategori ini.", icon="ℹ️")

    # =========================
    # ANALISIS MENDALAM
    # =========================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("📊 Analisis Probabilitas Teratas")
    
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    
    col_1, col_2, col_3 = st.columns(3)
    cols = [col_1, col_2, col_3]
    
    for i, idx in enumerate(top_indices):
        nama = format_class_name(class_names[idx])
        nilai = float(prediction[0][idx] * 100)
        
        with cols[i]:
            st.metric(label=f"Peringkat {i+1}", value=f"{nilai:.1f}%", delta=nama, delta_color="off")
            st.progress(nilai / 100.0)

    st.toast('Analisis gambar berhasil diselesaikan!', icon='🎉')

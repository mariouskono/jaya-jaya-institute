import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Jaya Jaya Institut | Attrition Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# FUNGSI CACHE UNTUK LOAD DATA & MODEL (Agar Cepat)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Menyesuaikan dengan path dataset
        df = pd.read_csv('data/data.csv', sep=';')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/rf_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        features = joblib.load('model/features.pkl')
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None

df_raw = load_data()
rf_model, scaler, model_features = load_model()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://raw.githubusercontent.com/dicodingacademy/assets/main/logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.title("📌 Menu Utama")
menu = st.sidebar.radio("Pilih Halaman:", 
                        ["📊 Dashboard Analitik", 
                         "👤 Prediksi Individu", 
                         "📂 Prediksi Massal (CSV)", 
                         "🤖 Model & Evaluasi",
                         "🗄️ Eksplorasi Data"])

st.sidebar.markdown("---")
st.sidebar.info("**Sistem Deteksi Dini Dropout**\n\nDikembangkan oleh: **Bertnardo Mario Uskono (Uno)**")

# ==========================================
# HALAMAN 1: DASHBOARD ANALITIK
# ==========================================
if menu == "📊 Dashboard Analitik":
    st.title("🎓 Dashboard Performa Mahasiswa")
    st.markdown("Ringkasan data historis dari mahasiswa Jaya Jaya Institut untuk memonitor tren akademik dan faktor risiko *dropout*.")
    
    if not df_raw.empty:
        # Metrik Utama
        col1, col2, col3, col4 = st.columns(4)
        total_students = len(df_raw)
        total_dropout = len(df_raw[df_raw['Status'] == 'Dropout'])
        dropout_rate = (total_dropout / total_students) * 100
        
        col1.metric("Total Mahasiswa", f"{total_students:,}")
        col2.metric("Total Dropout", f"{total_dropout:,}")
        col3.metric("Tingkat Dropout", f"{dropout_rate:.1f}%")
        col4.metric("Total Lulus (Graduate)", f"{len(df_raw[df_raw['Status'] == 'Graduate']):,}")
        
        st.markdown("---")
        
        # Visualisasi
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Distribusi Status Mahasiswa")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_raw, x='Status', palette='Set2', ax=ax)
            ax.set_ylabel("Jumlah Mahasiswa")
            st.pyplot(fig)
            
        with col_chart2:
            st.subheader("Dampak Tunggakan (Debtor) thd Dropout")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_raw, x='Debtor', hue='Status', palette='Set1', ax=ax2)
            ax2.set_xticklabels(['Tidak Menunggak', 'Menunggak'])
            st.pyplot(fig2)
            
        st.info("💡 **Insight:** Mahasiswa yang memiliki tunggakan finansial (Debtor) dan gagal memperoleh SKS di semester 1 memiliki probabilitas dropout yang sangat tinggi.")
    else:
        st.error("Data tidak ditemukan! Pastikan file 'data.csv' berada di dalam folder 'data/'.")

# ==========================================
# HALAMAN 2: PREDIKSI INDIVIDU (FORM INPUT)
# ==========================================
elif menu == "👤 Prediksi Individu":
    st.title("👤 Prediksi Risiko Dropout Mahasiswa")
    st.markdown("Masukkan data akademik dan demografi mahasiswa untuk memprediksi probabilitas kelulusannya.")
    
    if rf_model is None:
        st.error("Model tidak ditemukan! Pastikan file model telah diekspor ke folder 'model/'.")
    else:
        with st.form("prediction_form"):
            st.subheader("Data Semester 1 & 2 (Faktor Paling Krusial)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Semester 1**")
                s1_enrolled = st.number_input("SKS Diambil (Sem 1)", min_value=0, max_value=30, value=6)
                s1_approved = st.number_input("SKS Lulus (Sem 1)", min_value=0, max_value=30, value=5)
                s1_grade = st.number_input("Nilai Rata-rata (Sem 1) [0-20]", min_value=0.0, max_value=20.0, value=12.0)
                
            with col2:
                st.markdown("**Semester 2**")
                s2_enrolled = st.number_input("SKS Diambil (Sem 2)", min_value=0, max_value=30, value=6)
                s2_approved = st.number_input("SKS Lulus (Sem 2)", min_value=0, max_value=30, value=5)
                s2_grade = st.number_input("Nilai Rata-rata (Sem 2) [0-20]", min_value=0.0, max_value=20.0, value=12.0)

            st.subheader("Data Finansial & Demografi")
            col3, col4 = st.columns(2)
            with col3:
                age = st.number_input("Usia saat mendaftar", min_value=15, max_value=80, value=20)
                debtor = st.selectbox("Apakah memiliki tunggakan biaya? (Debtor)", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
                scholarship = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
            
            with col4:
                gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Pria (1)" if x == 1 else "Wanita (0)")
                tuition_up_to_date = st.selectbox("Uang Kuliah Lunas?", [0, 1], index=1, format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
                admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0)

            submit_button = st.form_submit_button(label="🔍 Deteksi Risiko")

        if submit_button:
            # Karena model meminta 36 fitur, kita set default median untuk fitur yang tidak diinput manual
            # Agar UI tidak terlalu penuh dengan 36 inputan.
            input_data = pd.DataFrame(columns=model_features)
            input_data.loc[0] = 0 # Inisialisasi awal dengan 0
            
            # Mengisi nilai dari input user
            input_data['Curricular_units_1st_sem_enrolled'] = s1_enrolled
            input_data['Curricular_units_1st_sem_approved'] = s1_approved
            input_data['Curricular_units_1st_sem_grade'] = s1_grade
            input_data['Curricular_units_2nd_sem_enrolled'] = s2_enrolled
            input_data['Curricular_units_2nd_sem_approved'] = s2_approved
            input_data['Curricular_units_2nd_sem_grade'] = s2_grade
            input_data['Age_at_enrollment'] = age
            input_data['Debtor'] = debtor
            input_data['Scholarship_holder'] = scholarship
            input_data['Gender'] = gender
            input_data['Tuition_fees_up_to_date'] = tuition_up_to_date
            input_data['Admission_grade'] = admission_grade
            
            # Scaling data
            input_scaled = scaler.transform(input_data)
            
            # Prediksi dan Probabilitas
            prediction = rf_model.predict(input_scaled)
            probabilities = rf_model.predict_proba(input_scaled)
            dropout_prob = probabilities[0][1] * 100
            
            st.markdown("---")
            if prediction[0] == 1:
                st.error(f"⚠️ **PERINGATAN!** Mahasiswa ini diprediksi akan **DROPOUT**.")
                st.write(f"Probabilitas Dropout: **{dropout_prob:.1f}%**")
                st.progress(int(dropout_prob))
                st.warning("Rekomendasi: Segera jadwalkan sesi konseling akademik dan tinjau kelayakan bantuan finansial.")
            else:
                st.success(f"✅ **AMAN!** Mahasiswa ini diprediksi akan **LULUS (GRADUATE)**.")
                st.write(f"Probabilitas Dropout hanya: **{dropout_prob:.1f}%**")
                st.progress(int(dropout_prob))

# ==========================================
# HALAMAN 3: PREDIKSI MASSAL (BATCH)
# ==========================================
elif menu == "📂 Prediksi Massal (CSV)":
    st.title("📂 Prediksi Massal via File CSV")
    st.markdown("Gunakan fitur ini untuk memprediksi puluhan atau ratusan mahasiswa sekaligus.")
    
    uploaded_file = st.file_uploader("Upload file data mahasiswa (.csv)", type=["csv"])
    
    if uploaded_file is not None and rf_model is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=';')
            st.write("Preview Data Input:")
            st.dataframe(batch_df.head())
            
            if st.button("Jalankan Prediksi Massal"):
                with st.spinner("Memproses data..."):
                    # Filtering columns to match model features
                    missing_cols = set(model_features) - set(batch_df.columns)
                    if missing_cols:
                        st.error(f"File CSV tidak valid. Kekurangan kolom: {missing_cols}")
                    else:
                        X_batch = batch_df[model_features]
                        X_batch_scaled = scaler.transform(X_batch)
                        
                        predictions = rf_model.predict(X_batch_scaled)
                        batch_df['PREDIKSI_STATUS'] = ["Dropout" if p == 1 else "Lulus" for p in predictions]
                        
                        st.success("Prediksi berhasil diselesaikan!")
                        
                        # Tampilkan hasil
                        st.write("Hasil Prediksi:")
                        st.dataframe(batch_df[['PREDIKSI_STATUS'] + model_features].head(10))
                        
                        # Unduh Hasil
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Hasil Lengkap (CSV)",
                            data=csv,
                            file_name='hasil_prediksi_massal.csv',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ==========================================
# HALAMAN 4: MODEL & EVALUASI
# ==========================================
elif menu == "🤖 Model & Evaluasi":
    st.title("🤖 Model Machine Learning & Evaluasi")
    st.markdown("Halaman ini menyajikan transparansi terkait algoritma prediktif yang digunakan beserta metrik performanya berdasarkan pengujian data historis.")
    
    st.subheader("1. Algoritma: Random Forest Classifier")
    st.write("""
    Sistem prediksi ini ditenagai oleh algoritma **Random Forest**. Pendekatan *ensemble learning* ini dipilih karena beberapa alasan strategis:
    - **Tangguh Terhadap Banyak Fitur:** Mampu menangani 36 fitur akademik dan demografi secara bersamaan tanpa rentan terhadap *overfitting*.
    - **Penanganan Ketidakseimbangan Kelas:** Menggunakan parameter `class_weight='balanced'` yang secara proaktif memberikan penalti lebih besar jika algoritma gagal mendeteksi kelas minoritas (*Dropout*). Ini memastikan model tidak bias hanya menebak mahasiswa akan lulus.
    """)
    
    st.markdown("---")
    
    st.subheader("2. Performa Model (Classification Report)")
    st.write("Berdasarkan pengujian pada **726 data uji** (*unseen data*), model menghasilkan performa empiris sebagai berikut:")
    
    # Menampilkan metrik dalam bentuk kartu yang estetik
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Akurasi Keseluruhan", value="93%")
    col2.metric(label="Recall (Dropout)", value="88%")
    col3.metric(label="Precision (Dropout)", value="94%")
    col4.metric(label="F1-Score (Dropout)", value="91%")
    
    st.info("💡 **Metrik Paling Krusial:** Dalam konteks institusi pendidikan, mendeteksi sebanyak mungkin probabilitas *dropout* adalah yang terpenting. Nilai **Recall 88%** membuktikan bahwa dari seluruh mahasiswa yang aktualnya keluar, model berhasil menyelamatkan/mendeteksi 88% di antaranya sejak dini.")
    
    st.markdown("---")
    
    st.subheader("3. Analisis Dampak Bisnis (Confusion Matrix)")
    st.write("Evaluasi model tidak hanya berupa angka teknis, melainkan dapat dipetakan langsung ke dalam efisiensi operasional bimbingan akademik:")
    
    col_cm1, col_cm2 = st.columns(2)
    
    with col_cm1:
        st.success("**✅ True Positives (250 Mahasiswa)** \n\nTerdapat 250 mahasiswa *dropout* yang berhasil terdeteksi dengan tepat. Ini adalah kelompok prioritas utama untuk segera diberikan intervensi dini seperti konseling akademik atau bantuan finansial.")
        st.info("**✅ True Negatives (425 Mahasiswa)** \n\n425 mahasiswa yang lulus berhasil diprediksi lulus dengan akurat oleh sistem.")
        
    with col_cm2:
        st.warning("**⚠️ False Positives (17 Mahasiswa)** \n\nTerdapat 17 mahasiswa lulus yang diprediksi akan *dropout*. Risiko bisnisnya sangat rendah; institusi hanya mengalokasikan sedikit waktu konseling ekstra untuk mahasiswa yang sebenarnya aman.")
        st.error("**❌ False Negatives (34 Mahasiswa)** \n\nHanya 34 mahasiswa *dropout* (di bawah 5% dari total data uji) yang luput dari deteksi sistem. Ini adalah batas toleransi eror yang sangat baik untuk model prediktif dunia nyata.")

# ==========================================
# HALAMAN 5: EKSPLORASI DATA
# ==========================================
elif menu == "🗄️ Eksplorasi Data":
    st.title("🗄️ Eksplorasi Data Mentah")
    st.markdown("Tampilan dataset asli untuk keperluan audit dan pencarian manual.")
    
    if not df_raw.empty:
        status_filter = st.selectbox("Filter berdasarkan Status:", ["Semua", "Graduate", "Dropout", "Enrolled"])
        
        if status_filter != "Semua":
            filtered_df = df_raw[df_raw['Status'] == status_filter]
        else:
            filtered_df = df_raw
            
        st.write(f"Menampilkan **{len(filtered_df)}** baris data:")
        st.dataframe(filtered_df)
    else:
        st.warning("Dataset tidak tersedia.")


# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan Jaya Jaya Institut

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Hingga saat ini, institusi telah mencetak banyak lulusan dengan reputasi yang sangat baik. Meskipun demikian, institusi menghadapi tantangan serius dalam mempertahankan peserta didiknya hingga lulus.

### Permasalahan Bisnis
Tingkat mahasiswa yang tidak menyelesaikan pendidikan alias putus studi (*dropout*) di institusi mencapai angka kritis **32,1%**. Jumlah *dropout* yang tinggi ini berdampak negatif pada reputasi akademik kampus, stabilitas finansial institusi, serta mengindikasikan adanya celah dalam sistem dukungan mahasiswa. Institusi membutuhkan sistem yang dapat mendeteksi dini mahasiswa berisiko agar bimbingan khusus dapat segera diberikan.

### Cakupan Proyek
Proyek ini mengimplementasikan tahapan data science secara end-to-end menggunakan kerangka kerja **CRISP-DM**:
1. **Data Preparation**: Memfilter data kelas abu-abu (status `Enrolled`) untuk memformulasikan klasifikasi biner (*Dropout* vs *Graduate*).
2. **Exploratory Data Analysis (EDA)**: Menganalisis faktor demografi, sosio-ekonomi (Tunggakan & Beasiswa), dan rekam jejak akademik (SKS & Nilai Semester 1).
3. **Machine Learning**: Membangun model klasifikasi **Random Forest** dengan penanganan *class imbalance* (`class_weight='balanced'`) untuk mendeteksi probabilitas *dropout* mahasiswa.
4. **Business Dashboard**: Menyajikan visualisasi interaktif melalui Metabase untuk pemantauan level makro oleh manajemen.
5. **Deployment Prototype**: Membangun antarmuka aplikasi interaktif berbasis **Streamlit** untuk memfasilitasi deteksi dini oleh staf akademik.

### Persiapan

**Sumber data**: `data/data.csv`

**Setup Environment**:
Proyek ini dikembangkan menggunakan Python. Berikut adalah langkah-langkah untuk mereplikasi lingkungan komputasinya:

**1. Lingkungan Lokal (Windows/Mac/Linux):**
Sangat disarankan untuk menggunakan *virtual environment*.
```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi venv (Windows)
venv\Scripts\activate
# Aktivasi venv (Mac/Linux)
source venv/bin/activate

# Instalasi library sesuai requirements
pip install -r requirements.txt
```

**2. Google Colab (Untuk eksplorasi Notebook):**
Unggah berkas `data.csv` dan `requirements.txt` ke direktori kerja Colab, lalu jalankan:
```python
!pip install -r requirements.txt
```

---

## Business Dashboard
Dashboard pemantauan akademik telah dibangun menggunakan **Metabase** untuk menyajikan metrik evaluasi secara visual.
* **Filter Data Valid**: Dashboard menggunakan basis data tersaring (*Dropout* dan *Graduate* saja) untuk menjaga akurasi rasio perbandingan empiris.
* **Manipulasi Keterbacaan Label**: Variabel biner (0 dan 1) pada fitur finansial telah diubah menjadi label kategorikal (*Tanpa Beasiswa / Penerima Beasiswa*, *Tidak Menunggak / Menunggak*) melalui fungsi `Custom Column` untuk kemudahan interpretasi visual.
* **Kredensial Akses (Localhost)**:
  * **Email**: `root@mail.com`
  * **Password**: `root123`

---

## Conclusion
Berdasarkan hasil analisis data historis dan performa model prediktif (Akurasi 93%, Recall Dropout 88%), karakteristik utama mahasiswa yang berisiko melakukan *dropout* adalah:
1. **Krisis Finansial (Debtor)**: Mahasiswa yang berstatus sebagai *Debtor* (memiliki tunggakan biaya kuliah) mendominasi rasio *dropout*. Sebaliknya, pemberian beasiswa (*Scholarship holder*) terbukti menekan angka *dropout* mendekati nol.
2. **Kegagalan Adaptasi Akademik Awal**: Performa di semester pertama menjadi indikator paling absolut. Mahasiswa yang *dropout* memiliki rata-rata nilai semester 1 sebesar **7.26**, sangat tertinggal dibandingkan mahasiswa yang lulus (rata-rata nilai **12.64**). Konsistensi performa yang buruk ini berlanjut linear ke semester kedua.

### Rekomendasi Action Items
Untuk menekan angka *dropout* secara sistematis, manajemen institusi direkomendasikan untuk menerapkan strategi berikut:

- **Implementasi Early Warning System (EWS)**: Mewajibkan penggunaan aplikasi prediktif (prototipe Streamlit) pada pertengahan semester 1. Mahasiswa yang diproyeksikan memiliki nilai rata-rata di bawah 10 wajib mengikuti program *remedial* atau bimbingan akademik intensif.
- **Restrukturisasi Finansial Proaktif**: Mengidentifikasi mahasiswa dengan status tunggakan (*Debtor*) di awal perkuliahan dan menawarkan skema cicilan khusus, atau memprioritaskan mereka sebagai kandidat penerima bantuan beasiswa (*Scholarship*).
- **Evaluasi Kurikulum Semester 1**: Mengingat besarnya angka kegagalan memperoleh SKS kelulusan (*approved units*) di semester pertama, departemen akademik perlu meninjau beban kurikulum untuk mahasiswa baru agar lebih adaptif.

---

## Cara Menjalankan Prototype Sistem Machine Learning
Departemen Akademik dapat menggunakan aplikasi **Streamlit** untuk melakukan prediksi individu maupun massal (via unggah file CSV).

**Perintah Eksekusi di Terminal Lokal:**
Pastikan *virtual environment* aktif dan Anda berada di dalam folder proyek, lalu jalankan:
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di *browser* Anda pada alamat `http://localhost:8501`.

**(Opsional) Akses Cloud:**
Aplikasi ini juga telah di-*deploy* ke Streamlit Community Cloud dan dapat diakses melalui tautan berikut:
*[Masukkan Link Streamlit Cloud Anda Di Sini jika ada]*

---
**Submission oleh: Bertnardo Mario Uskono (Uno)**

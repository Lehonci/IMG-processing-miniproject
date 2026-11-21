# Mini Project Pemrosesan Citra Digital: Sistem Ekstraksi & Basis Data Kartu Keluarga (DigiKK-OCR)

Aplikasi berbasis web dengan backend **Python (Flask)**, pemrosesan citra digital menggunakan **OpenCV & NumPy**, ekstraksi teks otomatis menggunakan **RapidOCR (ONNX Runtime)**, dan basis data **SQLite** untuk membaca serta mengelola data **Kartu Keluarga (KK)**.

---

## Fitur Utama & Spesifikasi

### 1. Implementasi Pemrosesan Citra Digital (`image_processor.py`)
1. **Akuisisi Citra, Properti Matriks, & Konversi Kanal**: Membaca matriks citra (`shape`, `dtype`, `size`), konversi ruang warna `BGR -> RGB`, dan pemisahan kanal `R, G, B`.
2. **Konversi Aras Keabuan (Grayscale)**: Mengubah citra RGB menjadi citra keabuan dengan rumus luminositas berbobot $\text{Gray} = 0.299R + 0.587G + 0.114B$ serta `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.
3. **Analisis Histogram & Contrast Stretching**: Menghitung distribusi histogram (`np.bincount`), normalisasi $h_i = n_i / n$, rata-rata kecerahan ($u$), standar deviasi kontras ($d$), dan *Contrast Stretching* berbasis persentil $2\%$–$98\%$ (`np.percentile`).
4. **Operasi Aras Titik (*Point Processing*) & Aras Lokal (*Neighborhood Processing*)**:
   - **Aras Titik**: *Brightening* (`np.clip(gray + b, 0, 255)`), *Citra Negatif* (`255 - gray`), dan *Thresholding* (`cv2.threshold` & `cv2.adaptiveThreshold`) untuk menghasilkan citra biner.
   - **Aras Lokal**: *Smoothing / Mean Filter* $3\times 3$ (`cv2.blur(gray, (3, 3))`) untuk mereduksi derau dan *Sharpening* dengan kernel Laplacian $3\times 3$ (`cv2.filter2D`).
5. **Morfologi Citra Biner, Slicing ROI (Cropping), & Resize**:
   - Menggunakan *Structuring Element* berbentuk garis dengan operasi **Erosi** (`cv2.erode`), **Dilasi** (`cv2.dilate`), dan **Opening** (`cv2.MORPH_OPEN`) untuk mendeteksi baris serta kolom Tabel 1 (Atas) dan Tabel 2 (Bawah) Kartu Keluarga.
   - Memotong (*crop*) setiap sel tabel dengan *array slicing* NumPy `gray[y1:y2, x1:x2]`, menyatukan karakter kata dengan operasi **Closing** (`cv2.MORPH_CLOSE`), memperbesar (*resize*) sel dengan `cv2.resize`, lalu membaca teks menggunakan OCR.

### 2. Struktur Dua Tabel Database (SQLite)
Database SQLite (`kk_database.db`) dibuat otomatis saat aplikasi pertama kali dijalankan, terdiri dari:
- **`tbl_identitasKK`**:
  - `id_kk` (PK), `no_kk` (Unique 16 digit), `nama_kepala_keluarga`, `alamat`, `rt_rw`, `kode_pos`, `desa_kelurahan`, `kecamatan`, `kabupaten_kota`, `provinsi`, `tanggal_dikeluarkan`, `image_path`, `created_at`, `updated_at`.
- **`tbl_anggotaKeluargaKK`**:
  - `id_anggota` (PK), `id_kk` (FK ke `tbl_identitasKK` dengan `ON DELETE CASCADE`), `no_urut`, `nama_lengkap`, `nik`, `jenis_kelamin`, `tempat_lahir`, `tanggal_lahir`, `agama`, `pendidikan`, `jenis_pekerjaan`, `golongan_darah`, `status_perkawinan`, `tanggal_perkawinan`, `status_hubungan`, `kewarganegaraan`, `no_paspor`, `no_kitap`, `nama_ayah`, `nama_ibu`.

### 3. Halaman Output Aplikasi
1. **Halaman Input Citra KK (`/`)**:
   - Unggah file citra KK (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) atau gunakan **Contoh Dummy KK** yang tersedia.
2. **Halaman Konfirmasi Hasil Pembacaan (`/confirm/<job_id>`)**:
   - Menampilkan pratinjau citra KK yang diunggah beserta **Form Editable** untuk mengoreksi hasil bacaan `tbl_identitasKK` dan seluruh baris/kolom `tbl_anggotaKeluargaKK` sebelum disimpan ke database.
3. **Halaman Penelusuran Tabel KK Saja (`/kk`)**:
   - Menampilkan khusus daftar data dari `tbl_identitasKK` dengan fitur pencarian.
   - Setiap baris memiliki link **"Detail Anggota"** menuju halaman detail anggota keluarga pada KK tersebut, serta aksi **Update (Edit)** dan **Delete (Hapus)** data identitas KK.
4. **Halaman Detail KK & Anggota Keluarga (`/kk/<id_kk>`)**:
   - Menampilkan detail `tbl_identitasKK` dan tabel lengkap `tbl_anggotaKeluargaKK` (Kolom 1–17).
   - Mendukung operasi **Create, Read, Update, Delete (CRUD)** penuh terhadap data identitas KK maupun setiap anggota keluarga pada KK tersebut.

---

## Cara Menjalankan Aplikasi

1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/Lehonci/IMG-processing-miniproject.git
   cd IMG-processing-miniproject
   ```

2. **Instal pustaka Python yang dibutuhkan**:
   ```bash
   pip install flask opencv-python numpy Pillow rapidocr-onnxruntime
   ```

3. **Jalankan server aplikasi Flask**:
   ```bash
   python app.py
   ```
   *(Saat dijalankan pertama kali, aplikasi akan otomatis menginisialisasi database SQLite serta membuat citra contoh Kartu Keluarga siap uji).*

4. **Buka aplikasi di browser**:
   ```text
   http://127.0.0.1:5000
   ```

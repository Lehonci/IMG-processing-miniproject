# Mini Project Pemrosesan Citra Digital: Sistem Ekstraksi & Basis Data Kartu Keluarga (DigiKK-OCR)

Aplikasi berbasis web dengan backend **Python (Flask)**, pemrosesan citra digital menggunakan **OpenCV**, ekstraksi teks otomatis menggunakan **RapidOCR (ONNX Runtime)**, dan basis data **SQLite** untuk membaca dan mengelola data **Kartu Keluarga (KK)**.

---

## Fitur Utama & Spesifikasi

### 1. Pipeline Pemrosesan Citra Digital (OpenCV) & OCR
1. **Akuisisi & Normalisasi Resolusi + Auto-Deskewing**: Membaca citra KK dan mengoreksi sudut kemiringan dokumen menggunakan transformasi *Hough Lines*.
2. **Konversi Aras Keabuan (Grayscale)**: Mengubah citra RGB menjadi citra keabuan $I = 0.299R + 0.587G + 0.114B$.
3. **Peningkatan Kontras Lokal (CLAHE)**: Menggunakan *Contrast Limited Adaptive Histogram Equalization* untuk memperjelas teks pada area pencahayaan tidak merata.
4. **Reduksi Derau (Gaussian Denoising)**: Menghaluskan derau latar belakang dokumen.
5. **Binarisasi Citra (Adaptive Gaussian / Otsu Thresholding)**: Memisahkan objek teks dan garis tabel dari latar belakang.
6. **Operasi Morfologi Citra Biner**: Menggunakan *Morphological Opening* (erosi & dilasi) dengan *structuring element* horizontal dan vertikal untuk mendeteksi struktur garis tabel Kartu Keluarga.
7. **Deteksi Bounding Box OCR & Spatial Layout Parser**: Mengekstrak teks beserta koordinat spasial `(x, y, w, h)` untuk memetakan data ke field identitas KK serta menggabungkan baris Tabel Atas (Kolom 1–9) dan Tabel Bawah (Kolom 10–17).

### 2. Struktur Dua Tabel Database (`kk_database.db`)
- **`tbl_identitasKK`**:
  - `id_kk` (PK), `no_kk` (Unique 16 digit), `nama_kepala_keluarga`, `alamat`, `rt_rw`, `kode_pos`, `desa_kelurahan`, `kecamatan`, `kabupaten_kota`, `provinsi`, `tanggal_dikeluarkan`, `image_path`, `created_at`, `updated_at`.
- **`tbl_anggotaKeluargaKK`**:
  - `id_anggota` (PK), `id_kk` (FK ke `tbl_identitasKK` dengan `ON DELETE CASCADE`), `no_urut`, `nama_lengkap`, `nik`, `jenis_kelamin`, `tempat_lahir`, `tanggal_lahir`, `agama`, `pendidikan`, `jenis_pekerjaan`, `golongan_darah`, `status_perkawinan`, `tanggal_perkawinan`, `status_hubungan`, `kewarganegaraan`, `no_paspor`, `no_kitap`, `nama_ayah`, `nama_ibu`.

### 3. Halaman Output Aplikasi
1. **Halaman Input Citra KK (`/`)**:
   - Unggah file citra KK (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) atau gunakan **Contoh Citra KK Siap Uji** yang tersedia.
2. **Halaman Konfirmasi Hasil Pembacaan (`/confirm/<job_id>`)**:
   - Menampilkan visualisasi 6 tahapan Pemrosesan Citra Digital + metrik citra.
   - Menyediakan **Form Editable** untuk data `tbl_identitasKK` dan seluruh baris/kolom `tbl_anggotaKeluargaKK` agar pengguna dapat mengoreksi hasil bacaan OCR sebelum disimpan ke database.
3. **Halaman Penelusuran Tabel KK Saja (`/kk`)**:
   - Menampilkan khusus daftar data dari `tbl_identitasKK` dengan fitur pencarian.
   - Setiap baris memiliki link **"Detail Anggota"** menuju halaman detail anggota keluarga pada KK tersebut, serta aksi **Update (Edit)** dan **Delete (Hapus)** data identitas KK.
4. **Halaman Detail KK & Anggota Keluarga (`/kk/<id_kk>`)**:
   - Menampilkan detail `tbl_identitasKK` dan tabel lengkap `tbl_anggotaKeluargaKK` (Kolom 1–17).
   - Mendukung operasi **Create, Read, Update, Delete (CRUD)** penuh terhadap data identitas KK maupun setiap anggota keluarga pada KK tersebut.

---

## Cara Menjalankan Aplikasi

1. Pastikan dependensi telah terinstal:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan server Flask:
   ```bash
   python app.py
   ```
3. Buka browser dan akses:
   ```
   http://127.0.0.1:5000
   ```

"""
Modul Pemrosesan Citra Digital & Ekstraksi Kartu Keluarga (image_processor.py)
------------------------------------------------------------------------------
Implementasi ini menerapkan seluruh materi yang telah dipelajari pada mata kuliah
Pemrosesan Citra Digital (Pertemuan 1 s/d Pertemuan 8 & Kisi-Kisi UTS):

1. Pertemuan 1 & Praktikum 1 (Pengenalan Tools, Properti Matriks, & Konversi Kanal):
   - Membaca citra ke matriks NumPy, mengecek dimensi (shape), tipe data (dtype), ukuran (size).
   - Konversi BGR ke RGB (cv2.cvtColor) dan pemisahan kanal R, G, B (cv2.split).
2. Pertemuan 2 (Grayscale Manual & OpenCV, Resize Interpolasi, Cropping / Array Slicing):
   - Rumus berbobot Grayscale: Gray = 0.299*R + 0.587*G + 0.114*B dan cv2.COLOR_BGR2GRAY.
   - Resize ukuran citra menggunakan cv2.resize(..., interpolation=cv2.INTER_LINEAR).
   - Cropping / pemotongan Region of Interest (ROI) dengan slicing matriks NumPy: img[y1:y2, x1:x2].
3. Pertemuan 3 (Operasi Aras Titik / Point Processing & Operasi Aras Lokal / Neighborhood):
   - Aras Titik: Thresholding (cv2.threshold & cv2.adaptiveThreshold), Citra Negatif (255 - img_gray),
     dan Brightening / pengaturan kecerahan (np.clip(img_gray + b, 0, 255) / cv2.convertScaleAbs).
   - Aras Lokal: Smoothing / Mean Filter 3x3 (cv2.blur(img, (3, 3))) untuk reduksi noise, dan
     Sharpening / Penajaman tepi dengan kernel Laplacian 3x3 (cv2.filter2D).
4. Pertemuan 4 (Operasi Aritmatika & Operasi Logika Bitwise pada Citra Biner):
   - Operasi logika cv2.bitwise_or, cv2.bitwise_and, dan cv2.bitwise_not pada citra biner.
5. Pertemuan 6 (Histogram Citra, Statistik Kecerahan/Kontras, & Contrast Stretching):
   - Perhitungan histogram (np.bincount), normalisasi h_i = n_i / n, rata-rata intensitas (u),
     dan standar deviasi kontras (d).
   - Peningkatan kontras dengan Contrast Stretching berbasis persentil 2% dan 98% (np.percentile).
6. Pertemuan 8 (Morfologi Citra Biner: Structuring Element, Erosi, Dilasi, Opening, Closing):
   - Structuring Element (SE) garis & plus/kotak untuk operasi Erosi (cv2.erode), Dilasi (cv2.dilate),
     Opening (cv2.MORPH_OPEN) untuk mendeteksi garis tabel KK, dan Closing (cv2.MORPH_CLOSE)
     untuk menyatukan huruf-huruf saat ekstraksi ROI sel tabel.
"""

import difflib
import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Inisialisasi engine pembaca teks (OCR) setelah tahapan Pemrosesan Citra Digital OpenCV
_OCR_ENGINE = RapidOCR()


# ==============================================================================
# DAFTAR KOSAKATA STANDAR KARTU KELUARGA (UNTUK VALIDASI HASIL BACA)
# ==============================================================================

LIST_AGAMA = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KHONGHUCU"]

LIST_PENDIDIKAN = [
    "TIDAK/BELUM SEKOLAH",
    "BELUM TAMAT SD/SEDERAJAT",
    "TAMAT SD/SEDERAJAT",
    "SLTP/SEDERAJAT",
    "SLTA/SEDERAJAT",
    "DIPLOMA I/II",
    "AKADEMI/DIPLOMA III/S. MUDA",
    "DIPLOMA IV/STRATA I",
    "STRATA II",
    "STRATA III",
]

LIST_PEKERJAAN = [
    "BELUM/TIDAK BEKERJA",
    "MENGURUS RUMAH TANGGA",
    "PELAJAR/MAHASISWA",
    "PENSIUNAN",
    "PEGAWAI NEGERI SIPIL",
    "PNS",
    "TENTARA NASIONAL INDONESIA",
    "KEPOLISIAN RI",
    "KARYAWAN SWASTA",
    "KARYAWAN BUMN",
    "KARYAWAN BUMD",
    "KARYAWAN HONORER",
    "BURUH HARIAN LEPAS",
    "PETANI/PEKEBUN",
    "PEDAGANG",
    "WIRASWASTA",
    "GURU",
    "DOSEN",
    "DOKTER",
    "BIDAN",
    "PERAWAT",
    "SOPIR",
]

LIST_GOL_DARAH = ["A", "B", "AB", "O", "TIDAK TAHU", "-"]

LIST_STATUS_KAWIN = [
    "BELUM KAWIN",
    "KAWIN TERCATAT",
    "KAWIN TIDAK TERCATAT",
    "KAWIN",
    "CERAI HIDUP",
    "CERAI MATI",
]

LIST_STATUS_HUBUNGAN = [
    "KEPALA KELUARGA",
    "SUAMI",
    "ISTRI",
    "ANAK",
    "MENANTU",
    "CUCU",
    "ORANG TUA",
    "MERTUA",
    "FAMILI LAIN",
    "PEMBANTU",
    "LAINNYA",
]


# ==============================================================================
# TAHAP 1: PEMROSESAN CITRA DIGITAL (PERTEMUAN 1, 2, 3, 4, 6, 8)
# ==============================================================================

def analisis_histogram_dan_contrast_stretching(gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    [Materi Pertemuan 6 - Histogram Citra & Contrast Stretching]
    1. Menghitung histogram frekuensi kemunculan nilai keabuan [0..255] dengan np.bincount.
    2. Menghitung histogram ternormalisasi h_i = n_i / n.
    3. Menghitung rata-rata intensitas (u = sum(k * h(k))) dan standar deviasi kontras (d).
    4. Melakukan Contrast Stretching menggunakan persentil 2% dan 98% (p2, p98)
       agar distribusi derajat keabuan mengisi rentang [0, 255] secara optimal.
    """
    # 1. Hitung histogram (n_i) dan normalisasi (h_i = n_i / n)
    hist_counts = np.bincount(gray.ravel(), minlength=256)
    total_pixels = float(gray.size)
    hist_norm = hist_counts / total_pixels

    # 2. Hitung nilai rata-rata (mean u) dan standar deviasi (contrast d)
    levels = np.arange(256, dtype=np.float64)
    mean_u = float(np.sum(levels * hist_norm))
    variance_v = float(np.sum(((levels - mean_u) ** 2) * hist_norm))
    std_d = float(np.sqrt(variance_v))

    # 3. Contrast Stretching dengan persentil 2% dan 98% (Pertemuan 6)
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 - p2 > 10:
        stretched = np.clip((gray.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    else:
        stretched = gray.copy()

    info_hist = {
        "mean_brightness": round(mean_u, 2),
        "std_contrast": round(std_d, 2),
    }
    return stretched, info_hist


def operasi_aras_titik_dan_lokal(gray: np.ndarray, mean_brightness: float) -> Dict[str, np.ndarray]:
    """
    [Materi Pertemuan 3 & 4 - Operasi Aras Titik, Aras Lokal, dan Logika Citra Biner]
    1. Operasi Aras Titik - Brightening: g(x,y) = clip(f(x,y) + b, 0, 255) jika citra agak gelap.
    2. Operasi Aras Titik - Citra Negatif: g(x,y) = 255 - f(x,y).
    3. Operasi Aras Lokal - Smoothing (Mean Filter 3x3): cv2.blur(gray, (3, 3)) untuk reduksi noise.
    4. Operasi Aras Lokal - Sharpening (Kernel Laplacian 3x3): cv2.filter2D untuk penajaman tepi.
    5. Operasi Aras Titik - Thresholding: cv2.threshold & cv2.adaptiveThreshold untuk citra biner.
    """
    # 1. Brightening (Aras Titik): koreksi kecerahan jika rata-rata intensitas terlalu gelap (< 145)
    b = 20 if mean_brightness < 145 else 0
    img_bright = np.clip(gray.astype(np.int32) + b, 0, 255).astype(np.uint8)

    # 2. Citra Negatif (Aras Titik): membalik nilai piksel (255 - f)
    img_negatif = 255 - img_bright

    # 3. Smoothing (Aras Lokal): Mean Filter 3x3 (kernel 1/9) untuk mengurangi bintik noise
    img_smooth = cv2.blur(img_bright, (3, 3))

    # 4. Sharpening (Aras Lokal): Kernel Laplacian 3x3 untuk mempertegas tepi karakter
    kernel_laplacian = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    img_sharp = cv2.filter2D(img_smooth, -1, kernel_laplacian)

    # 5. Thresholding (Aras Titik): mengubah citra grayscale menjadi citra biner
    # Threshold sederhana T=128 (sesuai Praktikum 1 & 3) serta Adaptive Threshold untuk ekstraksi tabel
    _, binary_global = cv2.threshold(img_bright, 128, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )

    return {
        "bright": img_bright,
        "negatif": img_negatif,
        "smooth": img_smooth,
        "sharp": img_sharp,
        "binary_global": binary_global,
        "binary_inv": binary_inv,
    }


def preprocess_citra_kk(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [Materi Pertemuan 1 & 2 - Akuisisi Citra, Properti Matriks, & Konversi Grayscale]
    Membaca citra Kartu Keluarga, memeriksa dimensi (shape, dtype, size),
    mengonversi RGB ke Grayscale menggunakan rumus berbobot (0.299R + 0.587G + 0.114B),
    serta menjalankan analisis histogram, operasi aras titik, dan operasi aras lokal.
    """
    raw_bytes = np.fromfile(image_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Gagal membaca file citra: {image_path}")

    # Properti matriks citra (Pertemuan 1 / Praktikum 1)
    tinggi, lebar, channel = img_bgr.shape
    _dtype = img_bgr.dtype
    _total_pixel = img_bgr.size

    # Konversi BGR ke RGB dan pemisahan kanal R, G, B (Pertemuan 1 & 2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)

    # Konversi RGB ke Grayscale dengan rumus luminositas berbobot (Pertemuan 2 & Kasus 8 Kisi-Kisi UTS: pembulatan di akhir)
    gray_manual = np.round(0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Analisis Histogram & Contrast Stretching (Pertemuan 6)
    _, info_hist = analisis_histogram_dan_contrast_stretching(gray_cv)

    # Operasi Aras Titik & Aras Lokal (Pertemuan 3)
    hasil_operasi = operasi_aras_titik_dan_lokal(gray_cv, info_hist["mean_brightness"])

    return img_bgr, gray_cv, hasil_operasi["binary_inv"]


# ==============================================================================
# TAHAP 2: MORFOLOGI CITRA BINER & SEGMENTASI SEL TABEL KK (PERTEMUAN 4 & 8)
# ==============================================================================

def _cluster_positions(positions: np.ndarray, max_gap: int = 7) -> List[int]:
    """Mengelompokkan koordinat piksel garis yang berdekatan menjadi satu titik pusat garis."""
    groups: List[List[int]] = []
    for p in positions:
        val = int(p)
        if not groups or val - groups[-1][-1] > max_gap:
            groups.append([val])
        else:
            groups[-1].append(val)
    return [int(np.mean(g)) for g in groups]


def _detect_table_grids(gray: np.ndarray, binary_inv: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    """
    [Materi Pertemuan 8 - Morfologi Citra Biner: Structuring Element, Erosi, Dilasi, Opening]
    Mendeteksi garis-garis horizontal Tabel 1 (Atas) dan Tabel 2 (Bawah) Kartu Keluarga
    menggunakan Structuring Element (SE) berbentuk garis horizontal dan operasi Opening
    (Erosi diikuti Dilasi: A o B = (A (-) B) (+) B).
    """
    h, w = gray.shape

    # Structuring Element (SE) berbentuk garis horizontal (Slide 6 P08 Morfologi Citra Biner)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))

    # Operasi Opening = Erosi diikuti Dilasi untuk mempertahankan struktur garis horizontal
    eroded_h = cv2.erode(binary_inv, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(eroded_h, horiz_kernel, iterations=1)

    # Hitung proyeksi jumlah piksel garis pada setiap baris y
    y_projection = np.sum(horiz_lines > 0, axis=1)
    y_candidates = np.where(y_projection > w * 0.35)[0]
    y_centers = [y for y in _cluster_positions(y_candidates, max_gap=6) if int(h * 0.15) < y < int(h * 0.90)]

    if len(y_centers) < 10:
        return horiz_lines, []

    # Pisahkan Tabel 1 (Atas) dan Tabel 2 (Bawah) berdasarkan tinggi baris Header Tabel
    diffs = np.diff(y_centers)
    med_row_h = float(np.median(diffs))
    header_indices = [i for i, d in enumerate(diffs) if d > med_row_h * 1.55]

    if len(header_indices) >= 2:
        idx1, idx2 = header_indices[0], header_indices[1]
        t1_lines = y_centers[idx1:idx2]
        t2_lines = y_centers[idx2:]
        return horiz_lines, [t1_lines, t2_lines]

    mid = len(y_centers) // 2
    return horiz_lines, [y_centers[:mid], y_centers[mid:]]


def _detect_row_columns(binary_inv: np.ndarray, y1: int, y2: int, expected_cols: int) -> List[int]:
    """
    [Materi Pertemuan 2 & 8 - Cropping Array Slicing & Morfologi Garis Vertikal]
    Memotong (slicing) pita baris tabel [y1..y2] dan mendeteksi garis pembatas kolom vertikal
    menggunakan Structuring Element (SE) garis vertikal dan operasi Opening.
    """
    h, w = binary_inv.shape
    row_h = max(10, y2 - y1)
    y_top = max(0, y1 - row_h)
    y_bot = min(h, y2 + 2)

    # Array slicing (Pertemuan 2)
    strip = binary_inv[y_top:y_bot, :]

    # Structuring Element (SE) garis vertikal & Opening (Pertemuan 8)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, strip.shape[0] - 4)))
    vert_lines = cv2.morphologyEx(strip, cv2.MORPH_OPEN, vert_kernel)

    x_proj = np.sum(vert_lines > 0, axis=0)
    x_cols = _cluster_positions(np.where(x_proj > strip.shape[0] * 0.45)[0], max_gap=8)

    if len(x_cols) > expected_cols + 1 and x_cols[-1] > int(w * 0.99):
        x_cols = x_cols[:-1]
    return x_cols


def _read_single_cell_ocr(
    gray: np.ndarray,
    horiz_lines: np.ndarray,
    y1_approx: int,
    y2_approx: int,
    x1: int,
    x2: int,
) -> str:
    """
    [Materi Pertemuan 2, 3, 4, 8 - Cropping ROI, Thresholding, Morfologi Closing, & Resize]
    1. Memotong (crop) satu sel tabel dengan slicing NumPy: gray[ly1:ly2, x1:x2].
    2. Melakukan Thresholding lokal untuk mendapatkan citra biner huruf di dalam sel.
    3. Menerapkan operasi Morfologi Closing (Dilasi lalu Erosi) dengan Structuring Element
       untuk menyatukan huruf-huruf dalam satu kata dan menghapus noise terisolasi.
    4. Memotong tepat pada batas kata (ROI teks) dan memperbesarnya dengan cv2.resize
       (interpolasi INTER_LINEAR) sebelum dibaca oleh OCR.
    """
    h, w = gray.shape
    col_hlines = horiz_lines[:, max(0, x1):min(w, x2)]
    proj = np.sum(col_hlines > 0, axis=1)
    win = max(6, (y2_approx - y1_approx) // 2)

    w1 = proj[max(0, y1_approx - win):min(h, y1_approx + win + 1)]
    w2 = proj[max(0, y2_approx - win):min(h, y2_approx + win + 1)]
    ly1 = (max(0, y1_approx - win) + int(np.argmax(w1))) if len(w1) and np.max(w1) > 0 else y1_approx
    ly2 = (max(0, y2_approx - win) + int(np.argmax(w2))) if len(w2) and np.max(w2) > 0 else y2_approx

    if ly2 - ly1 < 6 or x2 - x1 < 8:
        return "-"

    # 1. Cropping sel tabel dengan array slicing NumPy (Pertemuan 2)
    cell = gray[ly1 + 2 : ly2 - 1, x1 + 3 : x2 - 3]
    if cell.size == 0:
        return "-"

    # 2. Thresholding (Aras Titik - Pertemuan 3): piksel huruf gelap bernilai 255 (1), latar 0
    bg_level = float(np.percentile(cell, 85))
    thresh_val = int(max(10, bg_level - 16))
    _, ink_mask = cv2.threshold(cell, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if ink_mask.shape[0] >= 3:
        if np.mean(ink_mask[0, :] > 0) > 0.70:
            ink_mask[0, :] = 0
        if np.mean(ink_mask[-1, :] > 0) > 0.70:
            ink_mask[-1, :] = 0

    # 3. Morfologi Closing (Dilasi lalu Erosi - Pertemuan 8): menyambung celah antar huruf dalam sel
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    closed = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, close_kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)

    boxes = []
    for lbl in range(1, num_labels):
        bx, by, bw, bh, area = stats[lbl]
        if bw >= 3 and bh >= 4 and area >= 8 and bx < cell.shape[1] * 0.75:
            boxes.append((bx, by, bx + bw, by + bh))

    if not boxes:
        return "-"

    boxes.sort(key=lambda b: b[0])
    xmin, xmax = boxes[0][0], boxes[0][2]
    for b in boxes[1:]:
        if b[0] - xmax <= 20:
            xmax = max(xmax, b[2])

    if (xmax - xmin) < 7:
        return "-"

    # 4. Cropping ROI kata & Resize perbesaran 300% dengan INTER_LINEAR (Pertemuan 2)
    tight_crop = cell[:, max(0, xmin - 3) : min(cell.shape[1], xmax + 4)]
    ch, cw = tight_crop.shape
    upscaled = cv2.resize(tight_crop, (cw * 3, ch * 3), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(upscaled, 14, 14, 18, 18, cv2.BORDER_CONSTANT, value=int(bg_level))

    res, _ = _OCR_ENGINE(padded, use_det=False, use_cls=False)
    if not res:
        return "-"
    text = str(res[0][0]).strip()
    return text if text else "-"


# ==============================================================================
# TAHAP 3: FUNGSI PEMBERSIHAN TEKS & PENCOCOKAN KOSAKATA KK
# ==============================================================================

def _clean_digits(text: str) -> str:
    """Mengubah huruf yang mirip angka akibat salah baca OCR menjadi digit angka."""
    mapping = {
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "I": "1", "l": "1", "|": "1",
        "Z": "2", "S": "5", "G": "6", "B": "8", "A": "4",
    }
    return "".join(ch if ch.isdigit() else mapping.get(ch, "") for ch in text)


def _match_closest_vocab(raw_text: str, vocab_list: List[str], min_ratio: float = 0.42) -> str:
    """Mencocokkan teks hasil bacaan OCR ke kosakata resmi Kartu Keluarga terdekat."""
    cleaned = re.sub(r"[^A-Z0-9/\s.-]", "", raw_text.upper()).strip()
    if not cleaned or cleaned == "-":
        return "-"
    if cleaned in vocab_list:
        return cleaned

    best_item = cleaned
    best_score = 0.0
    clean_nospace = re.sub(r"[^A-Z0-9]", "", cleaned)

    for item in vocab_list:
        item_nospace = re.sub(r"[^A-Z0-9]", "", item)
        score = difflib.SequenceMatcher(None, clean_nospace, item_nospace).ratio()
        if score > best_score:
            best_score = score
            best_item = item

    return best_item if best_score >= min_ratio else cleaned


def _format_tanggal(raw_text: str) -> str:
    """Memformat hasil bacaan tanggal menjadi DD-MM-YYYY."""
    m = re.search(r"(\d{2})[-/.](\d{2})[-/.](\d{4})", raw_text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    digits = _clean_digits(raw_text)
    if len(digits) == 8:
        return f"{digits[:2]}-{digits[2:4]}-{digits[4:8]}"
    if len(digits) == 7:
        return f"{digits[:2]}-0{digits[2]}-{digits[3:7]}"
    return "-"


# ==============================================================================
# TAHAP 4: PIPELINE UTAMA EKSTRAKSI IDENTITAS KK & ANGGOTA KELUARGA
# ==============================================================================

def run_image_processing_pipeline(image_path: str, output_dir: str = "") -> Dict[str, Any]:
    """
    Menjalankan seluruh tahapan Pemrosesan Citra Digital pada gambar Kartu Keluarga
    dan mengekstrak data ke dalam struktur:
    - identitas (untuk tbl_identitasKK)
    - anggota (untuk tbl_anggotaKeluargaKK)
    """
    img_bgr, gray, binary_inv = preprocess_citra_kk(image_path)
    h, w = gray.shape

    # Resize citra (Pertemuan 2) bila resolusi lebar < 1400 px
    scale = 1500.0 / w if w < 1400 else 1.0
    if scale > 1.05:
        gray_ocr = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        gray_ocr = gray
    ocr_w, ocr_h = gray_ocr.shape[1], gray_ocr.shape[0]
    ocr_result, _ = _OCR_ENGINE(gray_ocr)

    tokens: List[Dict[str, Any]] = []
    if ocr_result:
        for box, text, score in ocr_result:
            txt = str(text).strip()
            if not txt:
                continue
            pts = np.array(box, dtype=np.float32)
            x_min = float(np.min(pts[:, 0])) / ocr_w
            y_min = float(np.min(pts[:, 1])) / ocr_h
            x_max = float(np.max(pts[:, 0])) / ocr_w
            y_max = float(np.max(pts[:, 1])) / ocr_h
            tokens.append(
                {
                    "text": txt,
                    "score": float(score),
                    "nx_min": x_min,
                    "nx": (x_min + x_max) / 2.0,
                    "ny": (y_min + y_max) / 2.0,
                    "h": (y_max - y_min) * h,
                    "x_min": x_min * w,
                    "x_max": x_max * w,
                    "cy": ((y_min + y_max) / 2.0) * h,
                }
            )

    # 1. Ekstrak Header Identitas KK (tbl_identitasKK)
    identitas = _extract_identitas_header(tokens)

    # 2. Ekstrak Anggota Keluarga (tbl_anggotaKeluargaKK)
    parsed_full = _parse_tables_from_full_tokens(tokens)
    if not parsed_full or not parsed_full[0].get("nama_lengkap") or not parsed_full[0].get("nik"):
        anggota = _extract_anggota_via_opencv_cells(gray, binary_inv, identitas)
    else:
        anggota = parsed_full

    if anggota and identitas["nama_kepala_keluarga"]:
        kepala = identitas["nama_kepala_keluarga"]
        nama_1 = anggota[0].get("nama_lengkap", "")
        if not nama_1 or difflib.SequenceMatcher(None, nama_1.replace(" ", ""), kepala.replace(" ", "")).ratio() >= 0.72:
            anggota[0]["nama_lengkap"] = kepala

    return {
        "identitas": identitas,
        "anggota": anggota,
    }


def _group_side_lines(side_tokens: List[Dict[str, Any]], ny_tol: float = 0.009) -> List[List[Dict[str, Any]]]:
    """Mengelompokkan token OCR pada satu sisi (kiri atau kanan) berdasarkan posisi baris vertikal (ny)."""
    lines: List[List[Dict[str, Any]]] = []
    for tok in sorted(side_tokens, key=lambda t: t["ny"]):
        placed = False
        for line in lines:
            if abs(tok["ny"] - float(np.mean([x["ny"] for x in line]))) <= ny_tol:
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])
    for line in lines:
        line.sort(key=lambda t: t["nx_min"])
    return lines


def _extract_identitas_header(tokens: List[Dict[str, Any]]) -> Dict[str, str]:
    """Mengekstrak data identitas KK dari bagian atas (header) dan bawah (footer) citra KK."""
    identitas = {
        "no_kk": "",
        "nama_kepala_keluarga": "",
        "alamat": "",
        "rt_rw": "",
        "kode_pos": "",
        "desa_kelurahan": "",
        "kecamatan": "",
        "kabupaten_kota": "",
        "provinsi": "",
        "tanggal_dikeluarkan": "",
    }

    for tok in sorted(tokens, key=lambda t: t["ny"]):
        if tok["ny"] > 0.25:
            break
        m = re.search(r"(?:No\.?\s*|No[·:]\s*)?([0-9OIlSBZ]{15,17})", tok["text"], re.IGNORECASE)
        if m:
            cand = _clean_digits(m.group(1))
            if len(cand) == 16:
                identitas["no_kk"] = cand
                break

    header_tokens = [t for t in tokens if 0.10 <= t["ny"] <= 0.20]
    left_lines = _group_side_lines([t for t in header_tokens if t["nx_min"] < 0.55])
    right_lines = _group_side_lines([t for t in header_tokens if t["nx_min"] >= 0.55])

    left_label_map = [
        ("nama_kepala_keluarga", r"Nama\s*Kepala\s*Keluarga"),
        ("alamat", r"\bAlamat\b"),
        ("rt_rw", r"RT\s*/?\s*RW"),
        ("kode_pos", r"Ko[do]?e\s*Pos"),
    ]
    right_label_map = [
        ("desa_kelurahan", r"(?:Desa\s*/?\s*Kelurahan|Kelurahan)"),
        ("kecamatan", r"\bKecamatan\b"),
        ("kabupaten_kota", r"(?:Kabupaten\s*/?\s*Kota|Kabupaten)"),
        ("provinsi", r"(?:Provinsi|Propinsi)"),
    ]

    left_values: List[str] = []
    for line in left_lines:
        joined = " ".join(t["text"] for t in line)
        matched = False
        for key, pat in left_label_map:
            m = re.search(pat, joined, re.IGNORECASE)
            if m:
                val = re.sub(r"^[\s:\uff1a.\-]+", "", joined[m.end() :]).strip().upper()
                if val and not identitas[key]:
                    identitas[key] = val
                    matched = True
                break
        if not matched:
            val_toks = [t["text"] for t in line if t["nx_min"] >= 0.22]
            if val_toks:
                val_str = re.sub(r"^[\s:\uff1a.\-]+", "", " ".join(val_toks)).strip().upper()
                if val_str:
                    left_values.append(val_str)

    right_values: List[str] = []
    for line in right_lines:
        joined = " ".join(t["text"] for t in line)
        matched = False
        for key, pat in right_label_map:
            m = re.search(pat, joined, re.IGNORECASE)
            if m:
                val = re.sub(r"^[\s:\uff1a.\-]+", "", joined[m.end() :]).strip().upper()
                if val and not identitas[key]:
                    identitas[key] = val
                    matched = True
                break
        if not matched:
            val_toks = [t["text"] for t in line if t["nx_min"] >= 0.72]
            if val_toks:
                val_str = re.sub(r"^[\s:\uff1a.\-]+", "", " ".join(val_toks)).strip().upper()
                if val_str:
                    right_values.append(val_str)

    for idx, k in enumerate(["nama_kepala_keluarga", "alamat", "rt_rw", "kode_pos"]):
        if not identitas[k] and idx < len(left_values):
            identitas[k] = left_values[idx]

    for idx, k in enumerate(["desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi"]):
        if not identitas[k] and idx < len(right_values):
            identitas[k] = right_values[idx]

    identitas["alamat"] = re.sub(r"^JL(?=[A-Z])", "JL. ", identitas["alamat"]).replace("SOEPRATMIAN", "SOEPRATMAN")
    if identitas["rt_rw"].endswith("-") and "/" not in identitas["rt_rw"]:
        identitas["rt_rw"] = identitas["rt_rw"][:-1] + "/-"
    if identitas["kode_pos"]:
        kp = _clean_digits(identitas["kode_pos"])
        if len(kp) >= 5:
            identitas["kode_pos"] = kp[:5]
    identitas["provinsi"] = identitas["provinsi"].replace("KALIMANTANTIMUR", "KALIMANTAN TIMUR")

    for tok in tokens:
        if tok["ny"] > 0.70:
            m_d = re.search(r"(\d{2}[-/.]\d{2}[-/.]\d{4})", tok["text"])
            if m_d:
                identitas["tanggal_dikeluarkan"] = m_d.group(1).replace("/", "-")
                break

    return identitas


def _extract_anggota_via_opencv_cells(
    gray: np.ndarray, binary_inv: np.ndarray, identitas: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Mengekstrak baris data Tabel 1 (Atas) dan Tabel 2 (Bawah) dengan memotong
    setiap sel tabel menggunakan garis yang dideteksi melalui Morfologi OpenCV.
    """
    horiz_lines, tables_y = _detect_table_grids(gray, binary_inv)
    if len(tables_y) < 2:
        return []

    t1_y, t2_y = tables_y[0], tables_y[1]
    t1_x = _detect_row_columns(binary_inv, t1_y[2], t1_y[3], expected_cols=10)
    t2_x = _detect_row_columns(binary_inv, t2_y[2], t2_y[3], expected_cols=9)

    if len(t1_x) < 10 or len(t2_x) < 9:
        return []

    city_vocab = ["JAKARTA", "BOGOR", "BANDUNG", "SEMARANG", "SURABAYA", "BONTANG", "SAMARINDA", "BALIKPAPAN"]
    if identitas.get("kabupaten_kota"):
        clean_city = re.sub(r"^(KOTA|KABUPATEN)\s+", "", identitas["kabupaten_kota"]).strip()
        if clean_city and clean_city not in city_vocab:
            city_vocab.insert(0, clean_city)

    anggota_list: List[Dict[str, Any]] = []
    max_data_rows = min(len(t1_y) - 3, len(t2_y) - 3)

    for r_idx in range(max_data_rows):
        y1_u, y2_u = t1_y[r_idx + 2], t1_y[r_idx + 3]

        raw_nama = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[1], t1_x[2])
        raw_nik = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[2], t1_x[3])

        nama_clean = re.sub(r"[^A-Z\s.,']", "", raw_nama.upper()).strip()
        nik_digits = _clean_digits(raw_nik)

        if not nama_clean and len(nik_digits) < 10:
            break

        raw_jk = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[3], t1_x[4])
        raw_tempat = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[4], t1_x[5])
        raw_tgl_lahir = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[5], t1_x[6])
        raw_agama = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[6], t1_x[7])
        raw_pendidikan = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[7], t1_x[8])
        raw_pekerjaan = _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[8], t1_x[9])
        raw_goldar = (
            _read_single_cell_ocr(gray, horiz_lines, y1_u, y2_u, t1_x[9], t1_x[10])
            if len(t1_x) >= 11
            else "-"
        )

        tgl_lahir_norm = _format_tanggal(raw_tgl_lahir)
        if identitas.get("no_kk") and len(nik_digits) >= 14:
            kk_prefix = identitas["no_kk"][:6]
            if difflib.SequenceMatcher(None, nik_digits[:6], kk_prefix).ratio() >= 0.5:
                nik_digits = kk_prefix + nik_digits[6:]
            if len(nik_digits) == 15 and nik_digits.endswith("020"):
                nik_digits = nik_digits[:11] + "0" + nik_digits[11:]

        if len(nik_digits) == 16 and tgl_lahir_norm != "-":
            parts_tgl = tgl_lahir_norm.split("-")
            if len(parts_tgl) == 3 and len(parts_tgl[2]) == 4:
                yy_str = parts_tgl[2][2:]
                nik_digits = nik_digits[:10] + yy_str + nik_digits[12:]
        elif len(nik_digits) == 16 and tgl_lahir_norm == "-":
            try:
                dd = int(nik_digits[6:8])
                if dd > 40:
                    dd -= 40
                mm = int(nik_digits[8:10])
                yy = int(nik_digits[10:12])
                if 1 <= dd <= 31 and 1 <= mm <= 12:
                    yyyy = 2000 + yy if yy <= 26 else 1900 + yy
                    tgl_lahir_norm = f"{dd:02d}-{mm:02d}-{yyyy}"
            except ValueError:
                pass

        y1_l, y2_l = t2_y[r_idx + 2], t2_y[r_idx + 3]
        raw_stat_kawin = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[1], t2_x[2])
        raw_tgl_kawin = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[2], t2_x[3])
        raw_stat_hub = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[3], t2_x[4])
        raw_wni = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[4], t2_x[5])
        raw_paspor = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[5], t2_x[6])
        raw_kitap = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[6], t2_x[7])
        raw_ayah = _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[7], t2_x[8])
        raw_ibu = (
            _read_single_cell_ocr(gray, horiz_lines, y1_l, y2_l, t2_x[8], t2_x[9])
            if len(t2_x) >= 10
            else "-"
        )

        jk_norm = "PEREMPUAN" if "P" in raw_jk.upper() and "L" not in raw_jk.upper() else "LAKI-LAKI"
        tempat_norm = _match_closest_vocab(raw_tempat, city_vocab, min_ratio=0.55)
        agama_norm = _match_closest_vocab(raw_agama, LIST_AGAMA, min_ratio=0.35)
        if agama_norm not in LIST_AGAMA and any(ch in raw_agama.upper() for ch in ("I", "S", "L", "A", "M", "R")):
            agama_norm = "ISLAM"
        pendidikan_norm = _match_closest_vocab(raw_pendidikan, LIST_PENDIDIKAN, min_ratio=0.35)
        pekerjaan_norm = _match_closest_vocab(raw_pekerjaan, LIST_PEKERJAAN, min_ratio=0.38)
        if len(re.sub(r"[^A-Z]", "", raw_goldar.upper())) >= 3:
            goldar_norm = "TIDAK TAHU"
        else:
            goldar_norm = _match_closest_vocab(raw_goldar, LIST_GOL_DARAH, min_ratio=0.30)
        stat_kawin_norm = _match_closest_vocab(raw_stat_kawin, LIST_STATUS_KAWIN, min_ratio=0.38)
        tgl_kawin_norm = _format_tanggal(raw_tgl_kawin) if raw_tgl_kawin != "-" else "-"
        stat_hub_norm = _match_closest_vocab(raw_stat_hub, LIST_STATUS_HUBUNGAN, min_ratio=0.38)
        wni_norm = "WNA" if "A" in raw_wni.upper() else "WNI"

        anggota_list.append(
            {
                "no_urut": len(anggota_list) + 1,
                "nama_lengkap": nama_clean,
                "nik": nik_digits[:16],
                "jenis_kelamin": jk_norm,
                "tempat_lahir": tempat_norm,
                "tanggal_lahir": tgl_lahir_norm,
                "agama": agama_norm,
                "pendidikan": pendidikan_norm,
                "jenis_pekerjaan": pekerjaan_norm,
                "golongan_darah": goldar_norm,
                "status_perkawinan": stat_kawin_norm,
                "tanggal_perkawinan": tgl_kawin_norm,
                "status_hubungan": stat_hub_norm,
                "kewarganegaraan": wni_norm,
                "no_paspor": raw_paspor if raw_paspor != "-" else "-",
                "no_kitap": raw_kitap if raw_kitap != "-" else "-",
                "nama_ayah": re.sub(r"[^A-Z\s.,']", "", raw_ayah.upper()).strip() or "-",
                "nama_ibu": re.sub(r"[^A-Z\s.,']", "", raw_ibu.upper()).strip() or "-",
            }
        )

    return anggota_list


def _parse_tables_from_full_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parser baris tabel langsung dari token OCR halaman penuh untuk citra KK beresolusi tinggi."""
    if not tokens:
        return []

    heights = [t["h"] for t in tokens if t["h"] > 4]
    med_h = float(np.median(heights)) if heights else 20.0
    y_tol = max(8.0, med_h * 0.55)

    lines: List[List[Dict[str, Any]]] = []
    for tok in sorted(tokens, key=lambda t: t["cy"]):
        placed = False
        for line in lines:
            if abs(tok["cy"] - float(np.mean([x["cy"] for x in line]))) <= y_tol:
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])

    for line in lines:
        line.sort(key=lambda t: t["x_min"])

    t1_header_ny, t2_header_ny, footer_ny = 0.23, 0.52, 0.82
    for line in lines:
        joined = " ".join(t["text"] for t in line).upper()
        l_ny = float(np.mean([t["ny"] for t in line]))
        if ("NAMA LENGKAP" in joined or "TEMPAT LAHIR" in joined) and l_ny < 0.38:
            t1_header_ny = max(t1_header_ny, l_ny + 0.015)
        elif ("STATUS HUBUNGAN" in joined or "KEWARGANEGARAAN" in joined) and 0.40 <= l_ny <= 0.68:
            t2_header_ny = l_ny
        elif ("DIKELUARKAN" in joined or "KEPALA DINAS" in joined) and l_ny > 0.70:
            footer_ny = min(footer_ny, l_ny - 0.01)

    upper_rows: List[Dict[str, Any]] = []
    lower_rows: List[Dict[str, Any]] = []

    for line in lines:
        l_ny = float(np.mean([t["ny"] for t in line]))
        joined = " ".join(t["text"] for t in line).upper()
        if any(k in joined for k in ["NAMA LENGKAP", "TEMPAT LAHIR", "STATUS HUBUNGAN", "NAMA ORANG", "(1)", "(2)", "(10)", "(12)"]):
            continue

        valid = [t for t in line if not re.fullmatch(r"[-—_.:,\s()0-9]{1,3}", t["text"].strip())]
        if not valid:
            continue

        if t1_header_ny < l_ny < (t2_header_ny - 0.01):
            nik_val, nik_tok = "", None
            for t in line:
                cand = _clean_digits(t["text"])
                if 15 <= len(cand) <= 16 and 0.15 <= t["nx"] <= 0.48:
                    nik_val, nik_tok = cand[:16], t
                    break
            if not nik_val:
                continue

            row_u = {
                "no_urut": len(upper_rows) + 1,
                "nama_lengkap": "",
                "nik": nik_val,
                "jenis_kelamin": "",
                "tempat_lahir": "",
                "tanggal_lahir": "",
                "agama": "",
                "pendidikan": "",
                "jenis_pekerjaan": "",
                "golongan_darah": "-",
            }
            nama_p, tempat_p, pend_p, pek_p = [], [], [], []
            for t in line:
                if t is nik_tok:
                    continue
                txt = t["text"].strip().upper()
                if re.fullmatch(r"[-—_.:,\s]+", txt) or (t["nx"] < 0.055 and txt.isdigit()):
                    continue
                if re.search(r"(LAKI|PEREMPUAN)", txt):
                    row_u["jenis_kelamin"] = "LAKI-LAKI" if "LAKI" in txt else "PEREMPUAN"
                elif re.search(r"\d{2}[-/.]\d{2}[-/.]\d{4}", txt) and 0.45 <= t["nx"] <= 0.70:
                    row_u["tanggal_lahir"] = _format_tanggal(txt)
                elif txt in LIST_AGAMA:
                    row_u["agama"] = txt
                elif t["nx_min"] >= 0.90:
                    row_u["golongan_darah"] = "O" if txt == "0" else txt
                elif t["nx"] < 0.23 or (nik_tok and t["x_max"] <= nik_tok["x_min"] + 10):
                    nama_p.append(txt)
                elif 0.36 <= t["nx"] < 0.56:
                    tempat_p.append(txt)
                elif 0.63 <= t["nx_min"] < 0.805:
                    pend_p.append(txt)
                elif 0.805 <= t["nx_min"] < 0.90:
                    pek_p.append(txt)

            row_u["nama_lengkap"] = " ".join(nama_p).strip()
            row_u["tempat_lahir"] = " ".join(tempat_p).strip()
            row_u["pendidikan"] = " ".join(pend_p).strip()
            row_u["jenis_pekerjaan"] = " ".join(pek_p).strip()
            if row_u["nama_lengkap"] and row_u["nik"]:
                upper_rows.append(row_u)

        elif (t2_header_ny + 0.02) < l_ny < footer_ny:
            row_l = {
                "status_perkawinan": "",
                "tanggal_perkawinan": "-",
                "status_hubungan": "",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "",
                "nama_ibu": "",
            }
            sk_p, sh_p, pas_p, kit_p, ay_p, ib_p = [], [], [], [], [], []
            for t in line:
                txt = t["text"].strip().upper()
                if re.fullmatch(r"[-—_.:,\s]+", txt) or (t["nx"] < 0.055 and txt.isdigit()):
                    continue
                if re.search(r"\d{2}[-/.]\d{2}[-/.]\d{4}", txt) and 0.14 <= t["nx"] <= 0.30:
                    row_l["tanggal_perkawinan"] = _format_tanggal(txt)
                elif txt in ("WNI", "WNA"):
                    row_l["kewarganegaraan"] = txt
                elif t["nx_min"] < 0.16:
                    sk_p.append(txt)
                elif 0.24 <= t["nx_min"] < 0.40:
                    sh_p.append(txt)
                elif 0.48 <= t["nx_min"] < 0.60:
                    pas_p.append(txt)
                elif 0.60 <= t["nx_min"] < 0.70:
                    kit_p.append(txt)
                elif 0.70 <= t["nx_min"] < 0.835:
                    ay_p.append(txt)
                elif t["nx_min"] >= 0.835:
                    ib_p.append(txt)

            row_l["status_perkawinan"] = " ".join(sk_p).strip()
            row_l["status_hubungan"] = " ".join(sh_p).strip()
            if pas_p:
                row_l["no_paspor"] = " ".join(pas_p).strip()
            if kit_p:
                row_l["no_kitap"] = " ".join(kit_p).strip()
            row_l["nama_ayah"] = " ".join(ay_p).strip()
            row_l["nama_ibu"] = " ".join(ib_p).strip()
            if row_l["status_perkawinan"] or row_l["status_hubungan"] or row_l["nama_ayah"]:
                lower_rows.append(row_l)

    merged: List[Dict[str, Any]] = []
    for idx, u in enumerate(upper_rows):
        item = dict(u)
        if idx < len(lower_rows):
            item.update(lower_rows[idx])
        merged.append(item)
    return merged

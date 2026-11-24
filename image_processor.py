"""
Modul Pemrosesan Citra Digital & Ekstraksi Kartu Keluarga (image_processor.py)
------------------------------------------------------------------------------
Implementasi ini menerapkan seluruh materi pada mata kuliah Pemrosesan Citra Digital
sekaligus mengoptimalkan pembacaan foto Kartu Keluarga nyata (miring/perspektif,
pencahayaan redup/tidak merata, pantulan kilap, maupun sedikit kabur):

1. Akuisisi Citra, Properti Matriks, & Konversi Kanal:
   - Membaca citra ke matriks NumPy (`shape`, `dtype`, `size`), konversi `BGR -> RGB`,
     pemisahan kanal `R, G, B`, serta ruang warna `LAB` untuk segmentasi kertas KK.
2. Pelurusan Dokumen Otomatis (Document Rectification & Deskew):
   - Deteksi bidang kertas 4-sudut (*Convex Hull* + `cv2.approxPolyDP`) & transformasi
     perspektif (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`).
   - Deteksi sudut kemiringan garis tabel via *Hough Line Transform* (`cv2.HoughLinesP`),
     rotasi otomatis (`cv2.warpAffine`), dan pelurusan trapesium berbasis grid tabel.
3. Konversi Grayscale, Normalisasi Pencahayaan, Histogram & Contrast Stretching:
   - Rumus luminositas berbobot `Gray = 0.299R + 0.587G + 0.114B` & `cv2.COLOR_BGR2GRAY`.
   - Perhitungan histogram (`np.bincount`), normalisasi `h_i = n_i / n`, rata-rata (`u`),
     standar deviasi kontras (`d`), perataan latar (*illumination normalization*), dan
     *Contrast Stretching* berbasis persentil (`np.percentile`).
4. Operasi Aras Titik (*Point Processing*) & Aras Lokal (*Neighborhood Processing*):
   - Aras Titik: *Brightening*, *Citra Negatif*, serta *Global & Adaptive Thresholding*.
   - Aras Lokal: *Mean Filter* 3x3 (`cv2.blur`), *Bilateral Filter*, dan *Sharpening*
     (kernel Laplacian 3x3 & *Unsharp Masking*).
5. Morfologi Citra Biner, Slicing ROI Sel Tabel, & Column-Strip OCR:
   - *Structuring Element* garis horizontal/vertikal dengan Erosi, Dilasi, dan Opening
     untuk memetakan seluruh baris & kolom Tabel 1 (Atas) dan Tabel 2 (Bawah).
   - Pemotongan sel dengan *local border snapping* (mengikuti lengkungan kertas),
     perbesaran `cv2.resize` (`INTER_CUBIC`), dan pembacaan teks RapidOCR terkalibrasi.
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Engine OCR halaman penuh (header/footer/fallback) & strip per-kolom tabel.
# use_cls=False mencegah RapidOCR membalik angka 16 digit (No. KK / NIK) 180 derajat.
# det_limit_type="max" mencegah scaling berlebihan pada potongan tabel lebar.
_OCR_ENGINE = RapidOCR(
    use_cls=False,
    det_limit_type="max",
    det_limit_side_len=2400,
    max_side_len=2600,
    det_thresh=0.18,
    det_box_thresh=0.25,
    det_unclip_ratio=1.45,
    text_score=0.20,
)

_OCR_STRIP = RapidOCR(
    use_cls=False,
    det_limit_type="max",
    det_limit_side_len=1280,
    max_side_len=1600,
    det_thresh=0.18,
    det_box_thresh=0.24,
    det_unclip_ratio=1.55,
    text_score=0.15,
)


# ==============================================================================
# DAFTAR KOSAKATA STANDAR KARTU KELUARGA (UNTUK VALIDASI HASIL BACA)
# ==============================================================================

LIST_AGAMA = [
    "ISLAM",
    "KRISTEN",
    "KATHOLIK",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "KHONGHUCU",
    "KEPERCAYAAN",
]

LIST_PENDIDIKAN = [
    "TIDAK/BELUM SEKOLAH",
    "BELUM TAMAT SD/SEDERAJAT",
    "TAMAT SD/SEDERAJAT",
    "SLTP/SEDERAJAT",
    "SLTA/SEDERAJAT",
    "DIPLOMA I/II",
    "DIPLOMA III",
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
    "NELAYAN/PERIKANAN",
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
    "KAWIN BELUM TERCATAT",
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

COMMON_NAME_TOKENS = [
    "BUDI", "SANTOSO", "DEWI", "LESTARI", "RAFI", "PRATAMA", "NADIA", "PUTRI",
    "HENDRA", "SITI", "AMINAH", "AHMAD", "HIDAYAT", "RATNA", "SARI",
    "IRWAN", "SETIAWAN", "RINA", "WULANDARI", "KEVIN", "ADITYA",
    "SOEHARTO", "MARYAM", "BAMBANG", "SUTRISNO", "SRI", "WAHYUNI",
    "MUHAMMAD", "ABDUL", "PUTRA", "AGUS", "ANDI", "THOMAS", "PAULUS", "PAULINA",
]

LIST_PROVINSI = [
    "ACEH",
    "SUMATERA UTARA",
    "SUMATERA BARAT",
    "RIAU",
    "KEPULAUAN RIAU",
    "JAMBI",
    "SUMATERA SELATAN",
    "KEPULAUAN BANGKA BELITUNG",
    "BENGKULU",
    "LAMPUNG",
    "DKI JAKARTA",
    "JAWA BARAT",
    "BANTEN",
    "JAWA TENGAH",
    "DI YOGYAKARTA",
    "JAWA TIMUR",
    "BALI",
    "NUSA TENGGARA BARAT",
    "NUSA TENGGARA TIMUR",
    "KALIMANTAN BARAT",
    "KALIMANTAN TENGAH",
    "KALIMANTAN SELATAN",
    "KALIMANTAN TIMUR",
    "KALIMANTAN UTARA",
    "SULAWESI UTARA",
    "GORONTALO",
    "SULAWESI TENGAH",
    "SULAWESI BARAT",
    "SULAWESI SELATAN",
    "SULAWESI TENGGARA",
    "MALUKU",
    "MALUKU UTARA",
    "PAPUA",
    "PAPUA BARAT",
    "PAPUA BARAT DAYA",
    "PAPUA SELATAN",
    "PAPUA TENGAH",
    "PAPUA PEGUNUNGAN",
]

COMMON_LOCATION_TOKENS = [
    "KOTA", "KABUPATEN", "DESA", "DUSUN", "KELURAHAN", "KECAMATAN",
    "TIMUR", "BARAT", "UTARA", "SELATAN", "TENGAH", "PUSAT", "DAYA",
    "JAWA", "SUMATERA", "KALIMANTAN", "SULAWESI", "NUSA", "TENGGARA",
    "PAPUA", "MALUKU", "BALI", "ACEH", "BANTEN", "DKI", "LAMPUNG", "RIAU",
    "JAKARTA", "BOGOR", "BANDUNG", "SEMARANG", "SURABAYA", "BONTANG",
    "SAMARINDA", "BALIKPAPAN", "TEBET", "BARANANGSIANG",
    "SUMBA", "KODI", "SERDANG", "BEDAGAI", "BANDAR", "KHALIPAH",
    "TANJUNG", "BALAI", "KUALA", "TEBING", "TINGGI", "SILAU", "BUOL",
    "GUAMONIAL", "MOMUNU", "RADA", "GUNDU", "TELAGA",
]

# Proporsi standar kolom Blanko KK Permendagri 109/2019 (dipakai hanya jika ada garis vertikal tertutup pantulan kilap)
_T1_COL_RATIOS = [0.000, 0.020, 0.224, 0.327, 0.380, 0.511, 0.572, 0.643, 0.778, 0.947, 1.000]
_T2_COL_RATIOS = [0.000, 0.020, 0.116, 0.178, 0.294, 0.391, 0.474, 0.556, 0.792, 1.000]


def _normalize_spaced_text(text: str, vocab_tokens: List[str]) -> str:
    """Memisahkan kata-kata yang menempel akibat pembacaan OCR (contoh: BUDISANTOSO -> BUDI SANTOSO)."""
    if not text or text == "-":
        return text
    val = re.sub(r"\s+", " ", text.upper()).strip()
    for tok in sorted(vocab_tokens, key=len, reverse=True):
        val = re.sub(rf"\b({tok})([A-Z]{{3,}})\b", r"\1 \2", val)
        val = re.sub(rf"\b([A-Z]{{3,}})({tok})\b", r"\1 \2", val)
    return re.sub(r"\s+", " ", val).strip()


def _strip_non_latin(text: str, allow_chars: str = r"A-Z0-9\s.,/'-") -> str:
    """Menghapus artefak karakter non-Latin (CJK/simbol acak) yang kadang muncul pada teks sangat kabur."""
    cleaned = re.sub(rf"[^{allow_chars}]", " ", text.upper())
    return re.sub(r"\s+", " ", cleaned).strip()


# ==============================================================================
# TAHAP 0: PELURUSAN DOKUMEN OTOMATIS (PERSPECTIVE WARP & HOUGH DESKEW)
# ==============================================================================

def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Mengurutkan 4 titik sudut menjadi [kiri-atas, kanan-atas, kanan-bawah, kiri-bawah]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _find_paper_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Mendeteksi area kertas putih Kartu Keluarga menggunakan ruang warna LAB (kecerahan tinggi, kroma rendah)."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    chroma = np.sqrt((A.astype(np.float32) - 128.0) ** 2 + (B.astype(np.float32) - 128.0) ** 2)
    k = max(15, (min(h, w) // 25) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    L_closed = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel)
    score = np.clip(L_closed.astype(np.float32) - 0.8 * chroma, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return mask


def _detect_dominant_line_angle(gray: np.ndarray) -> float:
    """Mendeteksi sudut kemiringan dominan garis tabel KK dalam rentang [-45, +45) derajat via HoughLinesP."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=70,
        minLineLength=int(min(h, w) * 0.14),
        maxLineGap=15,
    )
    if lines is None:
        return 0.0
    angles: List[float] = []
    weights: List[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        ang = float(np.degrees(np.arctan2(dy, dx)))
        while ang >= 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        angles.append(ang)
        weights.append(length)
    if not angles:
        return 0.0
    bins = np.arange(-45.0, 45.5, 0.5)
    hist, edges_b = np.histogram(angles, bins=bins, weights=weights)
    peak_idx = int(np.argmax(hist))
    peak_center = (edges_b[peak_idx] + edges_b[peak_idx + 1]) / 2.0
    near = [a for a, wt in zip(angles, weights) if abs(a - peak_center) <= 2.5]
    near_w = [wt for a, wt in zip(angles, weights) if abs(a - peak_center) <= 2.5]
    return float(np.average(near, weights=near_w)) if near else 0.0


def _rotate_bound_with_matrix(img: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Menghitung matriks rotasi 3x3 homogen dan ukuran kanvas tanpa memotong sudut dokumen."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    return M3, (nw, nh)


def _rotate_bound(img: np.ndarray, angle_deg: float, border_val: Any = (0, 0, 0)) -> np.ndarray:
    """Memutar citra tanpa memotong sudut dokumen."""
    if abs(angle_deg) < 0.35:
        return img
    M3, (nw, nh) = _rotate_bound_with_matrix(img, angle_deg)
    return cv2.warpAffine(img, M3[:2], (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border_val)


def _segment_intersection(seg1: Tuple[float, float, float, float], seg2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = seg1
    x3, y3, x4, y4 = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (float(px), float(py))


def _segs_cross(
    hseg: Tuple[float, float, float, float],
    vseg: Tuple[float, float, float, float],
    pad: float = 12.0,
) -> bool:
    """Memeriksa apakah dua segmen garis (horizontal & vertikal) benar-benar saling berpotongan."""
    pt = _segment_intersection(hseg, vseg)
    if pt is None:
        return False
    px, py = pt
    hx1, hy1, hx2, hy2 = hseg
    vx1, vy1, vx2, vy2 = vseg
    return (
        min(hx1, hx2) - pad <= px <= max(hx1, hx2) + pad
        and min(hy1, hy2) - pad <= py <= max(hy1, hy2) + pad
        and min(vx1, vx2) - pad <= px <= max(vx1, vx2) + pad
        and min(vy1, vy2) - pad <= py <= max(vy1, vy2) + pad
    )


def _find_table_quad_homography(
    img_bgr: np.ndarray, target_w: int = 2400, target_h: int = 1700
) -> Optional[np.ndarray]:
    """
    Mendeteksi 4 garis batas terluar (atas, bawah, kiri, kanan) dari grid tabel KK
    menggunakan HoughLinesP yang tahan kemiringan perspektif (±22°), menyaring outlier sudut,
    memulihkan batas kiri/kanan terluar dari ujung garis horizontal bila garis vertikal terluar
    pudar akibat sudut pandang/kilap, lalu mengembalikan matriks homografi 3x3.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bin_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9)
    non_black = cv2.erode((gray > 8).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    bin_inv = cv2.bitwise_and(bin_inv, non_black)

    min_h_len = int(w * 0.22)
    min_v_len = int(h * 0.06)
    lines_h = cv2.HoughLinesP(bin_inv, 1, np.pi / 360, threshold=70, minLineLength=min_h_len, maxLineGap=30)
    lines_v = cv2.HoughLinesP(bin_inv, 1, np.pi / 360, threshold=45, minLineLength=min_v_len, maxLineGap=25)
    if lines_h is None or lines_v is None:
        return None

    h_cands: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in lines_h.reshape(-1, 4):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang > 90:
            ang = 180 - ang
        if ang <= 22.0:
            h_cands.append((float(x1), float(y1), float(x2), float(y2)))

    v_cands: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in lines_v.reshape(-1, 4):
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang > 90:
            ang = 180 - ang
        if ang >= 68.0:
            v_cands.append((float(x1), float(y1), float(x2), float(y2)))

    # Pertahankan hanya garis yang saling berpotongan (menolak garis ubin lantai/meja)
    true_h = [hs for hs in h_cands if sum(1 for vs in v_cands if _segs_cross(hs, vs)) >= 3]
    true_v = [vs for vs in v_cands if sum(1 for hs in true_h if _segs_cross(hs, vs)) >= 3]
    if len(true_h) < 4 or len(true_v) < 3:
        return None

    # Buang segmen horizontal outlier yang sudutnya menyimpang jauh dari kemiringan dominan grid
    signed_h_angs = [float(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0]))) for s in true_h]
    med_h_ang = float(np.median(signed_h_angs))
    clean_h = [s for s, a in zip(true_h, signed_h_angs) if abs(a - med_h_ang) <= 7.0]
    if len(clean_h) >= 4:
        true_h = clean_h

    all_xs = [s[0] for s in true_h] + [s[2] for s in true_h]
    all_ys = [s[1] for s in true_h] + [s[3] for s in true_h]
    gx0, gx1 = min(all_xs), max(all_xs)
    gy0, gy1 = min(all_ys), max(all_ys)
    gw, gh = max(1.0, gx1 - gx0), max(1.0, gy1 - gy0)

    top_seg = min(true_h, key=lambda s: (s[1] + s[3]) / 2.0)
    bot_seg = max(true_h, key=lambda s: (s[1] + s[3]) / 2.0)
    left_v = min(true_v, key=lambda s: (s[0] + s[2]) / 2.0)
    right_v = max(true_v, key=lambda s: (s[0] + s[2]) / 2.0)

    def _fit_side_from_h_endpoints(is_right: bool) -> Optional[Tuple[float, float, float, float]]:
        mid_y = (gy0 + gy1) / 2.0
        upper = [
            (s[2], s[3]) if is_right else (s[0], s[1])
            for s in true_h
            if ((s[3] if is_right else s[1]) <= mid_y)
        ]
        lower = [
            (s[2], s[3]) if is_right else (s[0], s[1])
            for s in true_h
            if ((s[3] if is_right else s[1]) > mid_y)
        ]
        if not upper or not lower:
            return None
        upper.sort(key=lambda p: p[0], reverse=is_right)
        lower.sort(key=lambda p: p[0], reverse=is_right)
        ku = max(1, min(4, len(upper) // 5 + 1))
        kl = max(1, min(4, len(lower) // 5 + 1))
        ux = float(np.median([p[0] for p in upper[:ku]]))
        uy = float(np.median([p[1] for p in upper[:ku]]))
        lx = float(np.median([p[0] for p in lower[:kl]]))
        ly = float(np.median([p[1] for p in lower[:kl]]))
        if abs(ly - uy) < gh * 0.15:
            return None
        return (ux, uy, lx, ly)

    # Cegah pemotongan sisi kanan/kiri dokumen bila garis vertikal terluar gagal terdeteksi
    left_seg = left_v
    if ((left_v[0] + left_v[2]) / 2.0 - gx0) > gw * 0.06:
        repaired_l = _fit_side_from_h_endpoints(False)
        if repaired_l is not None:
            left_seg = repaired_l

    right_seg = right_v
    if (gx1 - (right_v[0] + right_v[2]) / 2.0) > gw * 0.06:
        repaired_r = _fit_side_from_h_endpoints(True)
        if repaired_r is not None:
            right_seg = repaired_r

    tl = _segment_intersection(top_seg, left_seg)
    tr = _segment_intersection(top_seg, right_seg)
    br = _segment_intersection(bot_seg, right_seg)
    bl = _segment_intersection(bot_seg, left_seg)
    if None in (tl, tr, br, bl):
        return None

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    for px, py in pts:
        if not (gx0 - gw * 0.30 <= px <= gx1 + gw * 0.30 and gy0 - gh * 0.30 <= py <= gy1 + gh * 0.30):
            return None

    vl, vr = pts[3] - pts[0], pts[2] - pts[1]
    ht, hb = pts[1] - pts[0], pts[2] - pts[3]
    ext = np.array(
        [
            pts[0] - 0.44 * vl - 0.03 * ht,
            pts[1] - 0.44 * vr + 0.03 * ht,
            pts[2] + 0.28 * vr + 0.03 * hb,
            pts[3] + 0.28 * vl - 0.03 * hb,
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(ext, dst).astype(np.float64)


def _warp_by_table_hough_quad(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Meluruskan perspektif dokumen berdasarkan matriks homografi grid tabel KK."""
    H = _find_table_quad_homography(img_bgr, target_w=2400, target_h=1700)
    if H is None:
        return None
    return cv2.warpPerspective(
        img_bgr,
        H,
        (2400, 1700),
        flags=cv2.INTER_CUBIC,
        borderValue=(245, 245, 245),
    )


def rectify_dokumen_kk(img_bgr: np.ndarray) -> np.ndarray:
    """
    Meluruskan foto dokumen KK secara otomatis:
    - Jika citra sudah berupa scan/dokumen lurus penuh, tidak dilakukan distorsi perspektif.
    - Jika kertas KK terlihat utuh pada meja/lantai, diterapkan 4-point Perspective Warp.
    - Jika foto miring ekstrem atau sudut kertas terpotong, diterapkan rotasi HoughLinesP
      diikuti komposisi homografi tunggal (`H_total = H2 @ H1 @ M_rot`) agar ketajaman
      teks tetap terjaga tanpa blur interpolasi berulang.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)
    gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask = _find_paper_mask(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        ratio = cv2.contourArea(cnt) / img_area

        # Dokumen scan / sintetis yang sudah memenuhi frame
        if ratio > 0.94:
            ang = _detect_dominant_line_angle(gray_raw)
            out = _rotate_bound(img_bgr, ang, border_val=(248, 251, 255)) if abs(ang) >= 0.6 else img_bgr
            if out.shape[1] < 2000:
                sc = 2200.0 / out.shape[1]
                out = cv2.resize(out, (2200, int(out.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            return out

        # Deteksi 4 sudut kertas KK (Paper Quad Warp)
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.015, 0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx
                break

        if quad is not None and 0.25 <= ratio <= 0.96:
            pts = _order_quad_points(quad)
            w_top = float(np.linalg.norm(pts[1] - pts[0]))
            w_bot = float(np.linalg.norm(pts[2] - pts[3]))
            h_left = float(np.linalg.norm(pts[3] - pts[0]))
            h_right = float(np.linalg.norm(pts[2] - pts[1]))
            out_w, out_h = int(max(w_top, w_bot)), int(max(h_left, h_right))
            aspect = out_w / max(out_h, 1)
            angles_ok = True
            for i in range(4):
                v1 = pts[(i - 1) % 4] - pts[i]
                v2 = pts[(i + 1) % 4] - pts[i]
                cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                if not (55.0 <= deg <= 125.0):
                    angles_ok = False
                    break

            if 1.15 <= aspect <= 1.85 and angles_ok:
                target_w = 2400
                target_h = int(round(target_w / aspect))
                dst = np.array(
                    [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
                    dtype=np.float32,
                )
                return cv2.warpPerspective(
                    img_bgr,
                    cv2.getPerspectiveTransform(pts, dst),
                    (target_w, target_h),
                    flags=cv2.INTER_CUBIC,
                )

    # Fallback: rotasi sudut dominan garis + pelurusan trapesium grid tabel dengan komposisi matriks tunggal
    ang = _detect_dominant_line_angle(gray_raw)
    if abs(ang) >= 0.35:
        M_rot, (rw, rh) = _rotate_bound_with_matrix(img_bgr, ang)
        rotated = cv2.warpAffine(img_bgr, M_rot[:2], (rw, rh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    else:
        M_rot = np.eye(3, dtype=np.float64)
        rotated = img_bgr

    H1 = _find_table_quad_homography(rotated, target_w=2400, target_h=1700)
    if H1 is not None:
        H_comp1 = H1 @ M_rot
        warped = cv2.warpPerspective(
            img_bgr, H_comp1, (2400, 1700), flags=cv2.INTER_CUBIC, borderValue=(245, 245, 245)
        )
        # Pass penyempurnaan kedua (dikomposisi langsung ke citra asal tanpa double-resample blur)
        enh_check, bin_check = _build_enhanced_views(warped)
        _, t1_c, t2_c, _ = _detect_kk_table_rows(enh_check, bin_check)
        if t1_c is None or t2_c is None:
            H2 = _find_table_quad_homography(warped, target_w=2400, target_h=1700)
            if H2 is not None:
                return cv2.warpPerspective(
                    img_bgr,
                    H2 @ H_comp1,
                    (2400, 1700),
                    flags=cv2.INTER_CUBIC,
                    borderValue=(245, 245, 245),
                )
        return warped

    if rotated.shape[1] < 2000:
        sc = 2200.0 / rotated.shape[1]
        rotated = cv2.resize(rotated, (2200, int(rotated.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
    return rotated


# ==============================================================================
# TAHAP 1: PEMROSESAN CITRA DIGITAL (PERTEMUAN 1, 2, 3, 4, 6, 8)
# ==============================================================================

def analisis_histogram_dan_contrast_stretching(gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    [Materi Pertemuan 6 - Histogram Citra & Contrast Stretching]
    Menghitung histogram (np.bincount), normalisasi h_i = n_i / n, rata-rata kecerahan (u),
    standar deviasi kontras (d), serta Contrast Stretching berbasis persentil 2% dan 98%.
    """
    hist_counts = np.bincount(gray.ravel(), minlength=256)
    total_pixels = float(gray.size)
    hist_norm = hist_counts / total_pixels

    levels = np.arange(256, dtype=np.float64)
    mean_u = float(np.sum(levels * hist_norm))
    variance_v = float(np.sum(((levels - mean_u) ** 2) * hist_norm))
    std_d = float(np.sqrt(variance_v))

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
    Menerapkan Brightening, Citra Negatif, Smoothing (Mean Filter 3x3),
    Sharpening (Kernel Laplacian 3x3), serta Global & Adaptive Thresholding.
    """
    b = 20 if mean_brightness < 145 else 0
    img_bright = np.clip(gray.astype(np.int32) + b, 0, 255).astype(np.uint8)
    img_negatif = 255 - img_bright
    img_smooth = cv2.blur(img_bright, (3, 3))

    kernel_laplacian = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    img_sharp = cv2.filter2D(img_smooth, -1, kernel_laplacian)

    _, binary_global = cv2.threshold(img_bright, 128, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        10,
    )
    return {
        "bright": img_bright,
        "negatif": img_negatif,
        "smooth": img_smooth,
        "sharp": img_sharp,
        "binary_global": binary_global,
        "binary_inv": binary_inv,
    }


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """Meratakan pencahayaan redup, bayangan, dan gradasi warna kertas menggunakan pembagian latar halus."""
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), 9)
    bg = cv2.resize(bg_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip((gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 248.0, 0, 255).astype(np.uint8)


def _build_enhanced_views(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Menghasilkan citra grayscale yang diratakan pencahayaannya & dipertajam halus beserta citra biner tabel."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    norm = _illumination_normalize(gray)
    den = cv2.bilateralFilter(norm, 5, 20, 20)
    p1, p99 = np.percentile(den, (1, 99))
    if p99 - p1 > 15:
        stretched = np.clip((den.astype(np.float32) - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)
    else:
        stretched = den
    blur = cv2.GaussianBlur(stretched, (0, 0), 1.05)
    sharp = cv2.addWeighted(stretched, 1.45, blur, -0.45, 0)
    bin_inv = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 10)
    return sharp, bin_inv


def preprocess_citra_kk(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [Materi Pertemuan 1, 2, 3, 6]
    Membaca citra KK, memeriksa properti matriks, melakukan konversi RGB -> Grayscale berbobot,
    meluruskan dokumen secara otomatis, meratakan pencahayaan, dan menjalankan operasi histogram
    serta aras titik/lokal.
    """
    raw_bytes = np.fromfile(image_path, dtype=np.uint8)
    img_bgr_raw = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img_bgr_raw is None:
        raise ValueError(f"Gagal membaca file citra: {image_path}")

    # Properti matriks citra & konversi kanal RGB -> Grayscale manual (Pertemuan 1 & 2)
    _tinggi, _lebar, _channel = img_bgr_raw.shape
    _dtype = img_bgr_raw.dtype
    _total_pixel = img_bgr_raw.size

    img_rgb = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    _gray_manual = np.round(0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    # Pelurusan dokumen otomatis (Perspective Warp / Hough Deskew)
    img_bgr = rectify_dokumen_kk(img_bgr_raw)
    gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Normalisasi pencahayaan + Histogram & Contrast Stretching (Pertemuan 6)
    norm_gray = _illumination_normalize(gray_cv)
    stretched, info_hist = analisis_histogram_dan_contrast_stretching(norm_gray)

    # Operasi Aras Titik & Aras Lokal (Pertemuan 3)
    hasil_operasi = operasi_aras_titik_dan_lokal(stretched, info_hist["mean_brightness"])
    _ = (_tinggi, _lebar, _channel, _dtype, _total_pixel, _gray_manual, hasil_operasi["negatif"])

    return img_bgr, norm_gray, hasil_operasi["binary_inv"]


# ==============================================================================
# TAHAP 2: MORFOLOGI CITRA BINER & SEGMENTASI GRID TABEL KK (PERTEMUAN 4 & 8)
# ==============================================================================

def _cluster_positions(positions: np.ndarray, max_gap: int = 10) -> List[int]:
    """Mengelompokkan koordinat piksel garis yang berdekatan menjadi satu titik pusat garis."""
    groups: List[List[int]] = []
    for p in positions:
        val = int(p)
        if not groups or val - groups[-1][-1] > max_gap:
            groups.append([val])
        else:
            groups[-1].append(val)
    return [int(np.mean(g)) for g in groups]


def _repair_missing_row_lines(lines: List[int], row_h: float) -> List[int]:
    """Menyisipkan titik tengah apabila 1 garis baris tipis hilang karena blur/kilap."""
    if len(lines) < 3:
        return lines
    out = [lines[0]]
    for y in lines[1:]:
        gap = y - out[-1]
        if 1.72 * row_h <= gap <= 2.38 * row_h:
            out.append(int(round((out[-1] + y) / 2.0)))
        out.append(y)
    return out


def _detect_kk_table_rows(
    gray: np.ndarray, binary_inv: np.ndarray
) -> Tuple[np.ndarray, Optional[List[int]], Optional[List[int]], Tuple[int, int]]:
    """
    [Materi Pertemuan 8 - Morfologi Citra Biner: Erosi, Dilasi, Opening]
    Mendeteksi garis horizontal baris data Tabel 1 (Atas) dan Tabel 2 (Bawah)
    beserta rentang horizontal [tx0, tx1] tabel, sembari menolak celah putih
    kosong antar-tabel agar baris 1 tidak tergeser.
    """
    h, w = gray.shape
    # Dilasi vertikal 3px menjembatani lengkungan kertas halus sebelum Opening horizontal
    b_v = cv2.dilate(binary_inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, w // 9), 1))
    eroded_h = cv2.erode(b_v, horiz_kernel, iterations=1)
    horiz_lines = cv2.dilate(eroded_h, horiz_kernel, iterations=1)

    y_proj = np.sum(horiz_lines > 0, axis=1)
    ys = [
        y
        for y in _cluster_positions(np.where(y_proj > w * 0.35)[0], max_gap=10)
        if int(h * 0.14) < y < int(h * 0.88)
    ]
    if len(ys) < 12:
        return horiz_lines, None, None, (0, w)

    x_span = np.where(np.sum(horiz_lines[ys[0] : ys[-1] + 1, :] > 0, axis=0) >= 6)[0]
    tx0, tx1 = (int(x_span[0]), int(x_span[-1])) if len(x_span) else (0, w)

    diffs = np.diff(ys)
    row_h = float(np.percentile(diffs, 35))

    def _is_inter_table_gap(y_a: int, y_b: int) -> bool:
        """Mendeteksi celah kosong antar Tabel 1 dan Tabel 2 secara toleran terhadap kemiringan tipis."""
        band = binary_inv[max(0, y_a + 3) : min(h, y_b - 3), tx0 + 30 : max(tx0 + 60, tx1 - 30)]
        bh, bw = band.shape
        if bh < 8 or bw < 150:
            return False
        empty_thirds = 0
        for s in range(3):
            sub = band[:, (s * bw) // 3 : ((s + 1) * bw) // 3]
            row_ink = np.mean(sub > 0, axis=1)
            if int(np.sum(row_ink < 0.015)) >= 3:
                empty_thirds += 1
        return empty_thirds >= 2

    is_data = [
        (0.72 * row_h <= d <= 1.32 * row_h) and not _is_inter_table_gap(ys[idx], ys[idx + 1])
        for idx, d in enumerate(diffs)
    ]

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(diffs):
        if is_data[i]:
            start = i
            while i < len(diffs) and (
                is_data[i]
                or (1.72 * row_h <= diffs[i] <= 2.38 * row_h and i + 1 < len(diffs) and is_data[i + 1])
            ):
                i += 1
            if i - start >= 5:
                runs.append((start, i))
        else:
            i += 1

    if len(runs) >= 2:
        r1_s, r1_e = runs[0]
        r2_s, r2_e = runs[-1]

        t1_y = ys[r1_s : r1_e + 1]
        if r1_s >= 1 and 1.72 * row_h <= (ys[r1_s] - ys[r1_s - 1]) <= 2.45 * row_h:
            t1_y = [int(round(ys[r1_s] - row_h))] + t1_y
        t1_y = _repair_missing_row_lines(t1_y, row_h)

        t2_y = ys[r2_s : r2_e + 1]
        if r2_s >= 1 and 1.72 * row_h <= (ys[r2_s] - ys[r2_s - 1]) <= 2.45 * row_h:
            t2_y = [int(round(ys[r2_s] - row_h))] + t2_y
        t2_y = _repair_missing_row_lines(t2_y, row_h)

        if len(t1_y) > 11:
            t1_y = t1_y[-11:]
        if len(t2_y) > 11:
            t2_y = t2_y[-11:]
        return horiz_lines, t1_y, t2_y, (tx0, tx1)

    return horiz_lines, None, None, (tx0, tx1)


def _snap_columns_with_ratios(
    detected_xs: List[int], tx0: int, tx1: int, ratios: List[float]
) -> List[int]:
    """
    Memetakan garis pembatas kolom vertikal nyata ke slot kolom blanko KK via Dynamic Programming
    sehingga SELURUH garis vertikal nyata dipertahankan (tidak tergeser oleh variasi margin blanko),
    dan hanya garis yang benar-benar hilang akibat kilap/blur yang diinterpolasi secara proporsional.
    """
    n = len(ratios)
    span = max(1, tx1 - tx0)
    template = [int(round(tx0 + r * span)) for r in ratios]
    if not detected_xs:
        return template

    # Gabungkan garis duplikat yang sangat rapat di tepi kanan/kiri tabel
    min_col_gap = max(24, int(span * 0.015))
    cleaned: List[int] = [detected_xs[0]]
    for x in detected_xs[1:]:
        if x - cleaned[-1] >= min_col_gap:
            cleaned.append(x)
        elif abs(x - tx1) < abs(cleaned[-1] - tx1):
            cleaned[-1] = x
    detected_xs = cleaned

    if len(detected_xs) == n:
        return detected_xs

    if len(detected_xs) > n:
        snapped = []
        tol = int(span * 0.045)
        for t_x in template:
            nearest = min(detected_xs, key=lambda x: abs(x - t_x))
            snapped.append(nearest if abs(nearest - t_x) <= tol else t_x)
        for i in range(1, len(snapped)):
            if snapped[i] <= snapped[i - 1] + 10:
                snapped[i] = template[i]
        return snapped

    # DP alignment: petakan k garis nyata ke n slot kolom dengan mempertahankan pola lebar antar-kolom
    k = len(detected_xs)
    det_r = [(x - tx0) / float(span) for x in detected_xs]
    INF = 1e9
    dp = [[INF] * n for _ in range(k)]
    parent = [[-1] * n for _ in range(k)]
    for j in range(n - k + 1):
        dp[0][j] = (det_r[0] - ratios[j]) ** 2
    for i in range(1, k):
        for j in range(i, n - (k - 1 - i)):
            pos_cost = (det_r[i] - ratios[j]) ** 2 * 0.35
            best_c, best_p = INF, -1
            for pj in range(i - 1, j):
                if dp[i - 1][pj] >= INF:
                    continue
                exp_gap = ratios[j] - ratios[pj]
                obs_gap = det_r[i] - det_r[i - 1]
                gap_cost = ((obs_gap - exp_gap) / max(0.02, exp_gap)) ** 2 * 0.08
                c = dp[i - 1][pj] + pos_cost + gap_cost
                if c < best_c:
                    best_c, best_p = c, pj
            dp[i][j] = best_c
            parent[i][j] = best_p

    last_j = min(range(k - 1, n), key=lambda j: dp[k - 1][j])
    assigned: Dict[int, int] = {}
    cur_j = last_j
    for i in range(k - 1, -1, -1):
        assigned[cur_j] = detected_xs[i]
        cur_j = parent[i][cur_j]

    assigned.setdefault(0, tx0)
    assigned.setdefault(n - 1, tx1)
    known = sorted(assigned.keys())
    out = [0] * n
    for s in known:
        out[s] = assigned[s]
    for idx in range(len(known) - 1):
        sa, sb = known[idx], known[idx + 1]
        if sb - sa > 1:
            xa, xb, ra, rb = out[sa], out[sb], ratios[sa], ratios[sb]
            for s in range(sa + 1, sb):
                out[s] = int(round(xa + (ratios[s] - ra) / max(1e-6, rb - ra) * (xb - xa)))
    return out


def _detect_table_columns(
    binary_inv: np.ndarray,
    y_top: int,
    y_bot: int,
    tx0: int,
    tx1: int,
    ratios: List[float],
) -> List[int]:
    """Mendeteksi garis pembatas kolom vertikal di separuh atas zona data tabel via Morfologi Opening."""
    h, _ = binary_inv.shape
    y_band_end = min(h, y_top + max(60, int((y_bot - y_top) * 0.48)))
    strip = binary_inv[max(0, y_top) : y_band_end, :]
    sh = strip.shape[0]
    if sh < 20:
        return _snap_columns_with_ratios([], tx0, tx1, ratios)

    b_h = cv2.dilate(strip, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(16, int(sh * 0.35))))
    vert_lines = cv2.morphologyEx(b_h, cv2.MORPH_OPEN, vert_kernel)

    x_proj = np.sum(vert_lines > 0, axis=0)
    xs = [
        x
        for x in _cluster_positions(np.where(x_proj > sh * 0.32)[0], max_gap=16)
        if (tx0 - 20) <= x <= (tx1 + 20)
    ]
    if xs and xs[0] - tx0 > 25:
        xs = [tx0] + xs
    if xs and tx1 - xs[-1] > 25:
        xs = xs + [tx1]

    expected_n = len(ratios)
    if len(xs) == expected_n:
        return xs
    return _snap_columns_with_ratios(xs, tx0, tx1, ratios)


def _track_all_columns_chained(
    binary_inv: np.ndarray,
    global_row_ys: List[int],
    col_xs: List[int],
) -> List[List[int]]:
    """
    Melacak koordinat vertikal garis batas baris pada setiap kolom secara berantai dari tengah
    ke kiri dan kanan (langkah antar-kolom dibatasi ±9px) menggunakan energi kolektif seluruh
    garis baris tabel, sehingga tahan terhadap kertas melengkung/miring tanpa melompat baris
    maupun terganggu teks 16-digit NIK.
    """
    h, w = binary_inv.shape
    ncols = len(col_xs) - 1
    if ncols <= 0 or len(global_row_ys) < 2:
        return [list(global_row_ys) for _ in range(max(0, ncols))]

    b_v = cv2.dilate(binary_inv, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    def _find_best_dy(x1: int, x2: int, center_dy: int, radius: int) -> Tuple[int, Optional[np.ndarray]]:
        x1c, x2c = max(0, x1 + 4), min(w, x2 - 4)
        if x2c - x1c < 10:
            return center_dy, None
        ye = np.mean(b_v[:, x1c:x2c] > 0, axis=1).astype(np.float32)
        best_dy, best_sc = center_dy, -1.0
        for dy in range(center_dy - radius, center_dy + radius + 1):
            sc = 0.0
            for y in global_row_ys:
                yy = y + dy
                if 0 <= yy < h:
                    sc += float(ye[yy])
            sc -= 0.004 * abs(dy - center_dy)
            if sc > best_sc:
                best_sc = sc
                best_dy = dy
        return best_dy, ye

    mid_c = min(max(1, ncols // 2), ncols - 1)
    dys = [0] * ncols
    yes: List[Optional[np.ndarray]] = [None] * ncols
    dys[mid_c], yes[mid_c] = _find_best_dy(col_xs[mid_c], col_xs[mid_c + 1], 0, 12)
    for c in range(mid_c + 1, ncols):
        dys[c], yes[c] = _find_best_dy(col_xs[c], col_xs[c + 1], dys[c - 1], 9)
    for c in range(mid_c - 1, -1, -1):
        dys[c], yes[c] = _find_best_dy(col_xs[c], col_xs[c + 1], dys[c + 1], 9)

    col_row_ys: List[List[int]] = []
    for c in range(ncols):
        ye = yes[c]
        dy = dys[c]
        refined: List[int] = []
        for y in global_row_ys:
            yc = y + dy
            if ye is not None:
                lo, hi = max(0, yc - 3), min(h - 1, yc + 3)
                sub = ye[lo : hi + 1]
                if len(sub) and float(np.max(sub)) >= 0.62:
                    refined.append(lo + int(np.argmax(sub)))
                    continue
            refined.append(max(0, min(h - 1, yc)))
        col_row_ys.append(refined)
    return col_row_ys


def _extract_cell_patch(
    norm_gray: np.ndarray,
    horiz_lines: Optional[np.ndarray],
    y1_a: int,
    y2_a: int,
    x1: int,
    x2: int,
    target_h: int = 40,
) -> np.ndarray:
    """
    Memotong satu sel tabel secara bersih di dalam batas garis kolom yang telah dilacak,
    menghapus sisa garis bingkai hitam tepi, lalu memperbesarnya ke tinggi ideal OCR.
    """
    h, w = norm_gray.shape
    x1c, x2c = max(0, x1), min(w, x2)
    if x2c - x1c < 10:
        return np.full((target_h, 40), 252, dtype=np.uint8)

    ly1, ly2 = y1_a, y2_a
    if horiz_lines is not None:
        col_h = horiz_lines[:, x1c:x2c]
        proj = np.sum(col_h > 0, axis=1)
        exp_h = max(10, y2_a - y1_a)
        win = max(4, min(8, int(exp_h * 0.25)))
        w1 = proj[max(0, y1_a - win) : min(h, y1_a + win + 1)]
        w2 = proj[max(0, y2_a - win) : min(h, y2_a + win + 1)]
        cw_raw = x2c - x1c
        p1_ok = len(w1) > 0 and np.max(w1) >= cw_raw * 0.45
        p2_ok = len(w2) > 0 and np.max(w2) >= cw_raw * 0.45
        c_ly1 = (max(0, y1_a - win) + int(np.argmax(w1))) if p1_ok else y1_a
        c_ly2 = (max(0, y2_a - win) + int(np.argmax(w2))) if p2_ok else y2_a
        if 0.72 * exp_h <= (c_ly2 - c_ly1) <= 1.28 * exp_h:
            ly1, ly2 = c_ly1, c_ly2

    if ly2 - ly1 < 8:
        return np.full((target_h, 40), 252, dtype=np.uint8)

    cell = norm_gray[max(0, ly1 + 2) : min(h, ly2 - 2), max(0, x1c + 2) : min(w, x2c - 2)].copy()
    if cell.size == 0 or cell.shape[0] < 6 or cell.shape[1] < 6:
        return np.full((target_h, 40), 252, dtype=np.uint8)

    # Hapus sisa garis bingkai hitam tepi saja (ambang tinggi >=0.78 di kiri/kanan agar huruf 'I'/'1'/'S' tidak terhapus!)
    for _ in range(3):
        if cell.shape[0] > 10 and np.mean(cell[0, :] < 170) > 0.58:
            cell = cell[1:, :]
        if cell.shape[0] > 10 and np.mean(cell[-1, :] < 170) > 0.58:
            cell = cell[:-1, :]
        if cell.shape[1] > 10 and np.mean(cell[:, 0] < 165) > 0.78:
            cell = cell[:, 1:]
        if cell.shape[1] > 10 and np.mean(cell[:, -1] < 165) > 0.78:
            cell = cell[:, :-1]

    p3, p97 = np.percentile(cell, (3, 97))
    if 15 < (p97 - p3) < 165:
        cell = np.clip((cell.astype(np.float32) - p3) * 242.0 / (p97 - p3) + 8.0, 0, 255).astype(np.uint8)

    ch, cw = cell.shape
    scale = float(target_h) / max(ch, 1)
    up = cv2.resize(cell, (max(20, int(round(cw * scale))), target_h), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 0.9)
    up = cv2.addWeighted(up, 1.28, blur, -0.28, 0)
    return up


def _read_column_strip(
    norm_gray: np.ndarray,
    horiz_lines: Optional[np.ndarray],
    row_ys: List[int],
    x1: int,
    x2: int,
) -> List[str]:
    """
    Menyusun seluruh sel pada satu kolom menjadi satu kanvas vertikal berjarak putih bersih,
    lalu membacanya dalam satu panggilan RapidOCR (use_det=True, use_cls=False).
    """
    nrows = len(row_ys) - 1
    if nrows <= 0:
        return []
    target_h = 40
    patches = [
        _extract_cell_patch(norm_gray, horiz_lines, row_ys[r], row_ys[r + 1], x1, x2, target_h=target_h)
        for r in range(nrows)
    ]
    pad_y, pad_x = 26, 24
    max_w = max(pt.shape[1] for pt in patches)
    slot_h = target_h + pad_y
    canvas = np.full((nrows * slot_h + pad_y, max_w + 2 * pad_x), 252, dtype=np.uint8)
    for r, pt in enumerate(patches):
        y0 = pad_y // 2 + r * slot_h
        canvas[y0 : y0 + target_h, pad_x : pad_x + pt.shape[1]] = pt

    res, _ = _OCR_STRIP(canvas)
    row_tokens: List[List[Tuple[float, str]]] = [[] for _ in range(nrows)]
    if res:
        for box, txt, _score in res:
            pts = np.array(box, dtype=np.float32)
            cy = float(np.mean(pts[:, 1]))
            cx_min = float(np.min(pts[:, 0]))
            r = int((cy - pad_y // 2) // slot_h)
            if 0 <= r < nrows:
                cleaned = str(txt).strip()
                if cleaned:
                    row_tokens[r].append((cx_min, cleaned))

    return [" ".join(t for _, t in sorted(rt)) or "-" for rt in row_tokens]


# ==============================================================================
# TAHAP 2B: CTC LATIN-CONSTRAINED & LEXICON SCORING (FALLBACK SEL KABUR)
# ==============================================================================

_CTC_CHARS: List[str] = list(_OCR_STRIP.text_rec.postprocess_op.character)
_CTC_UPPER_IDXS: List[int] = [
    i for i, c in enumerate(_CTC_CHARS)
    if i == 0 or (len(c) == 1 and c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /.-")
]
_CTC_UPPER_MAP: Dict[str, int] = {_CTC_CHARS[cid]: li for li, cid in enumerate(_CTC_UPPER_IDXS)}


def _get_ctc_probs_single(img_gray: np.ndarray) -> np.ndarray:
    """Menjalankan recognizer ONNX RapidOCR pada satu potongan teks rapat dan mengembalikan matriks probabilitas CTC."""
    rec = _OCR_STRIP.text_rec
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) if img_gray.ndim == 2 else img_gray
    wh_ratio = max(1.0, bgr.shape[1] / float(max(1, bgr.shape[0])))
    norm_img = rec.resize_norm_img(bgr, wh_ratio)[np.newaxis, :]
    return rec.session(norm_img)[0][0]


def _ctc_forward_log_likelihood(log_p: np.ndarray, word: str) -> float:
    """Menghitung log-likelihood CTC forward eksak untuk kandidat kosakata KK standar."""
    seq = [_CTC_UPPER_MAP[ch] for ch in word.upper() if ch in _CTC_UPPER_MAP]
    if not seq or log_p.shape[0] < len(seq):
        return -1e9
    ext = [0]
    for s in seq:
        ext.append(s)
        ext.append(0)
    S, T = len(ext), log_p.shape[0]
    NEG_INF = -1e18
    alpha = np.full(S, NEG_INF, dtype=np.float64)
    alpha[0] = log_p[0, ext[0]]
    if S > 1:
        alpha[1] = log_p[0, ext[1]]
    for t in range(1, T):
        new_a = np.full(S, NEG_INF, dtype=np.float64)
        lp_t = log_p[t]
        for s in range(S):
            v = alpha[s]
            if s > 0:
                ap = alpha[s - 1]
                m = max(v, ap)
                if m > NEG_INF / 2:
                    v = m + np.log(np.exp(v - m) + np.exp(ap - m))
            if s > 1 and ext[s] != 0 and ext[s] != ext[s - 2]:
                ask = alpha[s - 2]
                m = max(v, ask)
                if m > NEG_INF / 2:
                    v = m + np.log(np.exp(v - m) + np.exp(ask - m))
            new_a[s] = v + lp_t[ext[s]]
        alpha = new_a
    a1, a2 = alpha[S - 1], alpha[S - 2]
    m = max(a1, a2)
    return float(m + np.log(np.exp(a1 - m) + np.exp(a2 - m))) if m > NEG_INF / 2 else -1e9


def _tight_crop_cell_ink(
    norm_gray: np.ndarray, y1: int, y2: int, x1: int, x2: int
) -> Tuple[Optional[np.ndarray], float]:
    """
    Memotong area tinta teks secara rapat (tanpa bingkai garis sel & tanpa margin putih kanan)
    serta mengestimasi panjang karakter berdasarkan rasio lebar/tinggi font blanko KK.
    """
    h, w = norm_gray.shape
    row_h = max(10, y2 - y1)
    yt = y1 + max(4, int(row_h * 0.14))
    yb = y2 - max(2, int(row_h * 0.08))
    cell = norm_gray[max(0, yt) : min(h, yb), max(0, x1 + 2) : min(w, x2 - 2)].copy()
    if cell.size == 0 or cell.shape[0] < 8 or cell.shape[1] < 8:
        return None, 0.0

    bg_init = float(np.percentile(cell, 80))
    for _ in range(4):
        if cell.shape[0] > 10 and np.mean(cell[0, :] < bg_init - 16) > 0.45:
            cell = cell[1:, :]
        if cell.shape[0] > 10 and np.mean(cell[-1, :] < bg_init - 16) > 0.45:
            cell = cell[:-1, :]
        if cell.shape[1] > 10 and np.mean(cell[:, 0] < bg_init - 20) > 0.78:
            cell = cell[:, 1:]
        if cell.shape[1] > 10 and np.mean(cell[:, -1] < bg_init - 20) > 0.78:
            cell = cell[:, :-1]

    p2 = float(np.percentile(cell, 2))
    bg = float(np.percentile(cell, 78))
    contrast = bg - p2
    if contrast < 20.0:
        return None, 0.0

    thresh = bg - max(16.0, contrast * 0.38)
    ink = (cell < thresh).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    ch, cw = cell.shape
    mid_ink = ink[max(0, int(ch * 0.14)) : min(ch, int(ch * 0.92)), :]
    xs = np.where(np.sum(mid_ink > 0, axis=0) >= 2)[0]
    if len(xs) < 4:
        return None, 0.0

    x0 = int(xs[0])
    x1_i = x0
    max_gap = max(18, int(ch * 1.2))
    for x in xs[1:]:
        if x - x1_i > max_gap and (x1_i - x0) >= 12:
            break
        x1_i = int(x)
    if x1_i - x0 < 5:
        return None, 0.0

    ys = np.where(np.sum(ink[:, x0 : x1_i + 1] > 0, axis=1) >= 2)[0]
    if len(ys) < 4:
        return None, 0.0
    y0_i, y1_i = int(ys[0]), int(ys[-1])
    text_h = max(7, y1_i - y0_i + 1)
    text_w = max(5, x1_i - x0 + 1)
    # Jika hanya berupa tanda strip "-" kecil (lebar sangat sempit), anggap sel kosong "-"
    if text_w < max(10, int(text_h * 0.9)) and text_h < max(8, int(ch * 0.45)):
        return None, 0.0

    # Rasio rata-rata lebar/tinggi huruf kapital Arial Narrow blanko KK (~0.45)
    est_chars = (text_w / float(text_h)) / 0.45

    tight = cell[max(0, y0_i - 2) : min(ch, y1_i + 3), max(0, x0 - 2) : min(cw, x1_i + 3)].copy()
    tp2, tp96 = np.percentile(tight, (2, 96))
    if tp96 - tp2 > 14:
        tight = np.clip((tight.astype(np.float32) - tp2) * 240.0 / (tp96 - tp2) + 10.0, 0, 255).astype(np.uint8)
    tight = cv2.copyMakeBorder(tight, 3, 3, 5, 5, cv2.BORDER_CONSTANT, value=250)
    return tight, est_chars


def _ctc_score_vocab_cell(
    norm_gray: np.ndarray,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    vocab: List[str],
) -> str:
    """
    Fallback berbasis probabilitas CTC (terbatas karakter Latin kapital) untuk sel dropdown KK
    (Agama, Pendidikan, Jenis Pekerjaan, Status Perkawinan, Hubungan Keluarga) ketika deteksi
    strip OCR terhambat oleh blur/kontras rendah.
    """
    crop, est_chars = _tight_crop_cell_ink(norm_gray, y1, y2, x1, x2)
    if crop is None or est_chars < 1.2:
        return "-"
    try:
        probs_tv = _get_ctc_probs_single(crop)
    except Exception:
        return "-"

    sub = probs_tv[:, _CTC_UPPER_IDXS].copy()
    sub = sub / np.maximum(np.sum(sub, axis=1, keepdims=True), 1e-12)
    sub[:, 0] *= 0.35
    sub = sub / np.maximum(np.sum(sub, axis=1, keepdims=True), 1e-12)
    log_p = np.log(np.maximum(sub, 1e-12))

    best_word = "-"
    best_score = -1e9
    for w in vocab:
        if not w or w == "-":
            continue
        variants = [w]
        if "/" in w:
            variants.append(w.replace("/", ""))
            variants.append(w.replace("/", " "))
        for vw in variants:
            ll = _ctc_forward_log_likelihood(log_p, vw)
            sc = ll + 2.55 * len(vw) - 1.35 * abs(len(w) - est_chars)
            if sc > best_score:
                best_score = sc
                best_word = w
    return best_word if best_score > -42.0 else "-"


# ==============================================================================
# TAHAP 3: FUNGSI PEMBERSIHAN TEKS, KOSAKATA KK, & VALIDASI SILANG NIK
# ==============================================================================

def _clean_digits(text: str) -> str:
    """Mengubah huruf yang mirip angka akibat salah baca OCR menjadi digit angka."""
    mapping = {
        "O": "0", "o": "0", "D": "0", "Q": "0",
        "I": "1", "l": "1", "|": "1", "J": "1", "T": "1",
        "Z": "2", "z": "2",
        "S": "5", "s": "5",
        "G": "6",
        "B": "8",
        "A": "4",
        "g": "9",
    }
    return "".join(ch if ch.isdigit() else mapping.get(ch, "") for ch in text)


def _match_closest_vocab(raw_text: str, vocab_list: List[str], min_ratio: float = 0.42) -> str:
    """Mencocokkan teks hasil bacaan OCR ke kosakata resmi Kartu Keluarga terdekat."""
    cleaned = _strip_non_latin(raw_text, allow_chars=r"A-Z0-9/\s.-")
    if not cleaned or cleaned == "-":
        return "-"
    if cleaned in vocab_list:
        return cleaned

    best_item = cleaned
    best_score = 0.0
    clean_nospace = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not clean_nospace:
        return "-"

    for item in vocab_list:
        item_nospace = re.sub(r"[^A-Z0-9]", "", item)
        score = difflib.SequenceMatcher(None, clean_nospace, item_nospace).ratio()
        if score > best_score:
            best_score = score
            best_item = item

    return best_item if best_score >= min_ratio else cleaned


def _format_tanggal(raw_text: str) -> str:
    """Memformat hasil bacaan tanggal menjadi DD-MM-YYYY yang valid."""
    if not raw_text or raw_text == "-":
        return "-"
    # Bersihkan huruf mirip angka pada segmen tanggal
    m = re.search(r"([0-9OIlSBZ]{1,2})[-/.\s]+([0-9OIlSBZ]{1,2})[-/.\s]+([0-9OIlSBZ]{4})", raw_text)
    if m:
        dd_s = _clean_digits(m.group(1)).zfill(2)
        mm_s = _clean_digits(m.group(2)).zfill(2)
        yyyy_s = _clean_digits(m.group(3))
        if len(dd_s) == 2 and len(mm_s) == 2 and len(yyyy_s) == 4:
            dd, mm, yy = int(dd_s), int(mm_s), int(yyyy_s)
            if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yy <= 2035:
                return f"{dd:02d}-{mm:02d}-{yy:04d}"

    digits = _clean_digits(raw_text)
    if len(digits) == 8:
        dd, mm, yy = int(digits[:2]), int(digits[2:4]), int(digits[4:8])
        if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yy <= 2035:
            return f"{dd:02d}-{mm:02d}-{yy:04d}"
    return "-"


def _decode_nik_birth_info(nik: str) -> Tuple[str, str]:
    """
    Mengekstrak tanggal lahir (DD-MM-YYYY) dan jenis kelamin (LAKI-LAKI/PEREMPUAN)
    sesuai standar resmi NIK Dukcapil Indonesia (digit 7-12 = DDMMYY, perempuan DD+40).
    """
    if len(nik) != 16 or not nik.isdigit():
        return "-", ""
    try:
        raw_dd = int(nik[6:8])
        mm = int(nik[8:10])
        yy = int(nik[10:12])
    except ValueError:
        return "-", ""

    if raw_dd > 40:
        jk = "PEREMPUAN"
        dd = raw_dd - 40
    else:
        jk = "LAKI-LAKI"
        dd = raw_dd

    if not (1 <= dd <= 31 and 1 <= mm <= 12):
        return "-", ""
    yyyy = 2000 + yy if yy <= 26 else 1900 + yy
    return f"{dd:02d}-{mm:02d}-{yyyy:04d}", jk


def _extract_nik_16(raw_text: str, no_kk: str = "") -> str:
    """Membersihkan teks NIK menjadi 16 digit dan memperbaiki salah baca prefiks 1-digit terhadap No. KK."""
    digits = _clean_digits(raw_text)
    if len(digits) > 16:
        # Ambil potongan 16 digit yang paling cocok dengan awalan kode provinsi No. KK
        if len(no_kk) >= 2:
            idx = digits.find(no_kk[:2])
            if idx >= 0 and idx + 16 <= len(digits):
                digits = digits[idx : idx + 16]
        digits = digits[:16]

    if len(digits) == 16 and len(no_kk) == 16:
        kk_pref = no_kk[:6]
        if digits[:6] != kk_pref:
            # Perbaiki hanya jika beda 1 digit yang sering tertukar pada OCR (0/8/9/6, 1/7)
            diffs = sum(1 for a, b in zip(digits[:6], kk_pref) if a != b)
            if diffs == 1:
                digits = kk_pref + digits[6:]
    return digits


def _reconcile_family_consistency(identitas: Dict[str, str], anggota: List[Dict[str, Any]]) -> None:
    """
    Menyelaraskan konsistensi internal KK (berlaku umum untuk seluruh KK Indonesia):
    1. Validasi silang NIK <-> Tanggal Lahir <-> Jenis Kelamin.
    2. Jika prefiks 6 digit No. KK berbeda 1 digit confusable (mis. 0 vs 8) dibanding mayoritas NIK anggota,
       selaraskan ke prefiks konsensus.
    3. Selaraskan nama Kepala Keluarga (baris 1) dan nama Ayah/Ibu pada baris Anak bila terbaca kabur.
    """
    if not anggota:
        return

    # 1. Konsensus prefiks 6-digit kode wilayah antara No. KK & NIK anggota
    valid_niks = [a["nik"] for a in anggota if len(a.get("nik", "")) == 16 and a["nik"].isdigit()]
    if len(identitas.get("no_kk", "")) == 16 and len(valid_niks) >= 2:
        pref_counts: Dict[str, int] = {}
        for nk in valid_niks:
            pref_counts[nk[:6]] = pref_counts.get(nk[:6], 0) + 1
        top_pref, top_cnt = max(pref_counts.items(), key=lambda kv: kv[1])
        kk_pref = identitas["no_kk"][:6]
        if top_cnt >= 2 and top_pref != kk_pref:
            diff_pos = [i for i in range(6) if kk_pref[i] != top_pref[i]]
            if len(diff_pos) == 1:
                identitas["no_kk"] = top_pref + identitas["no_kk"][6:]

    # 2. Validasi silang NIK <-> Tanggal Lahir & Jenis Kelamin pada setiap anggota
    for a in anggota:
        nk = a.get("nik", "")
        nik_tgl, nik_jk = _decode_nik_birth_info(nk)
        cur_tgl = a.get("tanggal_lahir", "-")
        if nik_tgl != "-":
            # Isi dari NIK bila tanggal lahir tidak terbaca, tahun tidak realistis (<1925), atau DD-MM cocok tapi tahun beda 1 digit
            if cur_tgl == "-":
                a["tanggal_lahir"] = nik_tgl
            elif len(cur_tgl) == 10 and (int(cur_tgl[6:10]) < 1925 or (cur_tgl[:5] == nik_tgl[:5] and cur_tgl[6:10] != nik_tgl[6:10])):
                a["tanggal_lahir"] = nik_tgl
            elif len(cur_tgl) == 10 and len(nk) == 16 and nk.isdigit():
                # Sebaliknya, bila tanggal lahir terbaca valid & hanya 1-2 digit tanggal pada NIK meleset karena blur (DD cocok)
                try:
                    c_dd, c_mm, c_yy = int(cur_tgl[:2]), int(cur_tgl[3:5]), int(cur_tgl[8:10])
                    exp_dd = c_dd + 40 if nik_jk == "PEREMPUAN" else c_dd
                    exp_6 = f"{exp_dd:02d}{c_mm:02d}{c_yy:02d}"
                    if nk[6:8] == exp_6[:2] and sum(1 for x, y in zip(nk[6:12], exp_6) if x != y) in (1, 2):
                        a["nik"] = nk[:6] + exp_6 + nk[12:]
                except ValueError:
                    pass
        if nik_jk and len(nk) == 16:
            a["jenis_kelamin"] = nik_jk

    # 3. Selaraskan Nama Kepala Keluarga (header vs baris 1)
    kepala_header = identitas.get("nama_kepala_keluarga", "").strip()
    if kepala_header and anggota:
        nama_1 = anggota[0].get("nama_lengkap", "").strip()
        sim_1 = difflib.SequenceMatcher(None, nama_1.replace(" ", ""), kepala_header.replace(" ", "")).ratio()
        if not nama_1 or sim_1 >= 0.45 or (len(nama_1) < 7 and len(kepala_header) >= 6):
            anggota[0]["nama_lengkap"] = kepala_header
        anggota[0]["status_hubungan"] = "KEPALA KELUARGA"

    # 3B. Konsensus Agama keluarga pada foto blur (khusus pasangan 5 huruf mudah tertukar ISLAM <-> HINDU)
    ag_vals = [a.get("agama", "-") for a in anggota if a.get("agama") in LIST_AGAMA]
    if len(anggota) >= 3 and len(ag_vals) == len(anggota):
        top_ag = max(set(ag_vals), key=ag_vals.count)
        if ag_vals.count(top_ag) >= len(anggota) - 1:
            for a in anggota:
                if a["agama"] != top_ag and {a["agama"], top_ag} == {"ISLAM", "HINDU"}:
                    a["agama"] = top_ag

    # 4. Selaraskan nama Istri (baris 2) & nama Ayah/Ibu serta Status Hubungan Anak terhadap Kepala Keluarga & Istri
    kepala_clean = anggota[0].get("nama_lengkap", "").strip() if anggota else ""
    istri_clean = ""
    if len(anggota) >= 2 and (
        anggota[1].get("status_hubungan") == "ISTRI"
        or anggota[1].get("jenis_kelamin") == "PEREMPUAN"
    ):
        ibu_candidates = [
            a.get("nama_ibu", "").strip()
            for a in anggota[2:]
            if len(a.get("nama_ibu", "").strip()) >= 4
        ]
        nama_2 = anggota[1].get("nama_lengkap", "").strip()
        if ibu_candidates:
            best_ibu = max(ibu_candidates, key=lambda s: (ibu_candidates.count(s), " " in s, len(s)))
            if not nama_2 or difflib.SequenceMatcher(None, nama_2.replace(" ", ""), best_ibu.replace(" ", "")).ratio() >= 0.50:
                if " " in best_ibu and " " not in nama_2:
                    anggota[1]["nama_lengkap"] = best_ibu
        istri_clean = anggota[1].get("nama_lengkap", "").strip()
        # Baris 2 perempuan berstatus KAWIN (bukan BELUM KAWIN) dengan Ayah != Kepala Keluarga adalah ISTRI
        sk2 = anggota[1].get("status_perkawinan", "")
        ayah2 = anggota[1].get("nama_ayah", "").strip()
        ayah2_is_kepala = (
            bool(kepala_clean and ayah2)
            and difflib.SequenceMatcher(None, ayah2.replace(" ", ""), kepala_clean.replace(" ", "")).ratio() >= 0.65
        )
        if (
            anggota[0].get("jenis_kelamin") == "LAKI-LAKI"
            and anggota[1].get("jenis_kelamin") == "PEREMPUAN"
            and "KAWIN" in sk2
            and "BELUM" not in sk2
            and not ayah2_is_kepala
        ):
            anggota[1]["status_hubungan"] = "ISTRI"

    for a in anggota[2:]:
        ayah = a.get("nama_ayah", "").strip()
        ibu = a.get("nama_ibu", "").strip()
        ayah_is_kepala = (
            bool(kepala_clean and ayah)
            and difflib.SequenceMatcher(None, ayah.replace(" ", ""), kepala_clean.replace(" ", "")).ratio() >= 0.55
        )
        if ayah_is_kepala:
            a["nama_ayah"] = kepala_clean
        if istri_clean and ibu and difflib.SequenceMatcher(None, ibu.replace(" ", ""), istri_clean.replace(" ", "")).ratio() >= 0.55:
            a["nama_ibu"] = istri_clean
        # Anggota baris 3+ yang ayahnya adalah Kepala Keluarga dan belum kawin pasti berstatus ANAK
        if ayah_is_kepala and a.get("status_hubungan") in ("-", "", "ISTRI", "SUAMI", "MENANTU", "CUCU"):
            a["status_hubungan"] = "ANAK"
        elif not a.get("status_hubungan") or a["status_hubungan"] == "-":
            a["status_hubungan"] = "ANAK"


# ==============================================================================
# TAHAP 4: EKSTRAKSI HEADER IDENTITAS, FOOTER, & TABEL ANGGOTA KELUARGA
# ==============================================================================

def _run_full_ocr_tokens(gray_img: np.ndarray) -> List[Dict[str, Any]]:
    """Menjalankan RapidOCR pada halaman penuh dan mengembalikan daftar token berkoordinat normalisasi."""
    h, w = gray_img.shape[:2]
    ocr_result, _ = _OCR_ENGINE(gray_img)
    tokens: List[Dict[str, Any]] = []
    if not ocr_result:
        return tokens
    for box, text, score in ocr_result:
        txt = str(text).strip()
        if not txt:
            continue
        pts = np.array(box, dtype=np.float32)
        x_min = float(np.min(pts[:, 0])) / w
        y_min = float(np.min(pts[:, 1])) / h
        x_max = float(np.max(pts[:, 0])) / w
        y_max = float(np.max(pts[:, 1])) / h
        tokens.append(
            {
                "text": txt,
                "score": float(score),
                "nx_min": x_min,
                "nx_max": x_max,
                "nx": (x_min + x_max) / 2.0,
                "ny": (y_min + y_max) / 2.0,
                "h": (y_max - y_min) * h,
                "x_min": x_min * w,
                "x_max": x_max * w,
                "cy": ((y_min + y_max) / 2.0) * h,
            }
        )
    return tokens


def _group_side_lines(side_tokens: List[Dict[str, Any]], ny_tol: float = 0.011) -> List[List[Dict[str, Any]]]:
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


def _clean_rt_rw(raw: str) -> str:
    txt = raw.strip()
    if not txt:
        return ""
    if txt in ("+", "十", "-", "-/", "/-", "-/-", "—"):
        return "-/-"
    m = re.search(r"(\d{1,3})\s*[/\-]\s*(\d{1,3})", txt)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    digits = _clean_digits(txt)
    # Pola umum OCR membaca "/" sebagai "1": "002/001" -> "0021001"
    if len(digits) == 7 and digits[3] == "1":
        return f"{digits[:3]}/{digits[4:]}"
    if len(digits) == 6:
        return f"{digits[:3]}/{digits[3:]}"
    if len(digits) == 5 and digits[0] in ("0", "6", "9"):
        # Hindari salah baca 002/001 menjadi 5 digit tanpa slash bila bukan kode pos
        return f"{digits[:2]}/{digits[2:]}" if "/" in txt else txt
    if len(digits) == 3 and ("/" in txt or "-" in txt):
        return f"{digits}/-"
    return txt


def _clean_identitas_values(identitas: Dict[str, str]) -> None:
    """Membersihkan & menormalisasi seluruh field identitas KK (termasuk provinsi resmi & konsistensi alamat/desa)."""
    # Jika kode pos 5 digit terlanjur masuk ke alamat atau rt_rw, pindahkan ke kode_pos
    if re.fullmatch(r"\d{5}", identitas["alamat"].strip()):
        if not identitas["kode_pos"]:
            identitas["kode_pos"] = identitas["alamat"].strip()
        identitas["alamat"] = ""
    if re.fullmatch(r"\d{5}", identitas["rt_rw"].strip()):
        if not identitas["kode_pos"]:
            identitas["kode_pos"] = identitas["rt_rw"].strip()
        identitas["rt_rw"] = "-/-"
    # Jika alamat ("DESA ..." / "DUSUN ..." / "JL. ...") terlanjur masuk ke rt_rw, pindahkan ke alamat
    if re.search(r"\b(DESA|DUSUN|JALAN|JL\.|KELURAHAN|PERUM|KP\.|GG\.)\b", identitas["rt_rw"], re.IGNORECASE):
        if not identitas["alamat"] or len(identitas["alamat"]) < 4:
            identitas["alamat"] = identitas["rt_rw"]
        identitas["rt_rw"] = ""
    # Jika alamat terlanjur masuk ke nama_kepala_keluarga, pindahkan ke alamat
    if re.search(r"^(DESA|DUSUN|JALAN|JL\.|KELURAHAN)\b", identitas["nama_kepala_keluarga"], re.IGNORECASE):
        if not identitas["alamat"] or len(identitas["alamat"]) < 4:
            identitas["alamat"] = identitas["nama_kepala_keluarga"]
        identitas["nama_kepala_keluarga"] = ""

    identitas["nama_kepala_keluarga"] = _normalize_spaced_text(
        _strip_non_latin(identitas["nama_kepala_keluarga"], r"A-Z\s.,'"),
        COMMON_NAME_TOKENS,
    )
    alamat_c = _strip_non_latin(identitas["alamat"], r"A-Z0-9\s.,/'-")
    alamat_c = re.sub(r"^JL\.(?=[A-Z])", "JL. ", re.sub(r"^JL(?=[A-Z])", "JL. ", alamat_c))
    alamat_c = re.sub(r"\bNO\.(?=\d)", "NO. ", alamat_c)
    identitas["alamat"] = _normalize_spaced_text(alamat_c, COMMON_LOCATION_TOKENS)
    identitas["rt_rw"] = _clean_rt_rw(identitas["rt_rw"])

    kp_digits = _clean_digits(identitas["kode_pos"])
    if len(kp_digits) >= 5:
        identitas["kode_pos"] = kp_digits[:5]
    elif kp_digits == "0":
        identitas["kode_pos"] = "0"
    else:
        identitas["kode_pos"] = ""

    for loc_key in ("desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi"):
        cleaned_loc = _strip_non_latin(identitas[loc_key], r"A-Z0-9\s.,/'-")
        identitas[loc_key] = _normalize_spaced_text(cleaned_loc, COMMON_LOCATION_TOKENS)

    # Normalisasi Provinsi terhadap 38 Provinsi Resmi Indonesia
    if identitas["provinsi"]:
        prov_match = _match_closest_vocab(identitas["provinsi"], LIST_PROVINSI, min_ratio=0.52)
        if prov_match in LIST_PROVINSI:
            identitas["provinsi"] = prov_match

    # Konsistensi Alamat ("DESA <NAMA> ..." / "KELURAHAN <NAMA> ...") <-> Desa/Kelurahan
    m_desa = re.search(r"\b(?:DESA|KELURAHAN)\s+([A-Z]{4,24})\b", identitas["alamat"])
    if m_desa:
        desa_from_alamat = m_desa.group(1)
        cur_desa = identitas["desa_kelurahan"]
        if not cur_desa or len(cur_desa) < 4:
            identitas["desa_kelurahan"] = desa_from_alamat
        else:
            sim = difflib.SequenceMatcher(None, cur_desa.replace(" ", ""), desa_from_alamat).ratio()
            # Bila teks Desa/Kelurahan kabur tetapi beririsan kuat atau sangat terdistorsi, selaraskan ke ejaan Alamat yang lebih jelas
            if sim >= 0.42 and cur_desa != desa_from_alamat:
                identitas["desa_kelurahan"] = desa_from_alamat


def _extract_identitas_header(
    tokens: List[Dict[str, Any]],
    t1_top_ny: Optional[float] = None,
    t2_bot_ny: Optional[float] = None,
) -> Dict[str, str]:
    """Mengekstrak data identitas KK secara dinamis berdasarkan jangkar judul & posisi garis Tabel 1."""
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

    # 1. Tentukan jangkar vertikal judul "KARTU KELUARGA" & batas atas Tabel 1
    title_ny = 0.05
    nokk_ny = 0.10
    header_bottom_ny = t1_top_ny - 0.008 if t1_top_ny is not None else 0.22

    for tok in sorted(tokens, key=lambda t: t["ny"]):
        up = tok["text"].upper()
        if "KARTU" in up and "KELUARGA" in up and tok["ny"] < 0.25:
            title_ny = tok["ny"]
        # Selalu potong header_bottom_ny di atas judul kolom Tabel 1 (Nama Lengkap / NIK / Tempat Lahir)
        if (
            "NAMA LENGKAP" in up
            or "LAMA LENGKAP" in up
            or "TEMPAT LAHIR" in up
            or "TEMPAL LAH" in up
            or up.strip() == "NIK"
        ) and 0.16 <= tok["ny"] <= 0.40:
            header_bottom_ny = min(header_bottom_ny, tok["ny"] - 0.012)

    # 2. Ekstrak Nomor KK (16 digit di bawah judul KARTU KELUARGA)
    top_tokens = [t for t in sorted(tokens, key=lambda t: t["ny"]) if t["ny"] <= header_bottom_ny]
    for tok in top_tokens:
        m = re.search(r"(?:No\.?\s*|No[·:\-]\s*)?([0-9OIlSBZ\s]{15,20})", tok["text"], re.IGNORECASE)
        if m:
            cand = _clean_digits(m.group(1))
            if len(cand) == 16:
                identitas["no_kk"] = cand
                nokk_ny = tok["ny"]
                break

    # 3. Ambil token header identitas (strictly di antara No. KK dan batas atas Tabel 1)
    hdr_y_min = max(title_ny + 0.02, nokk_ny + 0.004)
    header_tokens = [
        t for t in tokens
        if hdr_y_min <= t["ny"] <= header_bottom_ny
        and "KARTU KELUARGA" not in t["text"].upper()
        and "KARTUKELUARGA" not in t["text"].upper()
        and "REPUBLIK" not in t["text"].upper()
        and "INDONESIA" not in t["text"].upper()
        and "INOONESIA" not in t["text"].upper()
        and "DOMESIA" not in t["text"].upper()
        and "DONE3LA" not in t["text"].upper()
        and "DONE2LA" not in t["text"].upper()
        and "DONESLA" not in t["text"].upper()
        and not re.fullmatch(r"NO\.?\s*\d{0,3}", t["text"].strip(), re.IGNORECASE)
        and not (identitas["no_kk"] and sum(c.isdigit() for c in t["text"]) >= 10 and identitas["no_kk"][:10] in _clean_digits(t["text"]))
        and sum(c.isdigit() for c in t["text"]) < 12
        and t["text"].strip().upper() not in ("NO.", "NO", "NIK", "NK", "JEN", "JENIS")
    ]

    # Tentukan batas pemisah Kiri vs Kanan secara dinamis dari posisi label kolom kanan
    right_label_xs = [
        t["nx_min"]
        for t in header_tokens
        if re.search(r"(Desa|Kelur|Onoal|Deoal|Dnoal|Ke[cp]s?am|[KYV]a[cr]an|Kab[uo]p|Yatop|Pr?o[vr]?in|Piovn)", t["text"], re.IGNORECASE)
        and t["nx_min"] > 0.42
    ]
    split_nx = (min(right_label_xs) - 0.02) if right_label_xs else 0.54

    left_tokens = [t for t in header_tokens if t["nx_min"] < split_nx and t["nx_min"] >= 0.01]
    right_tokens = [t for t in header_tokens if t["nx_min"] >= split_nx]

    left_label_map = [
        ("nama_kepala_keluarga", r"Nama\s*K[ae]p[aeild]{1,3}[ao]?\s*K[ae]?[lu][aeuwo]{1,3}r?ga"),
        ("alamat", r"(?:\bA[il]?a?m[ae]t\b|\bMamat\b|\bMaat\b|Aiema|ALsrTuat)"),
        ("rt_rw", r"RT\s*[/|I]?\s*RW|RTAW|RTIRW|RTRW"),
        ("kode_pos", r"Ko[do]?e\s*P[oe]s"),
    ]
    right_label_map = [
        ("desa_kelurahan", r"(?:Desa\s*[/|]?\s*Ke[lu][^:\s]*|Kelurahan|Deoal\w*|Dnoal\w*|Onoal\w*)"),
        ("kecamatan", r"(?:Ke[cp]s?am[ae]?t?an|Kepsm[a-z]+|[KYV]a[cr]a[mn][a-z]+)"),
        ("kabupaten_kota", r"(?:Kab[uo]p[ae][tl][eo]?n\s*[/|]?\s*Kota|Kabupaten|Katoa\w*|Kata\w*|Yatop\w*)"),
        ("provinsi", r"(?:Pr?o[vr]?[il]?n[gs]?[il]?|Propinsi|Piov[a-z]+|Proin[a-z]*)"),
    ]

    def _parse_side_tilt_safe(
        side_toks: List[Dict[str, Any]],
        label_map: List[Tuple[str, str]],
        ordered_keys: List[str],
        default_val_x: float,
    ) -> None:
        if not side_toks:
            return
        # Tentukan batas x kolom nilai di sebelah kanan blok label
        lbl_xmaxs = [
            t["nx_max"]
            for t in side_toks
            if any(re.search(pat, t["text"], re.IGNORECASE) for _, pat in label_map)
            and not re.search(r":\s*[A-Z0-9]{2,}", t["text"], re.IGNORECASE)
        ]
        val_x = (max(lbl_xmaxs) + 0.005) if lbl_xmaxs else default_val_x
        # Cek apakah ada token yang menggabungkan label & nilai dalam 1 box (mis. "Alamat : JL. MERDEKA")
        lines = _group_side_lines(side_toks, ny_tol=0.009)

        # Jika label & nilai terpisah kolom dan mungkin sedikit miring vertikal, ekstrak 4 baris nilai terpisah!
        pure_val_toks = [
            t for t in side_toks
            if t["nx_min"] >= val_x
            and not any(re.search(pat, t["text"], re.IGNORECASE) for _, pat in label_map)
        ]
        val_lines = _group_side_lines(pure_val_toks, ny_tol=0.009)
        if len(val_lines) == 4:
            for k, vl in zip(ordered_keys, val_lines):
                v = re.sub(r"^[\s:\uff1a.;\-]+", "", " ".join(t["text"] for t in vl)).strip().upper()
                if v and not identitas[k]:
                    identitas[k] = v
            return

        fallback_vals: List[str] = []
        for line in lines:
            joined = " ".join(t["text"] for t in line)
            matched_key = ""
            extracted = ""
            for key, pat in label_map:
                m = re.search(pat, joined, re.IGNORECASE)
                if m:
                    matched_key = key
                    after = joined[m.end() :]
                    extracted = re.sub(r"^[\s:\uff1a.;\-]+", "", after).strip()
                    if not extracted:
                        val_toks = [t["text"] for t in line if t["nx_min"] >= val_x]
                        extracted = re.sub(r"^[\s:\uff1a.;\-]+", "", " ".join(val_toks)).strip()
                    break

            if matched_key and extracted and not identitas[matched_key]:
                identitas[matched_key] = extracted.upper()
            elif not matched_key:
                val_toks = [t["text"] for t in line if t["nx_min"] >= val_x]
                if val_toks:
                    v = re.sub(r"^[\s:\uff1a.;\-]+", "", " ".join(val_toks)).strip().upper()
                    if v:
                        fallback_vals.append(v)

        if ordered_keys[0] == "nama_kepala_keluarga":
            for val in list(fallback_vals):
                if sum(c.isdigit() for c in val) >= 3 and re.fullmatch(r"\d{5}", _clean_digits(val)) and not identitas["kode_pos"]:
                    identitas["kode_pos"] = _clean_digits(val)
                    fallback_vals.remove(val)
        for idx, k in enumerate(ordered_keys):
            if not identitas[k] and idx < len(fallback_vals):
                identitas[k] = fallback_vals[idx]

    _parse_side_tilt_safe(
        left_tokens, left_label_map, ["nama_kepala_keluarga", "alamat", "rt_rw", "kode_pos"], max(0.13, split_nx * 0.30)
    )
    _parse_side_tilt_safe(
        right_tokens, right_label_map, ["desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi"], split_nx + 0.07
    )

    _clean_identitas_values(identitas)

    # 4. Ekstrak Tanggal Dikeluarkan & Kabupaten/Kota fallback di bagian footer (di bawah Tabel 2)
    footer_min_ny = (t2_bot_ny - 0.01) if t2_bot_ny is not None else 0.68
    for tok in sorted(tokens, key=lambda t: t["ny"]):
        if tok["ny"] < footer_min_ny:
            continue
        if len(identitas["kabupaten_kota"]) < 4:
            m_kab = re.search(r"(KABUPATEN|KOTA)\s*([A-Z][A-Z\s]{2,28})$", _strip_non_latin(tok["text"], r"A-Z\s"))
            if m_kab:
                identitas["kabupaten_kota"] = _normalize_spaced_text(
                    f"{m_kab.group(1)} {m_kab.group(2)}".strip(),
                    COMMON_LOCATION_TOKENS,
                )
        if sum(c.isdigit() for c in tok["text"]) >= 12:
            continue
        if not identitas["tanggal_dikeluarkan"]:
            tgl = _format_tanggal(tok["text"])
            if tgl != "-":
                identitas["tanggal_dikeluarkan"] = tgl

    return identitas


def _enhance_identitas_from_zoomed_header(
    norm_gray: np.ndarray,
    identitas: Dict[str, str],
    t1_y: Optional[List[int]],
    tx0: int,
    tx1: int,
    tokens: List[Dict[str, Any]],
) -> None:
    """
    Melakukan pembacaan terfokus (zoom 1.6x) pada blok Header Kiri dan Header Kanan
    apabila salah satu field identitas utama kosong atau terlalu pendek/kabur pada OCR halaman penuh.
    """
    left_keys = ["nama_kepala_keluarga", "alamat", "rt_rw", "kode_pos"]
    right_keys = ["desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi"]

    need_left = any(
        (not identitas[k]) or (k in ("nama_kepala_keluarga", "alamat") and len(identitas[k]) < 4)
        for k in left_keys
    )
    need_right = any(
        (not identitas[k])
        or (k in ("desa_kelurahan", "kecamatan", "kabupaten_kota") and len(identitas[k]) < 3)
        or (k == "provinsi" and identitas[k] not in LIST_PROVINSI)
        for k in right_keys
    )
    if not (need_left or need_right) or not t1_y or len(t1_y) < 3:
        return

    h, w = norm_gray.shape
    row_h = float(np.median(np.diff(t1_y)))
    # Batas bawah: tepat di atas garis atas Tabel 1 (~2.10*row_h di atas baris 1) agar Provinsi tidak terpotong!
    # Batas atas: sedikit di atas pusat No. KK (~6.9*row_h di atas baris 1) agar Nama Kepala & Desa tidak terpotong!
    nokk_ys = [
        t["cy"] - t["h"] * 0.15
        for t in tokens
        if t["ny"] < 0.22 and identitas.get("no_kk") and identitas["no_kk"][:8] in _clean_digits(t["text"])
    ]
    y1 = max(0, int(t1_y[0] - row_h * 2.08))
    y0 = int(min(nokk_ys)) if nokk_ys else max(0, int(t1_y[0] - row_h * 7.1))
    y0 = min(y0, max(0, y1 - int(row_h * 4.2)))
    if y1 - y0 < 30:
        return

    mid_x = tx0 + int((tx1 - tx0) * 0.56)

    def _read_zoom_side(x_start: int, x_end: int, keys: List[str], is_right: bool) -> None:
        crop = norm_gray[y0:y1, max(0, x_start) : min(w, x_end)]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 80:
            return
        up = cv2.resize(crop, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        up = cv2.addWeighted(up, 1.35, cv2.GaussianBlur(up, (0, 0), 1.0), -0.35, 0)
        res, _ = _OCR_STRIP(up)
        if not res:
            return
        uh, uw = up.shape[:2]
        skip_words = (
            "KARTU", "KELUARGA", "REPUBLIK", "INDONESIA", "DONESLA", "DOMESIA", "DONE3LA",
            "NAMA LENGKAP", "LAMA LENGKAP", "JENIS", "KELAMIN", "TEMPAT LAHIR", "TANGGAL",
            "AGAMA", "PENDIDIKAN", "PEKERJAAN", "GOLONGAN", "DARAH",
        )
        z_toks: List[Dict[str, Any]] = []
        for box, txt, sc in res:
            pts = np.array(box, dtype=np.float32)
            t_str = str(txt).strip()
            t_up = t_str.upper()
            if (
                not t_str
                or t_up in ("NO.", "NO", "NIK", "NK")
                or sum(c.isdigit() for c in t_str) >= 10
                or any(bad in t_up for bad in skip_words)
            ):
                continue
            z_toks.append(
                {
                    "text": t_str,
                    "score": float(sc),
                    "nx_min": float(np.min(pts[:, 0])) / uw,
                    "nx_max": float(np.max(pts[:, 0])) / uw,
                    "ny": float(np.mean(pts[:, 1])) / uh,
                }
            )
        # Buang token sisa baris No. KK di bagian atas box zoom (ny < 0.16)
        z_toks = [t for t in z_toks if t["ny"] >= 0.15 and not re.fullmatch(r"NO\.?\s*\d{0,3}", t["text"], re.IGNORECASE)]
        if not z_toks:
            return

        # Pisahkan token label (kiri) vs token nilai (kanan) secara otomatis dari posisi label
        label_pat = (
            r"(Desa|Kelur|Deoal|Dnoal|Dtsal|Onoal|Ke[cp]s?am|[KYV]a[cr]a[mn]|Kab[uo]p|Kata|Katoa|Pr?o[vr]?in)"
            if is_right
            else r"(Nama\s*K|Alamat|Aamat|Mamat|Maat|RT\s*[/|I]?\s*RW|RTIRW|RTRW|Kode\s*Pos)"
        )
        lbl_xs = [
            t["nx_max"]
            for t in z_toks
            if re.search(label_pat, t["text"], re.IGNORECASE)
            and t["nx_min"] < 0.35
            and not re.search(r":\s*[A-Z]{2,}", t["text"], re.IGNORECASE)
        ]
        val_split = (max(lbl_xs) + 0.01) if lbl_xs else (0.30 if is_right else 0.38)

        # Juga dukung token gabungan "Label : NILAI" (mis. "Desa/Kelurahan:KORI")
        val_toks = [t for t in z_toks if t["nx_min"] >= val_split and not re.search(label_pat, t["text"], re.IGNORECASE)]
        val_lines = _group_side_lines(val_toks, ny_tol=0.09)

        # Petakan setiap baris nilai ke 4 slot vertikal [0..3] berdasarkan rentang ny aktual label
        slot_vals: Dict[int, str] = {}
        if len(val_lines) == 4:
            for idx_s, vl in enumerate(val_lines):
                slot_vals[idx_s] = re.sub(r"^[\s:\uff1a.;\-]+", "", " ".join(t["text"] for t in vl)).strip().upper()
        else:
            y_min_z = min(t["ny"] for t in z_toks)
            y_max_z = max(t["ny"] for t in z_toks)
            y_span_z = max(0.25, y_max_z - y_min_z)
            for vl in val_lines:
                avg_ny = float(np.mean([t["ny"] for t in vl]))
                rel = (avg_ny - y_min_z) / y_span_z
                slot = min(3, max(0, int(round(rel * 3.0))))
                txt_v = re.sub(r"^[\s:\uff1a.;\-]+", "", " ".join(t["text"] for t in vl)).strip().upper()
                if txt_v and slot not in slot_vals:
                    slot_vals[slot] = txt_v

        for idx_s, k in enumerate(keys):
            cand = slot_vals.get(idx_s, "")
            if not cand:
                continue
            cand_clean = _strip_non_latin(cand, r"A-Z0-9\s.,/'-")
            if not cand_clean:
                continue
            cur = identitas[k]
            if k == "provinsi":
                pm = _match_closest_vocab(cand_clean, LIST_PROVINSI, min_ratio=0.50)
                if pm in LIST_PROVINSI:
                    identitas[k] = pm
                elif not cur:
                    identitas[k] = cand_clean
            elif k == "kode_pos":
                kp = _clean_digits(cand_clean)
                if len(kp) >= 5 and (not cur or len(cur) < 5):
                    identitas[k] = kp[:5]
            elif k == "rt_rw":
                rr = _clean_rt_rw(cand_clean)
                if "/" in rr and (not cur or "/" not in cur):
                    identitas[k] = rr
            elif not cur or len(cur) < 3 or (len(cand_clean) >= 4 and len(cur) < 4):
                identitas[k] = cand_clean

    if need_left:
        _read_zoom_side(tx0, mid_x + 120, left_keys, is_right=False)
    if need_right:
        _read_zoom_side(mid_x - 60, tx1 + 20, right_keys, is_right=True)

    _clean_identitas_values(identitas)


def _lookup_full_token_in_box(
    tokens: List[Dict[str, Any]],
    x1: int,
    x2: int,
    y1: int,
    y2: int,
) -> str:
    """Mengambil teks dari token full-page OCR yang pusatnya berada tepat di dalam sel [x1..x2, y1..y2]."""
    matches = [
        (t["x_min"], t["text"])
        for t in tokens
        if x1 <= ((t["x_min"] + t["x_max"]) / 2.0) <= x2
        and (y1 + 2) <= t["cy"] <= (y2 - 2)
    ]
    if not matches:
        return "-"
    matches.sort(key=lambda m: m[0])
    return " ".join(txt for _, txt in matches).strip() or "-"


def _extract_anggota_via_column_strips(
    norm_gray: np.ndarray,
    horiz_lines: np.ndarray,
    binary_inv: np.ndarray,
    t1_y: List[int],
    t2_y: List[int],
    tx0: int,
    tx1: int,
    identitas: Dict[str, str],
    full_tokens: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Mengekstrak seluruh anggota keluarga menggunakan Column-Strip OCR berbasis grid baris
    & kolom yang dilacak per-kolom secara berantai, dilengkapi fallback CTC-Lexicon
    untuk sel dropdown standar (Agama, Pendidikan, Jenis Pekerjaan, Status) saat kabur.
    """
    t1_x = _detect_table_columns(binary_inv, t1_y[0], t1_y[-1], tx0, tx1, _T1_COL_RATIOS)
    t2_x = _detect_table_columns(binary_inv, t2_y[0], t2_y[-1], tx0, tx1, _T2_COL_RATIOS)
    if len(t1_x) < 10 or len(t2_x) < 9:
        return []

    max_rows = min(len(t1_y) - 1, len(t2_y) - 1)
    if max_rows <= 0:
        return []

    # Lacak garis batas baris halus pada tiap kolom secara berantai (tahan kertas melengkung/miring)
    t1_col_ys = _track_all_columns_chained(binary_inv, t1_y[: max_rows + 1], t1_x)
    t2_col_ys = _track_all_columns_chained(binary_inv, t2_y[: max_rows + 1], t2_x)

    # Baca kolom Nama Lengkap (col 1) & NIK (col 2) lebih dulu untuk mengetahui baris aktif
    col_nama = _read_column_strip(norm_gray, None, t1_col_ys[1], t1_x[1], t1_x[2])
    col_nik = _read_column_strip(norm_gray, None, t1_col_ys[2], t1_x[2], t1_x[3])

    # Deteksi & lewati sub-header nomor kolom "(1) (2) (3)" bila terikut sebagai baris 0
    start_r = 0
    if max_rows >= 2:
        r0_nik_d = _clean_digits(col_nik[0])
        r1_nik_d = _clean_digits(col_nik[1])
        r0_nm_l = re.sub(r"[^A-Za-z]", "", col_nama[0])
        if len(r0_nik_d) <= 2 and len(r0_nm_l) <= 2 and len(r1_nik_d) >= 12:
            start_r = 1

    active_end = start_r
    for r in range(start_r, max_rows):
        nm_letters = re.sub(r"[^A-Za-z]", "", col_nama[r])
        nk_digits = _clean_digits(col_nik[r])
        if len(nm_letters) >= 3 or len(nk_digits) >= 10:
            active_end = r + 1
        elif active_end > start_r:
            if r + 1 < max_rows:
                next_nm = re.sub(r"[^A-Za-z]", "", col_nama[r + 1])
                next_nk = _clean_digits(col_nik[r + 1])
                if len(next_nm) >= 3 or len(next_nk) >= 10:
                    active_end = r + 2
                    continue
            break

    active_rows = active_end - start_r
    if active_rows <= 0:
        return []

    def _slice_t1(col_idx: int) -> List[int]:
        return t1_col_ys[col_idx][start_r : active_end + 1]

    def _slice_t2(col_idx: int) -> List[int]:
        return t2_col_ys[col_idx][: active_rows + 1]

    col_nama_act = col_nama[start_r:active_end]
    col_nik_act = col_nik[start_r:active_end]

    # Baca sisa kolom Tabel 1 (3..9) & seluruh kolom Tabel 2 (1..8) hanya untuk baris aktif
    col_jk = _read_column_strip(norm_gray, None, _slice_t1(3), t1_x[3], t1_x[4])
    col_tempat = _read_column_strip(norm_gray, None, _slice_t1(4), t1_x[4], t1_x[5])
    col_tgl_lahir = _read_column_strip(norm_gray, None, _slice_t1(5), t1_x[5], t1_x[6])
    col_agama = _read_column_strip(norm_gray, None, _slice_t1(6), t1_x[6], t1_x[7])
    col_pendidikan = _read_column_strip(norm_gray, None, _slice_t1(7), t1_x[7], t1_x[8])
    col_pekerjaan = _read_column_strip(norm_gray, None, _slice_t1(8), t1_x[8], t1_x[9])
    col_goldar = (
        _read_column_strip(norm_gray, None, _slice_t1(9), t1_x[9], t1_x[10])
        if len(t1_x) >= 11
        else ["-"] * active_rows
    )

    col_stat_kawin = _read_column_strip(norm_gray, None, _slice_t2(1), t2_x[1], t2_x[2])
    col_tgl_kawin = _read_column_strip(norm_gray, None, _slice_t2(2), t2_x[2], t2_x[3])
    col_stat_hub = _read_column_strip(norm_gray, None, _slice_t2(3), t2_x[3], t2_x[4])
    col_wni = _read_column_strip(norm_gray, None, _slice_t2(4), t2_x[4], t2_x[5])
    col_paspor = _read_column_strip(norm_gray, None, _slice_t2(5), t2_x[5], t2_x[6])
    col_kitap = _read_column_strip(norm_gray, None, _slice_t2(6), t2_x[6], t2_x[7])
    col_ayah = _read_column_strip(norm_gray, None, _slice_t2(7), t2_x[7], t2_x[8])
    col_ibu = (
        _read_column_strip(norm_gray, None, _slice_t2(8), t2_x[8], t2_x[9])
        if len(t2_x) >= 10
        else ["-"] * active_rows
    )

    def _match_with_ctc_fallback(
        raw_val: str,
        vocab: List[str],
        min_ratio: float,
        col_ys_slice: List[int],
        row_i: int,
        cx1: int,
        cx2: int,
        accept_ratio: float = 0.56,
    ) -> str:
        cleaned = _strip_non_latin(raw_val, allow_chars=r"A-Z0-9/\s.-")
        clean_ns = re.sub(r"[^A-Z0-9]", "", cleaned)
        best_item, best_score = "-", 0.0
        if clean_ns:
            for item in vocab:
                sc = difflib.SequenceMatcher(None, clean_ns, re.sub(r"[^A-Z0-9]", "", item)).ratio()
                if sc > best_score:
                    best_score = sc
                    best_item = item
        # Jika strip sudah menangkap teks panjang (>=8 huruf) dengan kemiripan >= 0.48 pada kosakata panjang (mis. Pendidikan/Pekerjaan), prioritaskan!
        eff_accept = 0.48 if (len(clean_ns) >= 8 and len(re.sub(r"[^A-Z0-9]", "", best_item)) >= 11) else accept_ratio
        if best_score >= eff_accept:
            return best_item
        # Fallback ke CTC-Lexicon bila bacaan strip kosong atau lemah pada foto kabur
        ctc_pick = _ctc_score_vocab_cell(norm_gray, col_ys_slice[row_i], col_ys_slice[row_i + 1], cx1, cx2, vocab)
        if ctc_pick != "-":
            return ctc_pick
        if best_score >= min_ratio:
            return best_item
        return "-"

    city_vocab = list(COMMON_LOCATION_TOKENS)
    for k in ("kabupaten_kota", "desa_kelurahan", "kecamatan", "alamat"):
        v = identitas.get(k, "")
        if v:
            clean_c = re.sub(r"^(KOTA|KABUPATEN|DESA|DUSUN)\s+", "", v).strip()
            if clean_c and clean_c not in city_vocab:
                city_vocab.insert(0, clean_c)

    no_kk = identitas.get("no_kk", "")
    anggota_list: List[Dict[str, Any]] = []

    t1_c1 = _slice_t1(1)
    t1_c2 = _slice_t1(2)
    t1_c4 = _slice_t1(4)
    t1_c5 = _slice_t1(5)
    t1_c6 = _slice_t1(6)
    t1_c7 = _slice_t1(7)
    t1_c8 = _slice_t1(8)
    t2_c1 = _slice_t2(1)
    t2_c3 = _slice_t2(3)

    for r in range(active_rows):
        raw_nama_cell = col_nama_act[r]
        raw_nik_cell = col_nik_act[r]
        raw_tempat_cell = col_tempat[r]
        raw_tgl_cell = col_tgl_lahir[r]
        if full_tokens:
            if len(_clean_digits(raw_nik_cell)) < 16:
                fb_nik = _lookup_full_token_in_box(full_tokens, t1_x[2], t1_x[3], t1_c2[r], t1_c2[r + 1])
                if len(_clean_digits(fb_nik)) > len(_clean_digits(raw_nik_cell)):
                    raw_nik_cell = fb_nik
            if len(re.sub(r"[^A-Za-z]", "", raw_nama_cell)) < 4:
                fb_nama = _lookup_full_token_in_box(full_tokens, t1_x[1], t1_x[2], t1_c1[r], t1_c1[r + 1])
                if len(re.sub(r"[^A-Za-z]", "", fb_nama)) > len(re.sub(r"[^A-Za-z]", "", raw_nama_cell)):
                    raw_nama_cell = fb_nama
            if len(re.sub(r"[^A-Za-z]", "", raw_tempat_cell)) < 3:
                fb_tmp = _lookup_full_token_in_box(full_tokens, t1_x[4], t1_x[5], t1_c4[r], t1_c4[r + 1])
                if len(re.sub(r"[^A-Za-z]", "", fb_tmp)) > len(re.sub(r"[^A-Za-z]", "", raw_tempat_cell)):
                    raw_tempat_cell = fb_tmp
            if _format_tanggal(raw_tgl_cell) == "-":
                fb_tgl = _lookup_full_token_in_box(full_tokens, t1_x[5], t1_x[6], t1_c5[r], t1_c5[r + 1])
                if _format_tanggal(fb_tgl) != "-":
                    raw_tgl_cell = fb_tgl

        nama_raw = _strip_non_latin(raw_nama_cell, r"A-Z\s.,'")
        nama_norm = _normalize_spaced_text(nama_raw, COMMON_NAME_TOKENS)
        nik_norm = _extract_nik_16(raw_nik_cell, no_kk=no_kk)

        jk_up = _strip_non_latin(col_jk[r], r"A-Z\-")
        if "P" in jk_up and "L" not in jk_up[:3]:
            jk_norm = "PEREMPUAN"
        elif "L" in jk_up:
            jk_norm = "LAKI-LAKI"
        else:
            jk_norm = _match_closest_vocab(jk_up, ["LAKI-LAKI", "PEREMPUAN"], min_ratio=0.35)
            if jk_norm not in ("LAKI-LAKI", "PEREMPUAN"):
                jk_norm = ""

        tempat_raw = _strip_non_latin(raw_tempat_cell, r"A-Z\s.,'-")
        tempat_norm = _match_closest_vocab(tempat_raw, city_vocab, min_ratio=0.68)
        if tempat_norm == tempat_raw:
            tempat_norm = _normalize_spaced_text(tempat_raw, COMMON_LOCATION_TOKENS)

        tgl_lahir_norm = _format_tanggal(raw_tgl_cell)
        agama_norm = _match_with_ctc_fallback(
            col_agama[r], LIST_AGAMA, 0.50, t1_c6, r, t1_x[6], t1_x[7], accept_ratio=0.58
        )
        pendidikan_norm = _match_with_ctc_fallback(
            col_pendidikan[r], LIST_PENDIDIKAN, 0.44, t1_c7, r, t1_x[7], t1_x[8], accept_ratio=0.56
        )
        pekerjaan_norm = _match_with_ctc_fallback(
            col_pekerjaan[r], LIST_PEKERJAAN, 0.45, t1_c8, r, t1_x[8], t1_x[9], accept_ratio=0.56
        )

        gd_clean = _strip_non_latin(col_goldar[r], r"A-Z\s-")
        if len(re.sub(r"[^A-Z]", "", gd_clean)) >= 4:
            goldar_norm = "TIDAK TAHU"
        else:
            goldar_norm = _match_closest_vocab(gd_clean, LIST_GOL_DARAH, min_ratio=0.45)
            if goldar_norm not in LIST_GOL_DARAH:
                goldar_norm = "-"

        stat_kawin_norm = _match_with_ctc_fallback(
            col_stat_kawin[r], LIST_STATUS_KAWIN, 0.50, t2_c1, r, t2_x[1], t2_x[2], accept_ratio=0.56
        )
        tgl_kawin_norm = _format_tanggal(col_tgl_kawin[r])
        stat_hub_norm = _match_with_ctc_fallback(
            col_stat_hub[r], LIST_STATUS_HUBUNGAN, 0.48, t2_c3, r, t2_x[3], t2_x[4], accept_ratio=0.56
        )
        wni_up = _strip_non_latin(col_wni[r], r"A-Z")
        wni_norm = "WNA" if wni_up == "WNA" else "WNI"

        paspor_c = _strip_non_latin(col_paspor[r], r"A-Z0-9-")
        kitap_c = _strip_non_latin(col_kitap[r], r"A-Z0-9-")
        paspor_norm = paspor_c if len(re.sub(r"[^A-Z0-9]", "", paspor_c)) >= 5 else "-"
        kitap_norm = kitap_c if len(re.sub(r"[^A-Z0-9]", "", kitap_c)) >= 5 else "-"

        ayah_norm = _normalize_spaced_text(_strip_non_latin(col_ayah[r], r"A-Z\s.,'"), COMMON_NAME_TOKENS) or "-"
        ibu_norm = _normalize_spaced_text(_strip_non_latin(col_ibu[r], r"A-Z\s.,'"), COMMON_NAME_TOKENS) or "-"

        anggota_list.append(
            {
                "no_urut": r + 1,
                "nama_lengkap": nama_norm,
                "nik": nik_norm,
                "jenis_kelamin": jk_norm or "LAKI-LAKI",
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
                "no_paspor": paspor_norm,
                "no_kitap": kitap_norm,
                "nama_ayah": ayah_norm,
                "nama_ibu": ibu_norm,
            }
        )

    return anggota_list


def _parse_tables_from_full_tokens(
    tokens: List[Dict[str, Any]], identitas: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Parser baris tabel dinamis berbasis jangkar posisi kolom header Tabel 1 & Tabel 2
    dari token OCR halaman penuh (tahan terhadap kertas melengkung/kurva halus).
    """
    if not tokens:
        return []

    heights = [t["h"] for t in tokens if t["h"] > 4]
    med_h = float(np.median(heights)) if heights else 20.0
    y_tol = max(10.0, med_h * 0.65)

    # 1. Temukan jangkar posisi header vertikal & posisi horizontal setiap kolom secara dinamis
    t1_header_ny, t2_header_ny, footer_ny = 0.25, 0.56, 0.84
    t1_col_anchors: Dict[str, float] = {}
    t2_col_anchors: Dict[str, float] = {}

    for t in tokens:
        up = t["text"].upper()
        ny, nx = t["ny"], t["nx"]
        if ny < 0.44:
            if re.search(r"NAMA\s*LENGKAP|LAMA\s*LENGKAP|\(1\)", up):
                t1_col_anchors.setdefault("nama", nx)
                t1_header_ny = max(t1_header_ny, ny + 0.012)
            elif re.fullmatch(r"NIK|\(2\)", up):
                t1_col_anchors["nik"] = nx
                t1_header_ny = max(t1_header_ny, ny + 0.012)
            elif re.search(r"KELAMIN|JENIS", up) and nx < 0.42:
                t1_col_anchors["jk"] = nx
                t1_header_ny = max(t1_header_ny, ny + 0.012)
            elif re.search(r"TEMPAT|TEMPAL", up):
                t1_col_anchors["tempat"] = nx
                t1_header_ny = max(t1_header_ny, ny + 0.012)
            elif re.search(r"TANGGAL|TANGA|LAHIR|LAHR", up) and nx > 0.40:
                t1_col_anchors.setdefault("tgl_lahir", nx)
                t1_header_ny = max(t1_header_ny, ny + 0.012)
            elif re.search(r"AGAMA", up):
                t1_col_anchors["agama"] = nx
            elif re.search(r"PENDID|PUROOL", up):
                t1_col_anchors["pendidikan"] = nx
            elif re.search(r"PEKERJA", up):
                t1_col_anchors["pekerjaan"] = nx
            elif re.search(r"DARAH|GOLONG", up):
                t1_col_anchors["goldar"] = nx
        elif 0.45 <= ny <= 0.72:
            if re.search(r"\(10\)|PERKAWIN", up) and nx < 0.12:
                t2_col_anchors["stat_kawin"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(11\)", up) or (re.search(r"TANGGAL|PERKAWIN", up) and 0.12 <= nx <= 0.19):
                t2_col_anchors["tgl_kawin"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(12\)|HUBUNG|KELUARGA", up) and 0.18 <= nx <= 0.30:
                t2_col_anchors["stat_hub"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(13\)|KEWARGA", up):
                t2_col_anchors["wni"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(14\)|PASPOR", up):
                t2_col_anchors["paspor"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(15\)|KITAP|KJTAP", up):
                t2_col_anchors["kitap"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(16\)|^AYAH$|^AEH$", up):
                t2_col_anchors["ayah"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
            elif re.search(r"\(17\)|^IBU$", up):
                t2_col_anchors["ibu"] = nx
                t2_header_ny = max(t2_header_ny, ny + 0.008)
        elif ny > 0.75 and ("DIKELUARKAN" in up or "KEPALA DINAS" in up or "KEPALA DNAS" in up):
            footer_ny = min(footer_ny, ny - 0.015)

    # Estimasi posisi kolom yang belum terbaca di header berdasarkan jangkar NIK & Tempat Lahir
    nik_x = t1_col_anchors.get("nik", 0.25)
    jk_x = t1_col_anchors.get("jk", nik_x + 0.075)
    tempat_x = t1_col_anchors.get("tempat", jk_x + 0.085)
    tgl_x = t1_col_anchors.get("tgl_lahir", tempat_x + 0.095)
    agama_x = t1_col_anchors.get("agama", tgl_x + 0.065)
    pend_x = t1_col_anchors.get("pendidikan", agama_x + 0.095)
    pek_x = t1_col_anchors.get("pekerjaan", pend_x + 0.145)
    gd_x = t1_col_anchors.get("goldar", min(0.95, pek_x + 0.12))

    # 2. Kelompokkan baris data Tabel 1 menggunakan jangkar NIK 15-16 digit!
    t1_tokens = [t for t in tokens if t1_header_ny < t["ny"] < (t2_header_ny - 0.04)]
    nik_tokens = []
    for t in sorted(t1_tokens, key=lambda x: x["ny"]):
        cand = _clean_digits(t["text"])
        if 15 <= len(cand) <= 16 and abs(t["nx"] - nik_x) <= 0.14:
            nik_tokens.append((t, cand[:16]))

    no_kk = (identitas or {}).get("no_kk", "")
    upper_rows: List[Dict[str, Any]] = []

    for idx, (ntok, nval) in enumerate(nik_tokens):
        row_cy = ntok["cy"]
        # Kumpulkan semua token pada baris vertikal yang sama (toleransi melengkung halus proporsional dx)
        row_toks = [
            t for t in t1_tokens
            if t is not ntok
            and abs(t["cy"] - row_cy) <= max(y_tol, 14.0 + abs(t["nx"] - ntok["nx"]) * 28.0)
        ]
        # Hindari menyerap token yang lebih dekat ke baris NIK tetangga
        filtered_row_toks = []
        for t in row_toks:
            dist_self = abs(t["cy"] - row_cy)
            if all(dist_self <= abs(t["cy"] - other[0]["cy"]) for j, other in enumerate(nik_tokens) if j != idx):
                filtered_row_toks.append(t)

        cols_t1: Dict[str, List[str]] = {
            "nama": [], "jk": [], "tempat": [], "tgl_lahir": [],
            "agama": [], "pendidikan": [], "pekerjaan": [], "goldar": [],
        }
        boundaries = [
            ("nama", (nik_x + 0.0) / 2.0),
            ("jk", jk_x),
            ("tempat", tempat_x),
            ("tgl_lahir", tgl_x),
            ("agama", agama_x),
            ("pendidikan", pend_x),
            ("pekerjaan", pek_x),
            ("goldar", gd_x),
        ]
        for t in sorted(filtered_row_toks, key=lambda x: x["nx"]):
            txt = _strip_non_latin(t["text"], r"A-Z0-9/\s.,'-")
            if not txt or re.fullmatch(r"[-—_.:,\s()0-9]{1,2}", txt):
                continue
            if t["x_max"] <= ntok["x_min"] + 8:
                cols_t1["nama"].append(txt)
                continue
            # Pilih kolom dengan pusat horizontal terdekat
            best_col = min(boundaries[1:], key=lambda kv: abs(t["nx"] - kv[1]))[0]
            cols_t1[best_col].append(txt)

        nik_clean = _extract_nik_16(nval, no_kk=no_kk)
        jk_raw = " ".join(cols_t1["jk"])
        jk_norm = "PEREMPUAN" if ("P" in jk_raw and "L" not in jk_raw[:2]) else ("LAKI-LAKI" if "L" in jk_raw else "")
        tempat_norm = _normalize_spaced_text(" ".join(cols_t1["tempat"]), COMMON_LOCATION_TOKENS) or "-"
        tgl_norm = _format_tanggal(" ".join(cols_t1["tgl_lahir"]))
        agama_raw = " ".join(cols_t1["agama"])
        agama_norm = _match_closest_vocab(agama_raw, LIST_AGAMA, min_ratio=0.45) if len(agama_raw) >= 3 else "-"
        pend_raw = " ".join(cols_t1["pendidikan"])
        pend_norm = _match_closest_vocab(pend_raw, LIST_PENDIDIKAN, min_ratio=0.40) if len(pend_raw) >= 3 else "-"
        pek_raw = " ".join(cols_t1["pekerjaan"])
        pek_norm = _match_closest_vocab(pek_raw, LIST_PEKERJAAN, min_ratio=0.40) if len(pek_raw) >= 3 else "-"
        gd_raw = " ".join(cols_t1["goldar"])
        gd_norm = "TIDAK TAHU" if len(re.sub(r"[^A-Z]", "", gd_raw)) >= 4 else (_match_closest_vocab(gd_raw, LIST_GOL_DARAH, min_ratio=0.45) if gd_raw else "-")

        upper_rows.append(
            {
                "no_urut": idx + 1,
                "nama_lengkap": _normalize_spaced_text(" ".join(cols_t1["nama"]), COMMON_NAME_TOKENS),
                "nik": nik_clean,
                "jenis_kelamin": jk_norm or "LAKI-LAKI",
                "tempat_lahir": tempat_norm,
                "tanggal_lahir": tgl_norm,
                "agama": agama_norm,
                "pendidikan": pend_norm,
                "jenis_pekerjaan": pek_norm,
                "golongan_darah": gd_norm if gd_norm in LIST_GOL_DARAH else "-",
                "status_perkawinan": "-",
                "tanggal_perkawinan": "-",
                "status_hubungan": "-",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "-",
                "nama_ibu": "-",
            }
        )

    if not upper_rows:
        return []

    # 3. Kelompokkan baris data Tabel 2 menggunakan jangkar kolom (10)..(17)
    t2_tokens = [t for t in tokens if t2_header_ny < t["ny"] < footer_ny]
    t2_lines: List[List[Dict[str, Any]]] = []
    for tok in sorted(t2_tokens, key=lambda t: t["cy"]):
        placed = False
        for line in t2_lines:
            if abs(tok["cy"] - float(np.mean([x["cy"] for x in line]))) <= max(y_tol, 16.0):
                line.append(tok)
                placed = True
                break
        if not placed:
            t2_lines.append([tok])

    sk_x = t2_col_anchors.get("stat_kawin", 0.08)
    tk_x = max(sk_x + 0.055, t2_col_anchors.get("tgl_kawin", sk_x + 0.07))
    sh_x = max(tk_x + 0.055, t2_col_anchors.get("stat_hub", tk_x + 0.075))
    wn_x = max(sh_x + 0.065, t2_col_anchors.get("wni", sh_x + 0.085))
    pa_x = max(wn_x + 0.065, t2_col_anchors.get("paspor", wn_x + 0.08))
    ki_x = max(pa_x + 0.060, t2_col_anchors.get("kitap", pa_x + 0.07))
    ay_x = max(ki_x + 0.085, t2_col_anchors.get("ayah", ki_x + 0.12))
    ib_x = max(ay_x + 0.145, t2_col_anchors.get("ibu", ay_x + 0.18))

    t2_bounds = [
        ("stat_kawin", sk_x),
        ("tgl_kawin", tk_x),
        ("stat_hub", sh_x),
        ("wni", wn_x),
        ("paspor", pa_x),
        ("kitap", ki_x),
        ("ayah", ay_x),
        ("ibu", ib_x),
    ]

    lower_rows: List[Dict[str, str]] = []
    for line in t2_lines:
        if len(lower_rows) >= len(upper_rows):
            break
        cols_t2: Dict[str, List[str]] = {k: [] for k, _ in t2_bounds}
        for t in sorted(line, key=lambda x: x["nx"]):
            txt = _strip_non_latin(t["text"], r"A-Z0-9/\s.,'-")
            if not txt or re.fullmatch(r"[-—_.:,\s()0-9]{1,2}", txt):
                continue
            # Tanggal pada Tabel 2 selalu merupakan kolom (11) Tanggal Perkawinan
            if _format_tanggal(txt) != "-" and 0.10 <= t["nx"] <= 0.25:
                cols_t2["tgl_kawin"].append(txt)
                continue
            best_col = min(t2_bounds, key=lambda kv: abs(t["nx"] - kv[1]))[0]
            cols_t2[best_col].append(txt)

        sk_raw = " ".join(cols_t2["stat_kawin"])
        sh_raw = " ".join(cols_t2["stat_hub"])
        ayah_raw = " ".join(cols_t2["ayah"])
        ibu_raw = " ".join(cols_t2["ibu"])
        if not (sk_raw or sh_raw or ayah_raw or ibu_raw):
            continue

        sk_norm = _match_closest_vocab(sk_raw, LIST_STATUS_KAWIN, min_ratio=0.45) if len(sk_raw) >= 4 else "-"
        sh_norm = _match_closest_vocab(sh_raw, LIST_STATUS_HUBUNGAN, min_ratio=0.45) if len(sh_raw) >= 3 else "-"
        wni_raw = " ".join(cols_t2["wni"])
        lower_rows.append(
            {
                "status_perkawinan": sk_norm if sk_norm in LIST_STATUS_KAWIN else (sk_raw or "-"),
                "tanggal_perkawinan": _format_tanggal(" ".join(cols_t2["tgl_kawin"])),
                "status_hubungan": sh_norm if sh_norm in LIST_STATUS_HUBUNGAN else (sh_raw or "-"),
                "kewarganegaraan": "WNA" if wni_raw == "WNA" else "WNI",
                "no_paspor": " ".join(cols_t2["paspor"]) or "-",
                "no_kitap": " ".join(cols_t2["kitap"]) or "-",
                "nama_ayah": _normalize_spaced_text(ayah_raw, COMMON_NAME_TOKENS) or "-",
                "nama_ibu": _normalize_spaced_text(ibu_raw, COMMON_NAME_TOKENS) or "-",
            }
        )

    for idx, u in enumerate(upper_rows):
        if idx < len(lower_rows):
            u.update(lower_rows[idx])
    return upper_rows


# ==============================================================================
# PIPELINE UTAMA EKSTRAKSI IDENTITAS KK & ANGGOTA KELUARGA
# ==============================================================================

def run_image_processing_pipeline(image_path: str, output_dir: str = "") -> Dict[str, Any]:
    """
    Menjalankan seluruh tahapan Pemrosesan Citra Digital pada foto/scan Kartu Keluarga
    (pelurusan dokumen, normalisasi pencahayaan, histogram & contrast stretching,
    morfologi grid tabel, dan Column-Strip OCR) lalu mengembalikan struktur:
    - identitas (untuk tbl_identitasKK)
    - anggota (untuk tbl_anggotaKeluargaKK)
    """
    img_bgr, norm_gray, _ = preprocess_citra_kk(image_path)
    enh_gray, binary_inv = _build_enhanced_views(img_bgr)
    h, _w = norm_gray.shape

    # Deteksi grid Tabel 1 & Tabel 2 lebih dulu untuk jangkar presisi header & ekstraksi per-kolom
    horiz_lines, t1_y, t2_y, (tx0, tx1) = _detect_kk_table_rows(enh_gray, binary_inv)

    # Baca token halaman penuh dari citra yang sudah dipertajam halus (lebih presisi pada header/footer kabur)
    tokens = _run_full_ocr_tokens(enh_gray)
    if any("KARTU" in t["text"].upper() and "KELUARGA" in t["text"].upper() and t["ny"] > 0.65 for t in tokens):
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
        norm_gray = cv2.rotate(norm_gray, cv2.ROTATE_180)
        enh_gray, binary_inv = _build_enhanced_views(img_bgr)
        horiz_lines, t1_y, t2_y, (tx0, tx1) = _detect_kk_table_rows(enh_gray, binary_inv)
        tokens = _run_full_ocr_tokens(enh_gray)

    t1_row_h = float(np.median(np.diff(t1_y))) if (t1_y and len(t1_y) >= 3) else 34.0
    t1_top_ny = (t1_y[0] - t1_row_h * 3.1) / h if t1_y else None
    t2_bot_ny = t2_y[-1] / h if t2_y else None

    # 1. Ekstrak Header Identitas KK (tbl_identitasKK) + zoom crop terfokus bila ada field kosong/kabur
    identitas = _extract_identitas_header(tokens, t1_top_ny=t1_top_ny, t2_bot_ny=t2_bot_ny)
    _enhance_identitas_from_zoomed_header(norm_gray, identitas, t1_y, tx0, tx1, tokens)

    # Jika tanggal_dikeluarkan atau kabupaten_kota belum terbaca, zoom crop area footer
    if (not identitas["tanggal_dikeluarkan"] or len(identitas["kabupaten_kota"]) < 4) and t2_y:
        fy0 = min(h - 1, t2_y[-1] + 5)
        fy1 = min(h, t2_y[-1] + 220)
        if not identitas["tanggal_dikeluarkan"]:
            fx0 = max(0, tx0)
            fx1 = min(norm_gray.shape[1], tx0 + int((tx1 - tx0) * 0.48))
            if fy1 - fy0 > 20 and fx1 - fx0 > 80:
                footer_up = cv2.resize(enh_gray[fy0:fy1, fx0:fx1], (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
                f_res, _ = _OCR_STRIP(footer_up)
                for _b, f_txt, _s in (f_res or []):
                    tgl_f = _format_tanggal(str(f_txt))
                    if tgl_f != "-":
                        identitas["tanggal_dikeluarkan"] = tgl_f
                        break
        if len(identitas["kabupaten_kota"]) < 4:
            fx0 = max(0, tx0 + int((tx1 - tx0) * 0.50))
            fx1 = min(norm_gray.shape[1], tx1 + 20)
            if fy1 - fy0 > 20 and fx1 - fx0 > 80:
                footer_r = cv2.resize(enh_gray[fy0:fy1, fx0:fx1], (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                fr_res, _ = _OCR_STRIP(footer_r)
                for _b, f_txt, _s in (fr_res or []):
                    m_kab = re.search(
                        r"(KABUPATEN|KOTA)\s*([A-Z][A-Z\s]{2,28})$",
                        _strip_non_latin(str(f_txt), r"A-Z\s"),
                    )
                    if m_kab:
                        identitas["kabupaten_kota"] = _normalize_spaced_text(
                            f"{m_kab.group(1)} {m_kab.group(2)}".strip(),
                            COMMON_LOCATION_TOKENS,
                        )
                        break

    # 2. Ekstrak Anggota Keluarga (prioritas utama: Column-Strip OCR berbasis grid tabel + fusi token penuh!)
    anggota: List[Dict[str, Any]] = []
    if t1_y and t2_y:
        anggota = _extract_anggota_via_column_strips(
            norm_gray, horiz_lines, binary_inv, t1_y, t2_y, tx0, tx1, identitas, full_tokens=tokens
        )

    # Jika foto melengkung sehingga grid garis tidak terdeteksi atau kolom utama kosong, gunakan parser token dinamis
    parsed_tokens = _parse_tables_from_full_tokens(tokens, identitas=identitas)
    if not anggota:
        anggota = parsed_tokens
    elif parsed_tokens:
        # Bandingkan kualitas ekstraksi (jumlah NIK 16 digit valid & sel terisi): pilih yang jauh lebih lengkap
        def _quality_score(rows: List[Dict[str, Any]]) -> int:
            sc = 0
            for r in rows:
                if len(r.get("nik", "")) == 16:
                    sc += 5
                for f in ("tempat_lahir", "tanggal_lahir", "agama", "pendidikan", "jenis_pekerjaan", "status_perkawinan", "status_hubungan", "nama_ayah", "nama_ibu"):
                    v = (r.get(f) or "").strip()
                    if v and v != "-":
                        sc += 1
            return sc

        if _quality_score(parsed_tokens) > _quality_score(anggota) + 3:
            anggota = parsed_tokens

    # 3. Validasi silang konsistensi internal KK (NIK <-> Tgl Lahir/JK, Kepala Keluarga, Ayah/Ibu)
    _reconcile_family_consistency(identitas, anggota)

    return {
        "identitas": identitas,
        "anggota": anggota,
    }

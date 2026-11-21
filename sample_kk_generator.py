import os
from typing import Any, Dict, List
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "static", "samples")


SAMPLE_KK_DATASETS: List[Dict[str, Any]] = [
    {
        "filename": "sample_kk_1_bogor.png",
        "label": "Contoh KK 1 - Keluarga Budi Santoso (Kota Bogor, 4 Anggota)",
        "identitas": {
            "no_kk": "3271041508200001",
            "nama_kepala_keluarga": "BUDI SANTOSO",
            "alamat": "JL. PAJAJARAN INDAH V NO. 18",
            "rt_rw": "004/009",
            "kode_pos": "16143",
            "desa_kelurahan": "BARANANGSIANG",
            "kecamatan": "BOGOR TIMUR",
            "kabupaten_kota": "KOTA BOGOR",
            "provinsi": "JAWA BARAT",
            "tanggal_dikeluarkan": "14-02-2023",
        },
        "anggota": [
            {
                "no_urut": 1,
                "nama_lengkap": "BUDI SANTOSO",
                "nik": "3271041205800001",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "BOGOR",
                "tanggal_lahir": "12-05-1980",
                "agama": "ISLAM",
                "pendidikan": "DIPLOMA IV/STRATA I",
                "jenis_pekerjaan": "KARYAWAN SWASTA",
                "golongan_darah": "O",
                "status_perkawinan": "KAWIN TERCATAT",
                "tanggal_perkawinan": "20-06-2006",
                "status_hubungan": "KEPALA KELUARGA",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "HENDRA SANTOSO",
                "nama_ibu": "SITI AMINAH",
            },
            {
                "no_urut": 2,
                "nama_lengkap": "DEWI LESTARI",
                "nik": "3271045809830002",
                "jenis_kelamin": "PEREMPUAN",
                "tempat_lahir": "BANDUNG",
                "tanggal_lahir": "18-09-1983",
                "agama": "ISLAM",
                "pendidikan": "DIPLOMA IV/STRATA I",
                "jenis_pekerjaan": "GURU",
                "golongan_darah": "A",
                "status_perkawinan": "KAWIN TERCATAT",
                "tanggal_perkawinan": "20-06-2006",
                "status_hubungan": "ISTRI",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "AHMAD HIDAYAT",
                "nama_ibu": "RATNA SARI",
            },
            {
                "no_urut": 3,
                "nama_lengkap": "RAFI PRATAMA SANTOSO",
                "nik": "3271042403080003",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "BOGOR",
                "tanggal_lahir": "24-03-2008",
                "agama": "ISLAM",
                "pendidikan": "SLTA/SEDERAJAT",
                "jenis_pekerjaan": "PELAJAR/MAHASISWA",
                "golongan_darah": "O",
                "status_perkawinan": "BELUM KAWIN",
                "tanggal_perkawinan": "-",
                "status_hubungan": "ANAK",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "BUDI SANTOSO",
                "nama_ibu": "DEWI LESTARI",
            },
            {
                "no_urut": 4,
                "nama_lengkap": "NADIA PUTRI SANTOSO",
                "nik": "3271046111120004",
                "jenis_kelamin": "PEREMPUAN",
                "tempat_lahir": "BOGOR",
                "tanggal_lahir": "21-11-2012",
                "agama": "ISLAM",
                "pendidikan": "SLTP/SEDERAJAT",
                "jenis_pekerjaan": "PELAJAR/MAHASISWA",
                "golongan_darah": "A",
                "status_perkawinan": "BELUM KAWIN",
                "tanggal_perkawinan": "-",
                "status_hubungan": "ANAK",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "BUDI SANTOSO",
                "nama_ibu": "DEWI LESTARI",
            },
        ],
    },
    {
        "filename": "sample_kk_2_jakarta.png",
        "label": "Contoh KK 2 - Keluarga Irwan Setiawan (Jakarta Selatan, 3 Anggota)",
        "identitas": {
            "no_kk": "3174082209190005",
            "nama_kepala_keluarga": "IRWAN SETIAWAN",
            "alamat": "JL. TEBET TIMUR DALAM RAYA NO. 27",
            "rt_rw": "002/005",
            "kode_pos": "12820",
            "desa_kelurahan": "TEBET TIMUR",
            "kecamatan": "TEBET",
            "kabupaten_kota": "JAKARTA SELATAN",
            "provinsi": "DKI JAKARTA",
            "tanggal_dikeluarkan": "09-10-2024",
        },
        "anggota": [
            {
                "no_urut": 1,
                "nama_lengkap": "IRWAN SETIAWAN",
                "nik": "3174080504780001",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "JAKARTA",
                "tanggal_lahir": "05-04-1978",
                "agama": "ISLAM",
                "pendidikan": "STRATA II",
                "jenis_pekerjaan": "PNS",
                "golongan_darah": "B",
                "status_perkawinan": "KAWIN TERCATAT",
                "tanggal_perkawinan": "12-12-2004",
                "status_hubungan": "KEPALA KELUARGA",
                "kewarganegaraan": "WNI",
                "no_paspor": "C8821940",
                "no_kitap": "-",
                "nama_ayah": "SOEHARTO SETIAWAN",
                "nama_ibu": "MARYAM",
            },
            {
                "no_urut": 2,
                "nama_lengkap": "RINA WULANDARI",
                "nik": "3174085007810002",
                "jenis_kelamin": "PEREMPUAN",
                "tempat_lahir": "SEMARANG",
                "tanggal_lahir": "10-07-1981",
                "agama": "ISLAM",
                "pendidikan": "DIPLOMA IV/STRATA I",
                "jenis_pekerjaan": "WIRASWASTA",
                "golongan_darah": "AB",
                "status_perkawinan": "KAWIN TERCATAT",
                "tanggal_perkawinan": "12-12-2004",
                "status_hubungan": "ISTRI",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "BAMBANG SUTRISNO",
                "nama_ibu": "SRI WAHYUNI",
            },
            {
                "no_urut": 3,
                "nama_lengkap": "KEVIN ADITYA SETIAWAN",
                "nik": "3174081901060003",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "JAKARTA",
                "tanggal_lahir": "19-01-2006",
                "agama": "ISLAM",
                "pendidikan": "SLTA/SEDERAJAT",
                "jenis_pekerjaan": "PELAJAR/MAHASISWA",
                "golongan_darah": "B",
                "status_perkawinan": "BELUM KAWIN",
                "tanggal_perkawinan": "-",
                "status_hubungan": "ANAK",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "IRWAN SETIAWAN",
                "nama_ibu": "RINA WULANDARI",
            },
        ],
    },
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Memuat font sistem Windows (Arial / Calibri / Segoe UI) untuk merender citra KK."""
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_kk_image(dataset: Dict[str, Any], output_path: str) -> str:
    """
    Merender citra Kartu Keluarga (KK) beresolusi tinggi (2200 x 1550 px)
    dengan tata letak standar blanko Kartu Keluarga Indonesia agar siap
    diproses oleh pipeline OpenCV dan OCR.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    W, H = 2200, 1550
    # Latar belakang kertas KK (putih kebiruan halus khas dokumen KK)
    img = Image.new("RGB", (W, H), color=(248, 251, 255))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(44, bold=True)
    font_nokk = _load_font(34, bold=True)
    font_header_b = _load_font(21, bold=True)
    font_header = _load_font(21, bold=False)
    font_th = _load_font(17, bold=True)
    font_td = _load_font(18, bold=False)

    # Bingkai luar dokumen
    draw.rectangle([25, 25, W - 25, H - 25], outline=(40, 60, 90), width=3)

    # Judul KARTU KELUARGA & Nomor KK
    ident = dataset["identitas"]
    title_text = "KARTU KELUARGA"
    draw.text((W // 2 - 200, 48), title_text, fill=(15, 25, 45), font=font_title)
    draw.text((W // 2 - 210, 102), f"No. {ident['no_kk']}", fill=(15, 25, 45), font=font_nokk)

    # Identitas KK Kiri & Kanan
    left_fields = [
        ("Nama Kepala Keluarga", ident["nama_kepala_keluarga"]),
        ("Alamat", ident["alamat"]),
        ("RT/RW", ident["rt_rw"]),
        ("Kode Pos", ident["kode_pos"]),
    ]
    right_fields = [
        ("Desa/Kelurahan", ident["desa_kelurahan"]),
        ("Kecamatan", ident["kecamatan"]),
        ("Kabupaten/Kota", ident["kabupaten_kota"]),
        ("Provinsi", ident["provinsi"]),
    ]

    y_start = 165
    line_gap = 34
    for idx, (label, val) in enumerate(left_fields):
        y = y_start + idx * line_gap
        draw.text((60, y), label, fill=(20, 30, 50), font=font_header_b)
        draw.text((330, y), f": {val}", fill=(10, 15, 25), font=font_header)

    for idx, (label, val) in enumerate(right_fields):
        y = y_start + idx * line_gap
        draw.text((1320, y), label, fill=(20, 30, 50), font=font_header_b)
        draw.text((1560, y), f": {val}", fill=(10, 15, 25), font=font_header)

    # ==========================================================================
    # TABEL 1 (ATAS): Kolom 1 - 9
    # ==========================================================================
    t1_x = [50, 105, 470, 760, 955, 1175, 1365, 1515, 1795, 2035, 2150]
    t1_headers = [
        "No",
        "Nama Lengkap",
        "NIK",
        "Jenis Kelamin",
        "Tempat Lahir",
        "Tanggal Lahir",
        "Agama",
        "Pendidikan",
        "Jenis Pekerjaan",
        "Gol. Darah",
    ]
    t1_nums = ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)"]

    t1_y0 = 320
    t1_header_h = 46
    t1_num_h = 28
    row_h = 42
    max_rows = 8

    # Warna latar header tabel
    draw.rectangle([t1_x[0], t1_y0, t1_x[-1], t1_y0 + t1_header_h + t1_num_h], fill=(225, 236, 248))

    for i in range(len(t1_headers)):
        cx = (t1_x[i] + t1_x[i + 1]) // 2
        draw.text((t1_x[i] + 10, t1_y0 + 12), t1_headers[i], fill=(15, 25, 45), font=font_th)
        if t1_nums[i]:
            draw.text((cx - 12, t1_y0 + t1_header_h + 4), t1_nums[i], fill=(40, 55, 80), font=font_th)

    # Isi baris Tabel 1
    anggota_list = dataset.get("anggota", [])
    t1_data_y0 = t1_y0 + t1_header_h + t1_num_h

    for r in range(max_rows):
        ry = t1_data_y0 + r * row_h
        if r < len(anggota_list):
            a = anggota_list[r]
            vals = [
                str(a["no_urut"]),
                a["nama_lengkap"],
                a["nik"],
                a["jenis_kelamin"],
                a["tempat_lahir"],
                a["tanggal_lahir"],
                a["agama"],
                a["pendidikan"],
                a["jenis_pekerjaan"],
                a["golongan_darah"],
            ]
        else:
            vals = [str(r + 1), "-", "-", "-", "-", "-", "-", "-", "-", "-"]

        for c_idx, val in enumerate(vals):
            draw.text((t1_x[c_idx] + 10, ry + 10), val, fill=(10, 15, 20), font=font_td)

    t1_y_end = t1_data_y0 + max_rows * row_h
    # Garis horizontal Tabel 1
    draw.line([t1_x[0], t1_y0, t1_x[-1], t1_y0], fill=(20, 30, 45), width=2)
    draw.line([t1_x[0], t1_y0 + t1_header_h, t1_x[-1], t1_y0 + t1_header_h], fill=(20, 30, 45), width=2)
    draw.line([t1_x[0], t1_data_y0, t1_x[-1], t1_data_y0], fill=(20, 30, 45), width=2)
    for r in range(1, max_rows + 1):
        y_line = t1_data_y0 + r * row_h
        draw.line([t1_x[0], y_line, t1_x[-1], y_line], fill=(60, 75, 95), width=2)
    # Garis vertikal Tabel 1
    for x_line in t1_x:
        draw.line([x_line, t1_y0, x_line, t1_y_end], fill=(20, 30, 45), width=2)

    # ==========================================================================
    # TABEL 2 (BAWAH): Kolom 10 - 17
    # ==========================================================================
    t2_x = [50, 105, 360, 585, 895, 1110, 1335, 1560, 1855, 2150]
    t2_headers = [
        "No",
        "Status Perkawinan",
        "Tgl Perkawinan",
        "Status Hubungan",
        "Kewarganegaraan",
        "No. Paspor",
        "No. KITAP",
        "Nama Ayah",
        "Nama Ibu",
    ]
    t2_nums = ["", "(10)", "(11)", "(12)", "(13)", "(14)", "(15)", "(16)", "(17)"]

    t2_y0 = t1_y_end + 45
    t2_header_h = 46
    t2_num_h = 28
    t2_data_y0 = t2_y0 + t2_header_h + t2_num_h

    draw.rectangle([t2_x[0], t2_y0, t2_x[-1], t2_data_y0], fill=(225, 236, 248))

    for i in range(len(t2_headers)):
        cx = (t2_x[i] + t2_x[i + 1]) // 2
        draw.text((t2_x[i] + 10, t2_y0 + 12), t2_headers[i], fill=(15, 25, 45), font=font_th)
        if t2_nums[i]:
            draw.text((cx - 14, t2_y0 + t2_header_h + 4), t2_nums[i], fill=(40, 55, 80), font=font_th)

    for r in range(max_rows):
        ry = t2_data_y0 + r * row_h
        if r < len(anggota_list):
            a = anggota_list[r]
            vals = [
                str(a["no_urut"]),
                a["status_perkawinan"],
                a["tanggal_perkawinan"],
                a["status_hubungan"],
                a["kewarganegaraan"],
                a["no_paspor"],
                a["no_kitap"],
                a["nama_ayah"],
                a["nama_ibu"],
            ]
        else:
            vals = [str(r + 1), "-", "-", "-", "-", "-", "-", "-", "-"]

        for c_idx, val in enumerate(vals):
            draw.text((t2_x[c_idx] + 10, ry + 10), val, fill=(10, 15, 20), font=font_td)

    t2_y_end = t2_data_y0 + max_rows * row_h
    draw.line([t2_x[0], t2_y0, t2_x[-1], t2_y0], fill=(20, 30, 45), width=2)
    draw.line([t2_x[0], t2_y0 + t2_header_h, t2_x[-1], t2_y0 + t2_header_h], fill=(20, 30, 45), width=2)
    draw.line([t2_x[0], t2_data_y0, t2_x[-1], t2_data_y0], fill=(20, 30, 45), width=2)
    for r in range(1, max_rows + 1):
        y_line = t2_data_y0 + r * row_h
        draw.line([t2_x[0], y_line, t2_x[-1], y_line], fill=(60, 75, 95), width=2)
    for x_line in t2_x:
        draw.line([x_line, t2_y0, x_line, t2_y_end], fill=(20, 30, 45), width=2)

    # Footer KK
    footer_y = t2_y_end + 35
    draw.text(
        (60, footer_y),
        f"Dikeluarkan Tanggal : {ident['tanggal_dikeluarkan']}",
        fill=(20, 30, 50),
        font=font_header_b,
    )
    draw.text(
        (950, footer_y),
        "KEPALA KELUARGA",
        fill=(20, 30, 50),
        font=font_header_b,
    )
    draw.text(
        (950, footer_y + 90),
        ident["nama_kepala_keluarga"],
        fill=(20, 30, 50),
        font=font_header_b,
    )
    draw.text(
        (1580, footer_y),
        "KEPALA DINAS KEPENDUDUKAN DAN PENCATATAN SIPIL",
        fill=(20, 30, 50),
        font=font_th,
    )

    img.save(output_path, "PNG")
    return output_path


def ensure_sample_images() -> List[Dict[str, Any]]:
    """Memastikan seluruh citra contoh KK tersedia di static/samples."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    results = []

    # 1. Sertakan Dummy KK Uji lokal jika tersedia di folder static/samples
    user_dummy_filename = "dummy_kk_riski_bontang.jpg"
    user_dummy_dst = os.path.join(SAMPLE_DIR, user_dummy_filename)

    if os.path.exists(user_dummy_dst):
        results.append(
            {
                "filename": user_dummy_filename,
                "label": "Dummy KK Uji (Riski Siswono - Kota Bontang)",
                "rel_path": f"samples/{user_dummy_filename}",
                "no_kk": "6474020907200002",
                "kepala": "RISKI SISWONO",
            }
        )

    # 2. Sertakan contoh KK sintetis tambahan
    for ds in SAMPLE_KK_DATASETS:
        out_path = os.path.join(SAMPLE_DIR, ds["filename"])
        if not os.path.exists(out_path):
            render_kk_image(ds, out_path)
        results.append(
            {
                "filename": ds["filename"],
                "label": ds["label"],
                "rel_path": f"samples/{ds['filename']}",
                "no_kk": ds["identitas"]["no_kk"],
                "kepala": ds["identitas"]["nama_kepala_keluarga"],
            }
        )
    return results

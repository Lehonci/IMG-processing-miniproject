import csv
import io
import json
import os
import shutil
import uuid
from typing import Any, Dict, List

from flask import Flask, Response, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import database as db
from image_processor import run_image_processing_pipeline
from sample_kk_generator import SAMPLE_DIR, ensure_sample_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
JOBS_DIR = os.path.join(BASE_DIR, "static", "jobs")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.secret_key = "mini_project_pcd_kk_secret_key"

for folder in (UPLOAD_DIR, JOBS_DIR, SAMPLE_DIR):
    os.makedirs(folder, exist_ok=True)

# Inisialisasi database SQLite dan contoh citra KK
db.init_db()
ensure_sample_images()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_global_context():
    all_kk = db.get_all_kk()
    total_anggota = sum(int(k.get("jumlah_anggota") or 0) for k in all_kk)
    return {
        "nav_total_kk": len(all_kk),
        "nav_total_anggota": total_anggota,
    }


# ==============================================================================
# 1. HALAMAN INPUT CITRA KARTU KELUARGA
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    samples = ensure_sample_images()
    total_kk = len(db.get_all_kk())
    return render_template("input.html", samples=samples, total_kk=total_kk)


@app.route("/process", methods=["POST"])
def process_image():
    """
    Menerima gambar KK yang diunggah pengguna, memprosesnya dengan OpenCV + OCR,
    lalu mengarahkan ke Halaman Konfirmasi (Form Editable).
    """
    sample_name = (request.form.get("sample_filename") or "").strip()
    saved_image_path = ""
    rel_image_path = ""

    if sample_name:
        src_path = os.path.join(SAMPLE_DIR, secure_filename(sample_name))
        if not os.path.exists(src_path):
            flash("File contoh citra KK tidak ditemukan.", "danger")
            return redirect(url_for("index"))
        unique_name = f"kk_{uuid.uuid4().hex[:8]}_{secure_filename(sample_name)}"
        saved_image_path = os.path.join(UPLOAD_DIR, unique_name)
        shutil.copyfile(src_path, saved_image_path)
        rel_image_path = f"uploads/{unique_name}"
    else:
        file = request.files.get("kk_image")
        if not file or not file.filename:
            flash("Silakan pilih file gambar Kartu Keluarga terlebih dahulu.", "warning")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Format file tidak didukung. Gunakan JPG, JPEG, PNG, WEBP, atau BMP.", "danger")
            return redirect(url_for("index"))

        unique_name = f"kk_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        saved_image_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(saved_image_path)
        rel_image_path = f"uploads/{unique_name}"

    try:
        hasil = run_image_processing_pipeline(saved_image_path)
    except Exception as exc:
        flash(f"Terjadi kesalahan saat membaca citra KK: {exc}", "danger")
        return redirect(url_for("index"))

    if not hasil["anggota"]:
        hasil["anggota"] = [
            {
                "no_urut": 1,
                "nama_lengkap": "",
                "nik": "",
                "jenis_kelamin": "LAKI-LAKI",
                "tempat_lahir": "",
                "tanggal_lahir": "",
                "agama": "ISLAM",
                "pendidikan": "",
                "jenis_pekerjaan": "",
                "golongan_darah": "-",
                "status_perkawinan": "BELUM KAWIN",
                "tanggal_perkawinan": "-",
                "status_hubungan": "KEPALA KELUARGA",
                "kewarganegaraan": "WNI",
                "no_paspor": "-",
                "no_kitap": "-",
                "nama_ayah": "",
                "nama_ibu": "",
            }
        ]

    job_id = uuid.uuid4().hex[:12]
    job_payload = {
        "job_id": job_id,
        "rel_image_path": rel_image_path,
        "identitas": hasil["identitas"],
        "anggota": hasil["anggota"],
    }
    with open(os.path.join(JOBS_DIR, f"{job_id}.json"), "w", encoding="utf-8") as f:
        json.dump(job_payload, f, ensure_ascii=False, indent=2)

    flash(
        "Pembacaan citra Kartu Keluarga selesai! Silakan periksa dan koreksi data pada form di bawah ini jika ada kesalahan baca sebelum disimpan ke database.",
        "success",
    )
    return redirect(url_for("confirm_page", job_id=job_id))


# ==============================================================================
# 2. HALAMAN KONFIRMASI HASIL PEMBACAAN (FORM EDITABLE)
# ==============================================================================

@app.route("/confirm/<job_id>", methods=["GET"])
def confirm_page(job_id: str):
    job_file = os.path.join(JOBS_DIR, f"{secure_filename(job_id)}.json")
    if not os.path.exists(job_file):
        flash("Data konfirmasi tidak ditemukan. Silakan unggah ulang citra KK.", "warning")
        return redirect(url_for("index"))

    with open(job_file, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    return render_template("confirm.html", job=job_data)


@app.route("/save-kk", methods=["POST"])
def save_confirmed_kk():
    """Menyimpan data hasil konfirmasi ke tbl_identitasKK dan tbl_anggotaKeluargaKK."""
    identitas = {
        "no_kk": request.form.get("no_kk", "").strip(),
        "nama_kepala_keluarga": request.form.get("nama_kepala_keluarga", "").strip(),
        "alamat": request.form.get("alamat", "").strip(),
        "rt_rw": request.form.get("rt_rw", "").strip(),
        "kode_pos": request.form.get("kode_pos", "").strip(),
        "desa_kelurahan": request.form.get("desa_kelurahan", "").strip(),
        "kecamatan": request.form.get("kecamatan", "").strip(),
        "kabupaten_kota": request.form.get("kabupaten_kota", "").strip(),
        "provinsi": request.form.get("provinsi", "").strip(),
        "tanggal_dikeluarkan": request.form.get("tanggal_dikeluarkan", "").strip(),
        "image_path": request.form.get("image_path", "").strip(),
    }

    if not identitas["no_kk"] or not identitas["nama_kepala_keluarga"]:
        flash("Nomor KK dan Nama Kepala Keluarga wajib diisi!", "danger")
        job_id = request.form.get("job_id", "")
        return redirect(url_for("confirm_page", job_id=job_id) if job_id else url_for("index"))

    fields = [
        "no_urut", "nama_lengkap", "nik", "jenis_kelamin", "tempat_lahir",
        "tanggal_lahir", "agama", "pendidikan", "jenis_pekerjaan", "golongan_darah",
        "status_perkawinan", "tanggal_perkawinan", "status_hubungan",
        "kewarganegaraan", "no_paspor", "no_kitap", "nama_ayah", "nama_ibu",
    ]
    columns = {f: request.form.getlist(f"{f}[]") for f in fields}
    row_count = len(columns["nama_lengkap"])

    anggota_list: List[Dict[str, Any]] = []
    for i in range(row_count):
        nama_val = columns["nama_lengkap"][i].strip() if i < len(columns["nama_lengkap"]) else ""
        nik_val = columns["nik"][i].strip() if i < len(columns["nik"]) else ""
        if not nama_val and not nik_val:
            continue

        row_dict: Dict[str, Any] = {}
        for f in fields:
            val = columns[f][i].strip() if i < len(columns[f]) else ""
            row_dict[f] = val
        row_dict["no_urut"] = int(row_dict["no_urut"] or (i + 1))
        anggota_list.append(row_dict)

    try:
        id_kk = db.create_kk_with_anggota(identitas, anggota_list)
        job_id = request.form.get("job_id", "").strip()
        if job_id:
            job_file = os.path.join(JOBS_DIR, f"{secure_filename(job_id)}.json")
            if os.path.exists(job_file):
                try:
                    os.remove(job_file)
                except OSError:
                    pass
        flash(
            f"Data KK No. {identitas['no_kk']} ({identitas['nama_kepala_keluarga']}) "
            f"beserta {len(anggota_list)} anggota keluarga berhasil disimpan ke database!",
            "success",
        )
        if request.form.get("after_save") == "detail":
            return redirect(url_for("detail_kk", id_kk=id_kk))
    except Exception as exc:
        flash(f"Gagal menyimpan ke database: {exc}", "danger")

    return redirect(url_for("list_kk"))


# ==============================================================================
# 3. HALAMAN PENELUSURAN TABEL KK SAJA (tbl_identitasKK)
# ==============================================================================

def _csv_safe_value(value: Any, force_text_digits: bool = False) -> str:
    """Menyiapkan nilai sel CSV yang aman terhadap formula injection dan menjaga angka 16 digit (No. KK/NIK) di Excel."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if force_text_digits and text.isdigit() and len(text) >= 11:
        return f'="{text}"'
    if text != "-" and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def _build_csv_response(rows: List[List[Any]], filename: str) -> Response:
    """Membangun response download CSV dengan UTF-8 BOM agar langsung rapi saat dibuka di Microsoft Excel."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode("utf-8-sig")
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/kk", methods=["GET"])
def list_kk():
    search_query = request.args.get("q", "").strip()
    selected_provinsi = request.args.get("provinsi", "").strip()
    selected_kabupaten = request.args.get("kabupaten_kota", "").strip()
    kk_list = db.get_all_kk(
        search_query=search_query,
        provinsi=selected_provinsi,
        kabupaten_kota=selected_kabupaten,
    )
    filter_options = db.get_kk_filter_options()
    filtered_total_anggota = sum(int(k.get("jumlah_anggota") or 0) for k in kk_list)
    return render_template(
        "list_kk.html",
        kk_list=kk_list,
        search_query=search_query,
        selected_provinsi=selected_provinsi,
        selected_kabupaten=selected_kabupaten,
        filter_options=filter_options,
        filtered_total_anggota=filtered_total_anggota,
    )


@app.route("/kk/export", methods=["GET"])
def export_kk_list():
    """Mengekspor daftar KK (atau KK beserta seluruh anggota keluarganya) sesuai filter aktif ke CSV."""
    search_query = request.args.get("q", "").strip()
    selected_provinsi = request.args.get("provinsi", "").strip()
    selected_kabupaten = request.args.get("kabupaten_kota", "").strip()
    scope = (request.args.get("scope") or "identitas").strip().lower()

    if scope == "lengkap":
        kk_list = db.get_all_kk_with_anggota(
            search_query=search_query,
            provinsi=selected_provinsi,
            kabupaten_kota=selected_kabupaten,
        )
        rows: List[List[Any]] = [
            [
                "No. KK",
                "Nama Kepala Keluarga",
                "Alamat",
                "RT/RW",
                "Kode Pos",
                "Desa/Kelurahan",
                "Kecamatan",
                "Kabupaten/Kota",
                "Provinsi",
                "Tanggal Dikeluarkan",
                "No. Urut",
                "Nama Lengkap Anggota",
                "NIK",
                "Jenis Kelamin",
                "Tempat Lahir",
                "Tanggal Lahir",
                "Agama",
                "Pendidikan",
                "Jenis Pekerjaan",
                "Gol. Darah",
                "Status Perkawinan",
                "Tanggal Perkawinan",
                "Status Hubungan",
                "Kewarganegaraan",
                "No. Paspor",
                "No. KITAP",
                "Nama Ayah",
                "Nama Ibu",
            ]
        ]
        for kk in kk_list:
            ident_cols = [
                _csv_safe_value(kk.get("no_kk"), force_text_digits=True),
                _csv_safe_value(kk.get("nama_kepala_keluarga")),
                _csv_safe_value(kk.get("alamat")),
                _csv_safe_value(kk.get("rt_rw")),
                _csv_safe_value(kk.get("kode_pos")),
                _csv_safe_value(kk.get("desa_kelurahan")),
                _csv_safe_value(kk.get("kecamatan")),
                _csv_safe_value(kk.get("kabupaten_kota")),
                _csv_safe_value(kk.get("provinsi")),
                _csv_safe_value(kk.get("tanggal_dikeluarkan")),
            ]
            anggota = kk.get("anggota") or []
            if not anggota:
                rows.append(ident_cols + [""] * 18)
            else:
                for a in anggota:
                    rows.append(
                        ident_cols
                        + [
                            a.get("no_urut", ""),
                            _csv_safe_value(a.get("nama_lengkap")),
                            _csv_safe_value(a.get("nik"), force_text_digits=True),
                            _csv_safe_value(a.get("jenis_kelamin")),
                            _csv_safe_value(a.get("tempat_lahir")),
                            _csv_safe_value(a.get("tanggal_lahir")),
                            _csv_safe_value(a.get("agama")),
                            _csv_safe_value(a.get("pendidikan")),
                            _csv_safe_value(a.get("jenis_pekerjaan")),
                            _csv_safe_value(a.get("golongan_darah")),
                            _csv_safe_value(a.get("status_perkawinan")),
                            _csv_safe_value(a.get("tanggal_perkawinan")),
                            _csv_safe_value(a.get("status_hubungan")),
                            _csv_safe_value(a.get("kewarganegaraan")),
                            _csv_safe_value(a.get("no_paspor")),
                            _csv_safe_value(a.get("no_kitap")),
                            _csv_safe_value(a.get("nama_ayah")),
                            _csv_safe_value(a.get("nama_ibu")),
                        ]
                    )
        return _build_csv_response(rows, "data_kk_lengkap.csv")

    kk_list = db.get_all_kk(
        search_query=search_query,
        provinsi=selected_provinsi,
        kabupaten_kota=selected_kabupaten,
    )
    rows = [
        [
            "No",
            "No. KK",
            "Nama Kepala Keluarga",
            "Alamat",
            "RT/RW",
            "Kode Pos",
            "Desa/Kelurahan",
            "Kecamatan",
            "Kabupaten/Kota",
            "Provinsi",
            "Tanggal Dikeluarkan",
            "Jumlah Anggota",
        ]
    ]
    for idx, kk in enumerate(kk_list, start=1):
        rows.append(
            [
                idx,
                _csv_safe_value(kk.get("no_kk"), force_text_digits=True),
                _csv_safe_value(kk.get("nama_kepala_keluarga")),
                _csv_safe_value(kk.get("alamat")),
                _csv_safe_value(kk.get("rt_rw")),
                _csv_safe_value(kk.get("kode_pos")),
                _csv_safe_value(kk.get("desa_kelurahan")),
                _csv_safe_value(kk.get("kecamatan")),
                _csv_safe_value(kk.get("kabupaten_kota")),
                _csv_safe_value(kk.get("provinsi")),
                _csv_safe_value(kk.get("tanggal_dikeluarkan")),
                int(kk.get("jumlah_anggota") or 0),
            ]
        )
    return _build_csv_response(rows, "daftar_kk.csv")


@app.route("/kk/<int:id_kk>/export", methods=["GET"])
def export_single_kk(id_kk: int):
    """Mengekspor detail satu Kartu Keluarga beserta seluruh anggota keluarganya (Kolom 1-17) ke CSV."""
    kk = db.get_kk_by_id(id_kk)
    if not kk:
        flash("Data Kartu Keluarga tidak ditemukan.", "warning")
        return redirect(url_for("list_kk"))

    anggota_list = db.get_anggota_by_kk(id_kk)
    rows: List[List[Any]] = [
        ["KARTU KELUARGA", ""],
        ["No. KK", _csv_safe_value(kk.get("no_kk"), force_text_digits=True)],
        ["Nama Kepala Keluarga", _csv_safe_value(kk.get("nama_kepala_keluarga"))],
        ["Alamat", _csv_safe_value(kk.get("alamat"))],
        ["RT/RW", _csv_safe_value(kk.get("rt_rw"))],
        ["Kode Pos", _csv_safe_value(kk.get("kode_pos"))],
        ["Desa/Kelurahan", _csv_safe_value(kk.get("desa_kelurahan"))],
        ["Kecamatan", _csv_safe_value(kk.get("kecamatan"))],
        ["Kabupaten/Kota", _csv_safe_value(kk.get("kabupaten_kota"))],
        ["Provinsi", _csv_safe_value(kk.get("provinsi"))],
        ["Tanggal Dikeluarkan", _csv_safe_value(kk.get("tanggal_dikeluarkan"))],
        [],
        [
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
            "Status Perkawinan",
            "Tanggal Perkawinan",
            "Status Hubungan",
            "Kewarganegaraan",
            "No. Paspor",
            "No. KITAP",
            "Nama Ayah",
            "Nama Ibu",
        ],
    ]
    for a in anggota_list:
        rows.append(
            [
                a.get("no_urut", ""),
                _csv_safe_value(a.get("nama_lengkap")),
                _csv_safe_value(a.get("nik"), force_text_digits=True),
                _csv_safe_value(a.get("jenis_kelamin")),
                _csv_safe_value(a.get("tempat_lahir")),
                _csv_safe_value(a.get("tanggal_lahir")),
                _csv_safe_value(a.get("agama")),
                _csv_safe_value(a.get("pendidikan")),
                _csv_safe_value(a.get("jenis_pekerjaan")),
                _csv_safe_value(a.get("golongan_darah")),
                _csv_safe_value(a.get("status_perkawinan")),
                _csv_safe_value(a.get("tanggal_perkawinan")),
                _csv_safe_value(a.get("status_hubungan")),
                _csv_safe_value(a.get("kewarganegaraan")),
                _csv_safe_value(a.get("no_paspor")),
                _csv_safe_value(a.get("no_kitap")),
                _csv_safe_value(a.get("nama_ayah")),
                _csv_safe_value(a.get("nama_ibu")),
            ]
        )

    safe_no_kk = secure_filename(kk.get("no_kk") or str(id_kk)) or str(id_kk)
    return _build_csv_response(rows, f"kk_{safe_no_kk}.csv")


@app.route("/kk/create-manual", methods=["POST"])
def create_kk_manual():
    identitas = {
        k: request.form.get(k, "").strip()
        for k in [
            "no_kk", "nama_kepala_keluarga", "alamat", "rt_rw", "kode_pos",
            "desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi", "tanggal_dikeluarkan",
        ]
    }
    identitas["image_path"] = ""
    try:
        id_kk = db.create_kk_with_anggota(identitas, [])
        flash(f"Identitas KK No. {identitas['no_kk']} berhasil ditambahkan.", "success")
        return redirect(url_for("detail_kk", id_kk=id_kk))
    except Exception as exc:
        flash(f"Gagal menambahkan KK: {exc}", "danger")
        return redirect(url_for("list_kk"))


@app.route("/kk/<int:id_kk>/update", methods=["POST"])
def update_kk(id_kk: int):
    data = {
        k: request.form.get(k, "").strip()
        for k in [
            "no_kk", "nama_kepala_keluarga", "alamat", "rt_rw", "kode_pos",
            "desa_kelurahan", "kecamatan", "kabupaten_kota", "provinsi", "tanggal_dikeluarkan",
        ]
    }
    try:
        db.update_identitas_kk(id_kk, data)
        flash(f"Data Identitas KK No. {data['no_kk']} berhasil diperbarui.", "success")
    except Exception as exc:
        flash(f"Gagal memperbarui Identitas KK: {exc}", "danger")

    if request.form.get("redirect_to") == "detail":
        return redirect(url_for("detail_kk", id_kk=id_kk))
    return redirect(url_for("list_kk"))


@app.route("/kk/<int:id_kk>/delete", methods=["POST"])
def delete_kk(id_kk: int):
    kk = db.get_kk_by_id(id_kk)
    if kk:
        db.delete_identitas_kk(id_kk)
        flash(f"Data KK No. {kk['no_kk']} beserta anggota keluarganya berhasil dihapus.", "info")
    return redirect(url_for("list_kk"))


# ==============================================================================
# 4. HALAMAN DETAIL KK & CRUD ANGGOTA KELUARGA (tbl_anggotaKeluargaKK)
# ==============================================================================

@app.route("/kk/<int:id_kk>", methods=["GET"])
def detail_kk(id_kk: int):
    kk = db.get_kk_by_id(id_kk)
    if not kk:
        flash("Data Kartu Keluarga tidak ditemukan.", "warning")
        return redirect(url_for("list_kk"))
    anggota_list = db.get_anggota_by_kk(id_kk)
    return render_template("detail_kk.html", kk=kk, anggota_list=anggota_list)


@app.route("/kk/<int:id_kk>/anggota/add", methods=["POST"])
def add_anggota(id_kk: int):
    data = {k: request.form.get(k, "").strip() for k in request.form}
    try:
        db.add_anggota_keluarga(id_kk, data)
        flash(f"Anggota keluarga '{data.get('nama_lengkap')}' berhasil ditambahkan.", "success")
    except Exception as exc:
        flash(f"Gagal menambahkan anggota: {exc}", "danger")
    return redirect(url_for("detail_kk", id_kk=id_kk))


@app.route("/anggota/<int:id_anggota>/update", methods=["POST"])
def update_anggota(id_anggota: int):
    existing = db.get_anggota_by_id(id_anggota)
    if not existing:
        flash("Data anggota keluarga tidak ditemukan.", "warning")
        return redirect(url_for("list_kk"))

    data = {k: request.form.get(k, "").strip() for k in request.form}
    try:
        db.update_anggota_keluarga(id_anggota, data)
        flash(f"Data anggota keluarga '{data.get('nama_lengkap')}' berhasil diperbarui.", "success")
    except Exception as exc:
        flash(f"Gagal memperbarui anggota: {exc}", "danger")
    return redirect(url_for("detail_kk", id_kk=existing["id_kk"]))


@app.route("/anggota/<int:id_anggota>/delete", methods=["POST"])
def delete_anggota(id_anggota: int):
    existing = db.get_anggota_by_id(id_anggota)
    if not existing:
        return redirect(url_for("list_kk"))
    db.delete_anggota_keluarga(id_anggota)
    flash(f"Anggota keluarga '{existing['nama_lengkap']}' berhasil dihapus.", "info")
    return redirect(url_for("detail_kk", id_kk=existing["id_kk"]))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

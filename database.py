import os
import sqlite3
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kk_database.db")


def get_db_connection() -> sqlite3.Connection:
    """Membuka koneksi ke database SQLite dengan Row factory dan Foreign Keys aktif."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Inisialisasi tabel tbl_identitasKK dan tbl_anggotaKeluargaKK."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Tabel Identitas Kartu Keluarga (tbl_identitasKK)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tbl_identitasKK (
            id_kk INTEGER PRIMARY KEY AUTOINCREMENT,
            no_kk TEXT NOT NULL UNIQUE,
            nama_kepala_keluarga TEXT NOT NULL,
            alamat TEXT DEFAULT '',
            rt_rw TEXT DEFAULT '',
            kode_pos TEXT DEFAULT '',
            desa_kelurahan TEXT DEFAULT '',
            kecamatan TEXT DEFAULT '',
            kabupaten_kota TEXT DEFAULT '',
            provinsi TEXT DEFAULT '',
            tanggal_dikeluarkan TEXT DEFAULT '',
            image_path TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # 2. Tabel Anggota Keluarga pada KK (tbl_anggotaKeluargaKK)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tbl_anggotaKeluargaKK (
            id_anggota INTEGER PRIMARY KEY AUTOINCREMENT,
            id_kk INTEGER NOT NULL,
            no_urut INTEGER DEFAULT 1,
            nama_lengkap TEXT NOT NULL,
            nik TEXT NOT NULL,
            jenis_kelamin TEXT DEFAULT '',
            tempat_lahir TEXT DEFAULT '',
            tanggal_lahir TEXT DEFAULT '',
            agama TEXT DEFAULT '',
            pendidikan TEXT DEFAULT '',
            jenis_pekerjaan TEXT DEFAULT '',
            golongan_darah TEXT DEFAULT '-',
            status_perkawinan TEXT DEFAULT '',
            tanggal_perkawinan TEXT DEFAULT '-',
            status_hubungan TEXT DEFAULT '',
            kewarganegaraan TEXT DEFAULT 'WNI',
            no_paspor TEXT DEFAULT '-',
            no_kitap TEXT DEFAULT '-',
            nama_ayah TEXT DEFAULT '',
            nama_ibu TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_kk) REFERENCES tbl_identitasKK (id_kk) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# ==============================================================================
# OPERASI CRUD UNTUK tbl_identitasKK
# ==============================================================================

def create_kk_with_anggota(
    identitas: Dict[str, Any], anggota_list: List[Dict[str, Any]]
) -> int:
    """
    Menyimpan data identitas KK ke tbl_identitasKK beserta seluruh daftar
    anggota keluarga ke tbl_anggotaKeluargaKK dalam satu transaksi.
    Jika no_kk sudah ada, akan melakukan update pada identitas KK dan mengganti daftar anggota.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    no_kk = (identitas.get("no_kk") or "").strip()
    if not no_kk:
        raise ValueError("Nomor KK (no_kk) wajib diisi.")

    try:
        # Cek apakah no_kk sudah terdaftar
        existing = cursor.execute(
            "SELECT id_kk FROM tbl_identitasKK WHERE no_kk = ?", (no_kk,)
        ).fetchone()

        if existing:
            id_kk = existing["id_kk"]
            cursor.execute(
                """
                UPDATE tbl_identitasKK
                SET nama_kepala_keluarga = ?,
                    alamat = ?,
                    rt_rw = ?,
                    kode_pos = ?,
                    desa_kelurahan = ?,
                    kecamatan = ?,
                    kabupaten_kota = ?,
                    provinsi = ?,
                    tanggal_dikeluarkan = ?,
                    image_path = COALESCE(NULLIF(?, ''), image_path),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id_kk = ?
                """,
                (
                    (identitas.get("nama_kepala_keluarga") or "").strip(),
                    (identitas.get("alamat") or "").strip(),
                    (identitas.get("rt_rw") or "").strip(),
                    (identitas.get("kode_pos") or "").strip(),
                    (identitas.get("desa_kelurahan") or "").strip(),
                    (identitas.get("kecamatan") or "").strip(),
                    (identitas.get("kabupaten_kota") or "").strip(),
                    (identitas.get("provinsi") or "").strip(),
                    (identitas.get("tanggal_dikeluarkan") or "").strip(),
                    (identitas.get("image_path") or "").strip(),
                    id_kk,
                ),
            )
            cursor.execute("DELETE FROM tbl_anggotaKeluargaKK WHERE id_kk = ?", (id_kk,))
        else:
            cursor.execute(
                """
                INSERT INTO tbl_identitasKK (
                    no_kk, nama_kepala_keluarga, alamat, rt_rw, kode_pos,
                    desa_kelurahan, kecamatan, kabupaten_kota, provinsi,
                    tanggal_dikeluarkan, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    no_kk,
                    (identitas.get("nama_kepala_keluarga") or "").strip(),
                    (identitas.get("alamat") or "").strip(),
                    (identitas.get("rt_rw") or "").strip(),
                    (identitas.get("kode_pos") or "").strip(),
                    (identitas.get("desa_kelurahan") or "").strip(),
                    (identitas.get("kecamatan") or "").strip(),
                    (identitas.get("kabupaten_kota") or "").strip(),
                    (identitas.get("provinsi") or "").strip(),
                    (identitas.get("tanggal_dikeluarkan") or "").strip(),
                    (identitas.get("image_path") or "").strip(),
                ),
            )
            id_kk = cursor.lastrowid

        for idx, anggota in enumerate(anggota_list, start=1):
            nama = (anggota.get("nama_lengkap") or "").strip()
            nik = (anggota.get("nik") or "").strip()
            if not nama and not nik:
                continue
            cursor.execute(
                """
                INSERT INTO tbl_anggotaKeluargaKK (
                    id_kk, no_urut, nama_lengkap, nik, jenis_kelamin,
                    tempat_lahir, tanggal_lahir, agama, pendidikan,
                    jenis_pekerjaan, golongan_darah, status_perkawinan,
                    tanggal_perkawinan, status_hubungan, kewarganegaraan,
                    no_paspor, no_kitap, nama_ayah, nama_ibu
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id_kk,
                    int(anggota.get("no_urut") or idx),
                    nama or "-",
                    nik or "-",
                    (anggota.get("jenis_kelamin") or "").strip(),
                    (anggota.get("tempat_lahir") or "").strip(),
                    (anggota.get("tanggal_lahir") or "").strip(),
                    (anggota.get("agama") or "").strip(),
                    (anggota.get("pendidikan") or "").strip(),
                    (anggota.get("jenis_pekerjaan") or "").strip(),
                    (anggota.get("golongan_darah") or "-").strip() or "-",
                    (anggota.get("status_perkawinan") or "").strip(),
                    (anggota.get("tanggal_perkawinan") or "-").strip() or "-",
                    (anggota.get("status_hubungan") or "").strip(),
                    (anggota.get("kewarganegaraan") or "WNI").strip() or "WNI",
                    (anggota.get("no_paspor") or "-").strip() or "-",
                    (anggota.get("no_kitap") or "-").strip() or "-",
                    (anggota.get("nama_ayah") or "").strip(),
                    (anggota.get("nama_ibu") or "").strip(),
                ),
            )

        conn.commit()
        return id_kk
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_all_kk(search_query: str = "") -> List[Dict[str, Any]]:
    """
    Mengambil seluruh daftar KK dari tbl_identitasKK beserta jumlah anggota keluarganya.
    Mendukung pencarian berdasarkan No KK, Nama Kepala Keluarga, Alamat, Kelurahan, Kecamatan, atau Kota.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    base_sql = """
        SELECT
            k.*,
            COUNT(a.id_anggota) AS jumlah_anggota
        FROM tbl_identitasKK k
        LEFT JOIN tbl_anggotaKeluargaKK a ON k.id_kk = a.id_kk
    """
    params: List[Any] = []
    if search_query and search_query.strip():
        q = f"%{search_query.strip()}%"
        base_sql += """
            WHERE k.no_kk LIKE ?
               OR k.nama_kepala_keluarga LIKE ?
               OR k.alamat LIKE ?
               OR k.desa_kelurahan LIKE ?
               OR k.kecamatan LIKE ?
               OR k.kabupaten_kota LIKE ?
               OR k.provinsi LIKE ?
        """
        params.extend([q, q, q, q, q, q, q])

    base_sql += " GROUP BY k.id_kk ORDER BY k.updated_at DESC, k.id_kk DESC"

    rows = cursor.execute(base_sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kk_by_id(id_kk: int) -> Optional[Dict[str, Any]]:
    """Mengambil satu record tbl_identitasKK berdasarkan id_kk."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM tbl_identitasKK WHERE id_kk = ?", (id_kk,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_identitas_kk(id_kk: int, data: Dict[str, Any]) -> bool:
    """Memperbarui data pada tbl_identitasKK."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE tbl_identitasKK
        SET no_kk = ?,
            nama_kepala_keluarga = ?,
            alamat = ?,
            rt_rw = ?,
            kode_pos = ?,
            desa_kelurahan = ?,
            kecamatan = ?,
            kabupaten_kota = ?,
            provinsi = ?,
            tanggal_dikeluarkan = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id_kk = ?
        """,
        (
            (data.get("no_kk") or "").strip(),
            (data.get("nama_kepala_keluarga") or "").strip(),
            (data.get("alamat") or "").strip(),
            (data.get("rt_rw") or "").strip(),
            (data.get("kode_pos") or "").strip(),
            (data.get("desa_kelurahan") or "").strip(),
            (data.get("kecamatan") or "").strip(),
            (data.get("kabupaten_kota") or "").strip(),
            (data.get("provinsi") or "").strip(),
            (data.get("tanggal_dikeluarkan") or "").strip(),
            id_kk,
        ),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_identitas_kk(id_kk: int) -> bool:
    """Menghapus KK dari tbl_identitasKK (otomatis menghapus anggota keluarga via ON DELETE CASCADE)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tbl_identitasKK WHERE id_kk = ?", (id_kk,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ==============================================================================
# OPERASI CRUD UNTUK tbl_anggotaKeluargaKK
# ==============================================================================

def get_anggota_by_kk(id_kk: int) -> List[Dict[str, Any]]:
    """Mengambil seluruh anggota keluarga pada sebuah KK berdasarkan id_kk."""
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM tbl_anggotaKeluargaKK
        WHERE id_kk = ?
        ORDER BY no_urut ASC, id_anggota ASC
        """,
        (id_kk,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_anggota_by_id(id_anggota: int) -> Optional[Dict[str, Any]]:
    """Mengambil satu data anggota keluarga berdasarkan id_anggota."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM tbl_anggotaKeluargaKK WHERE id_anggota = ?", (id_anggota,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def add_anggota_keluarga(id_kk: int, data: Dict[str, Any]) -> int:
    """Menambahkan satu anggota keluarga baru ke tbl_anggotaKeluargaKK."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tentukan no_urut berikutnya jika tidak diisi
    no_urut = data.get("no_urut")
    if not no_urut:
        max_row = cursor.execute(
            "SELECT COALESCE(MAX(no_urut), 0) AS max_urut FROM tbl_anggotaKeluargaKK WHERE id_kk = ?",
            (id_kk,),
        ).fetchone()
        no_urut = (max_row["max_urut"] if max_row else 0) + 1

    cursor.execute(
        """
        INSERT INTO tbl_anggotaKeluargaKK (
            id_kk, no_urut, nama_lengkap, nik, jenis_kelamin,
            tempat_lahir, tanggal_lahir, agama, pendidikan,
            jenis_pekerjaan, golongan_darah, status_perkawinan,
            tanggal_perkawinan, status_hubungan, kewarganegaraan,
            no_paspor, no_kitap, nama_ayah, nama_ibu
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            id_kk,
            int(no_urut),
            (data.get("nama_lengkap") or "").strip(),
            (data.get("nik") or "").strip(),
            (data.get("jenis_kelamin") or "").strip(),
            (data.get("tempat_lahir") or "").strip(),
            (data.get("tanggal_lahir") or "").strip(),
            (data.get("agama") or "").strip(),
            (data.get("pendidikan") or "").strip(),
            (data.get("jenis_pekerjaan") or "").strip(),
            (data.get("golongan_darah") or "-").strip() or "-",
            (data.get("status_perkawinan") or "").strip(),
            (data.get("tanggal_perkawinan") or "-").strip() or "-",
            (data.get("status_hubungan") or "").strip(),
            (data.get("kewarganegaraan") or "WNI").strip() or "WNI",
            (data.get("no_paspor") or "-").strip() or "-",
            (data.get("no_kitap") or "-").strip() or "-",
            (data.get("nama_ayah") or "").strip(),
            (data.get("nama_ibu") or "").strip(),
        ),
    )
    new_id = cursor.lastrowid
    cursor.execute(
        "UPDATE tbl_identitasKK SET updated_at = CURRENT_TIMESTAMP WHERE id_kk = ?",
        (id_kk,),
    )
    conn.commit()
    conn.close()
    return new_id


def update_anggota_keluarga(id_anggota: int, data: Dict[str, Any]) -> bool:
    """Memperbarui satu record anggota keluarga pada tbl_anggotaKeluargaKK."""
    conn = get_db_connection()
    cursor = conn.cursor()

    existing = cursor.execute(
        "SELECT id_kk FROM tbl_anggotaKeluargaKK WHERE id_anggota = ?", (id_anggota,)
    ).fetchone()
    if not existing:
        conn.close()
        return False

    cursor.execute(
        """
        UPDATE tbl_anggotaKeluargaKK
        SET no_urut = ?,
            nama_lengkap = ?,
            nik = ?,
            jenis_kelamin = ?,
            tempat_lahir = ?,
            tanggal_lahir = ?,
            agama = ?,
            pendidikan = ?,
            jenis_pekerjaan = ?,
            golongan_darah = ?,
            status_perkawinan = ?,
            tanggal_perkawinan = ?,
            status_hubungan = ?,
            kewarganegaraan = ?,
            no_paspor = ?,
            no_kitap = ?,
            nama_ayah = ?,
            nama_ibu = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id_anggota = ?
        """,
        (
            int(data.get("no_urut") or 1),
            (data.get("nama_lengkap") or "").strip(),
            (data.get("nik") or "").strip(),
            (data.get("jenis_kelamin") or "").strip(),
            (data.get("tempat_lahir") or "").strip(),
            (data.get("tanggal_lahir") or "").strip(),
            (data.get("agama") or "").strip(),
            (data.get("pendidikan") or "").strip(),
            (data.get("jenis_pekerjaan") or "").strip(),
            (data.get("golongan_darah") or "-").strip() or "-",
            (data.get("status_perkawinan") or "").strip(),
            (data.get("tanggal_perkawinan") or "-").strip() or "-",
            (data.get("status_hubungan") or "").strip(),
            (data.get("kewarganegaraan") or "WNI").strip() or "WNI",
            (data.get("no_paspor") or "-").strip() or "-",
            (data.get("no_kitap") or "-").strip() or "-",
            (data.get("nama_ayah") or "").strip(),
            (data.get("nama_ibu") or "").strip(),
            id_anggota,
        ),
    )
    updated = cursor.rowcount > 0
    cursor.execute(
        "UPDATE tbl_identitasKK SET updated_at = CURRENT_TIMESTAMP WHERE id_kk = ?",
        (existing["id_kk"],),
    )
    conn.commit()
    conn.close()
    return updated


def delete_anggota_keluarga(id_anggota: int) -> bool:
    """Menghapus satu record anggota keluarga dari tbl_anggotaKeluargaKK."""
    conn = get_db_connection()
    cursor = conn.cursor()
    existing = cursor.execute(
        "SELECT id_kk FROM tbl_anggotaKeluargaKK WHERE id_anggota = ?", (id_anggota,)
    ).fetchone()
    cursor.execute(
        "DELETE FROM tbl_anggotaKeluargaKK WHERE id_anggota = ?", (id_anggota,)
    )
    deleted = cursor.rowcount > 0
    if existing:
        cursor.execute(
            "UPDATE tbl_identitasKK SET updated_at = CURRENT_TIMESTAMP WHERE id_kk = ?",
            (existing["id_kk"],),
        )
    conn.commit()
    conn.close()
    return deleted

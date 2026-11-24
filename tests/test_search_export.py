import csv
import io
import os
import tempfile
import unittest

import database as db
from sample_kk_generator import SAMPLE_KK_DATASETS


class TestDatabaseSearchAndFilter(unittest.TestCase):
    """Pengujian pencarian NIK/Nama anggota dan filter wilayah pada database.py."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmp_dir.name, "test_kk.db")
        db.init_db()

        self.id_bogor = db.create_kk_with_anggota(
            SAMPLE_KK_DATASETS[0]["identitas"],
            SAMPLE_KK_DATASETS[0]["anggota"],
        )
        self.id_jakarta = db.create_kk_with_anggota(
            SAMPLE_KK_DATASETS[1]["identitas"],
            SAMPLE_KK_DATASETS[1]["anggota"],
        )
        self.id_kosong = db.create_kk_with_anggota(
            {
                "no_kk": "3578010101220009",
                "nama_kepala_keluarga": "AGUS SALIM",
                "alamat": "JL. DARMO NO. 10",
                "rt_rw": "001/002",
                "kode_pos": "60264",
                "desa_kelurahan": "DARMO",
                "kecamatan": "WONOKROMO",
                "kabupaten_kota": "KOTA SURABAYA",
                "provinsi": "JAWA TIMUR",
                "tanggal_dikeluarkan": "01-01-2024",
            },
            [],
        )

    def tearDown(self):
        db.DB_PATH = self.orig_db_path
        self.tmp_dir.cleanup()

    def test_search_by_member_name_preserves_full_member_count(self):
        results = db.get_all_kk(search_query="DEWI LESTARI")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["no_kk"], "3271041508200001")
        # Jumlah anggota harus tetap 4 (seluruh anggota KK), bukan 1
        self.assertEqual(results[0]["jumlah_anggota"], 4)

    def test_search_by_member_nik_preserves_full_member_count(self):
        # NIK anak ke-4 (NADIA PUTRI SANTOSO)
        results = db.get_all_kk(search_query="3271046111120004")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["no_kk"], "3271041508200001")
        self.assertEqual(results[0]["jumlah_anggota"], 4)

    def test_filter_by_provinsi(self):
        results = db.get_all_kk(provinsi="dki jakarta")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["no_kk"], "3174082209190005")

    def test_filter_by_kabupaten_kota(self):
        results = db.get_all_kk(kabupaten_kota="KOTA BOGOR")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["no_kk"], "3271041508200001")

    def test_combined_search_and_region_filter(self):
        matched = db.get_all_kk(search_query="KEVIN ADITYA", provinsi="DKI JAKARTA")
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["no_kk"], "3174082209190005")

        unmatched = db.get_all_kk(search_query="KEVIN ADITYA", provinsi="JAWA BARAT")
        self.assertEqual(unmatched, [])

    def test_get_kk_filter_options_returns_sorted_unique_regions(self):
        options = db.get_kk_filter_options()
        self.assertEqual(options["provinsi"], ["DKI JAKARTA", "JAWA BARAT", "JAWA TIMUR"])
        self.assertEqual(
            options["kabupaten_kota"],
            ["JAKARTA SELATAN", "KOTA BOGOR", "KOTA SURABAYA"],
        )

    def test_get_all_kk_with_anggota_includes_nested_members(self):
        rows = db.get_all_kk_with_anggota(provinsi="JAWA BARAT")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["no_kk"], "3271041508200001")
        self.assertEqual(len(rows[0]["anggota"]), 4)
        self.assertEqual(rows[0]["anggota"][0]["nama_lengkap"], "BUDI SANTOSO")
        self.assertEqual(rows[0]["anggota"][3]["nama_lengkap"], "NADIA PUTRI SANTOSO")


class TestSearchExportAndPrintRoutes(unittest.TestCase):
    """Pengujian rute pencarian, filter wilayah, ekspor CSV, dan tombol cetak pada detail KK."""

    @classmethod
    def setUpClass(cls):
        import app as flask_app_module

        cls.app = flask_app_module.app
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.tmp_dir.name, "test_routes.db")
        db.init_db()

        self.id_bogor = db.create_kk_with_anggota(
            SAMPLE_KK_DATASETS[0]["identitas"],
            SAMPLE_KK_DATASETS[0]["anggota"],
        )
        self.id_jakarta = db.create_kk_with_anggota(
            SAMPLE_KK_DATASETS[1]["identitas"],
            SAMPLE_KK_DATASETS[1]["anggota"],
        )
        self.id_kosong = db.create_kk_with_anggota(
            {
                "no_kk": "3578010101220009",
                "nama_kepala_keluarga": "AGUS SALIM",
                "alamat": "JL. DARMO NO. 10",
                "rt_rw": "001/002",
                "kode_pos": "60264",
                "desa_kelurahan": "DARMO",
                "kecamatan": "WONOKROMO",
                "kabupaten_kota": "KOTA SURABAYA",
                "provinsi": "JAWA TIMUR",
                "tanggal_dikeluarkan": "01-01-2024",
            },
            [],
        )
        self.client = self.app.test_client()

    def tearDown(self):
        db.DB_PATH = self.orig_db_path
        self.tmp_dir.cleanup()

    def test_list_kk_filters_by_member_nik_and_updates_filtered_member_kpi(self):
        resp = self.client.get("/kk?q=3271046111120004&provinsi=JAWA+BARAT")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("BUDI SANTOSO", html)
        self.assertNotIn("IRWAN SETIAWAN", html)
        self.assertIn("4 Jiwa", html)

    def test_export_all_kk_csv_identitas_with_bom_and_filter(self):
        resp = self.client.get("/kk/export?scope=identitas&provinsi=JAWA+BARAT")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.content_type)
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))
        self.assertIn("daftar_kk.csv", resp.headers.get("Content-Disposition", ""))

        raw = resp.get_data()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "CSV harus memakai UTF-8 BOM untuk Excel")

        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["No. KK"], '="3271041508200001"')
        self.assertEqual(rows[0]["Nama Kepala Keluarga"], "BUDI SANTOSO")
        self.assertEqual(rows[0]["Jumlah Anggota"], "4")

    def test_export_all_kk_csv_lengkap_includes_members_and_empty_kk(self):
        resp = self.client.get("/kk/export?scope=lengkap")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("data_kk_lengkap.csv", resp.headers.get("Content-Disposition", ""))

        raw = resp.get_data()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))

        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        # 4 anggota Bogor + 3 anggota Jakarta + 1 baris KK tanpa anggota (Surabaya) = 8 baris
        self.assertEqual(len(rows), 8)
        nama_anggota = [r["Nama Lengkap Anggota"] for r in rows]
        self.assertIn("DEWI LESTARI", nama_anggota)
        self.assertIn("KEVIN ADITYA SETIAWAN", nama_anggota)
        kepala_names = [r["Nama Kepala Keluarga"] for r in rows]
        self.assertIn("AGUS SALIM", kepala_names)

    def test_export_single_kk_csv(self):
        resp = self.client.get(f"/kk/{self.id_bogor}/export")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("kk_3271041508200001.csv", resp.headers.get("Content-Disposition", ""))

        raw = resp.get_data()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        parsed_rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        kv = {r[0]: r[1] for r in parsed_rows[:11] if len(r) >= 2}
        self.assertEqual(kv["No. KK"], '="3271041508200001"')
        self.assertEqual(kv["Nama Kepala Keluarga"], "BUDI SANTOSO")

        member_rows = parsed_rows[13:]
        self.assertEqual(len(member_rows), 4)
        self.assertEqual(member_rows[3][1], "NADIA PUTRI SANTOSO")
        self.assertEqual(member_rows[3][2], '="3271046111120004"')

    def test_export_csv_neutralizes_formula_injection(self):
        id_formula = db.create_kk_with_anggota(
            {
                "no_kk": "3271041508209999",
                "nama_kepala_keluarga": "=CMD('calc')",
                "alamat": "+HYPERLINK(\"http://evil\")",
                "provinsi": "JAWA BARAT",
            },
            [],
        )
        resp = self.client.get(f"/kk/{id_formula}/export")
        self.assertEqual(resp.status_code, 200)
        parsed_rows = list(csv.reader(io.StringIO(resp.get_data().decode("utf-8-sig"))))
        kv = {r[0]: r[1] for r in parsed_rows[:11] if len(r) >= 2}
        self.assertEqual(kv["Nama Kepala Keluarga"], "'=CMD('calc')")
        self.assertEqual(kv["Alamat"], "'+HYPERLINK(\"http://evil\")")

    def test_core_pages_remain_accessible(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/kk").status_code, 200)
        self.assertEqual(self.client.get(f"/kk/{self.id_bogor}").status_code, 200)

    def test_export_single_kk_not_found_redirects(self):
        resp = self.client.get("/kk/999999/export")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/kk", resp.headers.get("Location", ""))

    def test_detail_kk_includes_export_and_print_controls(self):
        resp = self.client.get(f"/kk/{self.id_bogor}")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn(f"/kk/{self.id_bogor}/export", html)
        self.assertIn("window.print()", html)
        self.assertIn("@media print", html)


if __name__ == "__main__":
    unittest.main()

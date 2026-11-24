import difflib
import os
import tempfile
import unittest

import cv2
import numpy as np

from image_processor import run_image_processing_pipeline
from sample_kk_generator import SAMPLE_KK_DATASETS, render_kk_image


class TestImageProcessorRobustness(unittest.TestCase):
    """Pengujian presisi OCR pada citra KK bersih maupun foto simulasi kabur, redup, dan miring."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.clean_path = os.path.join(self.tmp_dir.name, "clean_kk.png")
        render_kk_image(SAMPLE_KK_DATASETS[0], self.clean_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_clean_kk_extracts_all_identitas_and_members(self):
        res = run_image_processing_pipeline(self.clean_path)
        ident = res["identitas"]
        self.assertEqual(ident["no_kk"], "3271041508200001")
        self.assertEqual(ident["nama_kepala_keluarga"], "BUDI SANTOSO")
        self.assertEqual(ident["kabupaten_kota"], "KOTA BOGOR")
        self.assertEqual(ident["provinsi"], "JAWA BARAT")
        self.assertEqual(ident["tanggal_dikeluarkan"], "14-02-2023")

        anggota = res["anggota"]
        self.assertEqual(len(anggota), 4)
        self.assertEqual(anggota[0]["nik"], "3271041205800001")
        self.assertEqual(anggota[3]["nama_lengkap"], "NADIA PUTRI SANTOSO")
        self.assertEqual(anggota[3]["nik"], "3271046111120004")

    def test_skewed_dim_blurry_kk_photo_is_rectified_and_extracted(self):
        # Simulasikan foto HP: latar meja gelap, miring 14 derajat, pencahayaan redup, dan sedikit kabur
        src = cv2.imread(self.clean_path)
        sh, sw = src.shape[:2]
        # Perkecil sedikit seperti foto kamera HP lalu letakkan di atas latar gelap
        small = cv2.resize(src, (1100, int(1100 * sh / sw)), interpolation=cv2.INTER_AREA)
        canvas = np.full((1150, 1450, 3), (42, 34, 28), dtype=np.uint8)
        y0, x0 = 180, 175
        canvas[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small

        # Rotasi 14 derajat + redupkan 30% + Gaussian blur (simulasi fokus kamera agak kabur)
        M = cv2.getRotationMatrix2D((canvas.shape[1] / 2.0, canvas.shape[0] / 2.0), 14.0, 0.92)
        warped = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=(42, 34, 28))
        dim_blur = cv2.GaussianBlur((warped.astype(np.float32) * 0.72).astype(np.uint8), (3, 3), 0.9)

        blurry_path = os.path.join(self.tmp_dir.name, "blurry_skewed_kk.jpg")
        cv2.imwrite(blurry_path, dim_blur, [cv2.IMWRITE_JPEG_QUALITY, 82])

        res = run_image_processing_pipeline(blurry_path)
        ident = res["identitas"]
        self.assertEqual(ident["no_kk"], "3271041508200001")
        self.assertGreaterEqual(
            difflib.SequenceMatcher(None, ident["nama_kepala_keluarga"], "BUDI SANTOSO").ratio(),
            0.85,
            f"Nama kepala keluarga terlalu jauh: {ident['nama_kepala_keluarga']}",
        )
        self.assertEqual(ident["provinsi"], "JAWA BARAT")
        self.assertEqual(len(res["anggota"]), 4)
        nik_1 = res["anggota"][0]["nik"]
        self.assertEqual(len(nik_1), 16)
        self.assertTrue(nik_1.isdigit())
        self.assertTrue(nik_1.startswith("327104"))
        self.assertGreaterEqual(
            difflib.SequenceMatcher(None, nik_1, "3271041205800001").ratio(),
            0.85,
            f"NIK baris 1 terlalu jauh pada foto kabur: {nik_1}",
        )


if __name__ == "__main__":
    unittest.main()

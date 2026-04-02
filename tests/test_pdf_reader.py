import base64
import tempfile
import unittest
from pathlib import Path

from src.core.pdf_reader import build_inline_pdf_data_url, list_pdf_files, read_pdf_bytes


class TestPdfReader(unittest.TestCase):
    def test_list_pdf_files_returns_sorted_pdf_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "b.pdf").write_bytes(b"b")
            (root / "a.PDF").write_bytes(b"a")
            (root / "notes.txt").write_text("not a pdf")

            pdf_paths = list_pdf_files(root)

            self.assertEqual([p.name for p in pdf_paths], ["a.PDF", "b.pdf"])

    def test_build_inline_pdf_data_url_encodes_binary_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.pdf"
            payload = b"%PDF-1.7 fake content"
            path.write_bytes(payload)

            data_url = build_inline_pdf_data_url(path)

            self.assertTrue(data_url.startswith("data:application/pdf;base64,"))
            encoded = data_url.split(",", maxsplit=1)[1]
            self.assertEqual(base64.b64decode(encoded), payload)

    def test_read_pdf_bytes_returns_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "downloadable.pdf"
            payload = b"%PDF-1.4 bytes for download"
            path.write_bytes(payload)

            self.assertEqual(read_pdf_bytes(path), payload)


if __name__ == "__main__":
    unittest.main()

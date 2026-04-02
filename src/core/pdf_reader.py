from __future__ import annotations

import base64
from pathlib import Path


def list_pdf_files(data_dir: Path) -> list[Path]:
    """Return PDF files in data_dir sorted by filename (case-insensitive)."""
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    return sorted(
        [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )


def build_inline_pdf_data_url(pdf_path: Path) -> str:
    """Build a base64 data URL for embedding a PDF inline."""
    payload = read_pdf_bytes(pdf_path)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def read_pdf_bytes(pdf_path: Path) -> bytes:
    """Read raw PDF bytes for download/export actions."""
    return pdf_path.read_bytes()

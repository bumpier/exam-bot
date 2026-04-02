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


def build_pdf_embed_html(pdf_data_url: str, height_px: int = 700) -> str:
    """Build robust HTML for inline PDF preview with browser fallback."""
    safe_height = max(300, int(height_px))
    return (
        "<object "
        f'data="{pdf_data_url}" '
        'type="application/pdf" '
        'width="100%" '
        f'height="{safe_height}" '
        'style="border: 1px solid #ddd; border-radius: 8px;">'
        "<iframe "
        f'src="{pdf_data_url}" '
        'width="100%" '
        f'height="{safe_height}" '
        'style="border: 1px solid #ddd; border-radius: 8px;">'
        "<p>"
        "This browser could not render the PDF preview. "
        f'<a href="{pdf_data_url}" target="_blank" rel="noopener noreferrer">'
        "Open PDF in a new tab"
        "</a>."
        "</p>"
        "</iframe>"
        "</object>"
    )


def read_pdf_bytes(pdf_path: Path) -> bytes:
    """Read raw PDF bytes for download/export actions."""
    return pdf_path.read_bytes()

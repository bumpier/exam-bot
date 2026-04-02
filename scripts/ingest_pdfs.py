#!/usr/bin/env python3
"""
Ingest PDF compliance documents into ChromaDB.

Usage:
    # Ingest all PDFs in ./data/ with default importance (Medium)
    python scripts/ingest_pdfs.py

    # Set importance for all PDFs in this run
    python scripts/ingest_pdfs.py --importance High

    # Target a specific file
    python scripts/ingest_pdfs.py --file data/aml_handbook_2024__ch3.pdf

    # Force re-ingestion even if source already exists
    python scripts/ingest_pdfs.py --force

Filename convention:
    <source_tag>__<chapter>.pdf   →  source=aml_handbook_2024, chapter=ch3_beneficial_ownership
    <source_tag>.pdf              →  source=aml_handbook_2024, chapter=""

The script skips PDFs whose source_tag is already present in the collection
unless --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import DATA_DIR
from src.db.chroma_client import get_collection, list_unique_values

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Smaller batches reduce Ollama embedding timeouts on large OCR ingests.
BATCH_SIZE = 10  # ChromaDB upsert batch size


def parse_filename(stem: str) -> tuple[str, str]:
    """Extract (source_tag, chapter) from a filename stem.

    'aml_handbook_2024__ch3_beneficial_ownership'  →  ('aml_handbook_2024', 'ch3_beneficial_ownership')
    'fatf_recommendations'                          →  ('fatf_recommendations', '')
    """
    if "__" in stem:
        source_tag, chapter = stem.split("__", 1)
        return source_tag.strip(), chapter.strip()
    return stem.strip(), ""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Return concatenated text from all pages of a PDF.

    Tries multiple extractors:
    - PyMuPDF (fitz) if installed (often better with complex PDFs)
    - pypdf fallback

    Note: If the PDF is scanned (images only), both methods may return empty
    text; OCR would be required.
    """
    # 1) Try PyMuPDF if available
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        pages_text: list[str] = []
        for page in doc:
            t = page.get_text("text")
            if t:
                pages_text.append(t)
        text = "\n\n".join(pages_text)
        if text.strip():
            return text
    except Exception:
        # Any failure falls back to pypdf
        pass

    # 2) Fallback: pypdf
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def ocr_pdf_to_text(pdf_path: Path, language: str = "eng", dpi: int = 200) -> str:
    """OCR a PDF using the system `tesseract` binary and PyMuPDF rendering.

    This is used when PDFs are scanned/image-only, so text extraction returns empty.
    """
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract not found on PATH; cannot OCR scanned PDFs.")

    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is required for OCR rendering.") from exc

    doc = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    page_texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="exam_bot_ocr_") as tmpdir:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_path = Path(tmpdir) / f"page_{i+1:04d}.png"
            pix.save(str(img_path))

            proc = subprocess.run(
                ["tesseract", str(img_path), "stdout", "-l", language],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"tesseract failed on page {i+1} (exit {proc.returncode}). "
                    f"stderr: {proc.stderr[:300]}"
                )
            if proc.stdout:
                page_texts.append(proc.stdout)

    return "\n\n".join(page_texts)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def stable_chunk_id(source_tag: str, chapter: str, chunk_index: int, text: str) -> str:
    """Produce a deterministic ID for a chunk so re-ingestion is idempotent."""
    raw = f"{source_tag}|{chapter}|{chunk_index}|{text[:64]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def ingest_pdf(
    pdf_path: Path,
    importance: str,
    force: bool,
    existing_sources: set[str],
) -> int:
    """Ingest a single PDF. Returns number of chunks added."""
    stem = pdf_path.stem
    source_tag, chapter = parse_filename(stem)

    if source_tag in existing_sources and not force:
        print(f"  [SKIP] {pdf_path.name}  (source '{source_tag}' already ingested; use --force to overwrite)")
        return 0

    print(f"  [READ] {pdf_path.name}  →  source='{source_tag}'  chapter='{chapter}'")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return 0

    chunks = chunk_text(text)
    print(f"         {len(chunks)} chunks created")

    collection = get_collection()

    # Remove existing chunks for this source if force-reinserting
    if source_tag in existing_sources and force:
        print(f"         Removing existing chunks for source='{source_tag}'...")
        collection.delete(where={"source": {"$eq": source_tag}})

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = stable_chunk_id(source_tag, chapter, i, chunk)
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(
            {
                "source": source_tag,
                "chapter": chapter,
                "importance": importance,
                "chunk_index": i,
            }
        )

    # Upsert in batches to avoid memory issues with large PDFs
    total_added = 0
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        total_added += len(ids[start:end])

    print(f"         ✓ {total_added} chunks stored  (importance={importance})")
    return total_added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF compliance documents into ChromaDB."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Ingest a specific PDF file instead of scanning ./data/.",
    )
    parser.add_argument(
        "--importance",
        choices=["High", "Medium", "Low"],
        default="Medium",
        help="Importance tag to assign to all chunks (default: Medium).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the source tag already exists in the DB.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="If text extraction fails, attempt OCR using tesseract (slower).",
    )
    parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code (default: eng).",
    )
    args = parser.parse_args()

    existing_sources = set(list_unique_values("source"))

    if args.file:
        pdf_files = [args.file.resolve()]
        if not pdf_files[0].exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
    else:
        if not DATA_DIR.exists():
            print(f"Error: data directory not found: {DATA_DIR}")
            print("Create the ./data/ directory and place your PDF files inside it.")
            sys.exit(1)
        pdf_files = sorted(DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {DATA_DIR}")
            print("Drop .pdf files into ./data/ and re-run this script.")
            sys.exit(0)

    print(f"\nFound {len(pdf_files)} PDF file(s) to process.\n")

    total_chunks = 0
    for pdf_path in pdf_files:
        added = ingest_pdf(
            pdf_path=pdf_path,
            importance=args.importance,
            force=args.force,
            existing_sources=existing_sources,
        )
        if added == 0:
            # If no text was extractable, optionally attempt OCR.
            text = extract_text_from_pdf(pdf_path)
            if (not text.strip()) and args.ocr:
                print(f"  [OCR ] {pdf_path.name}  (attempting OCR via tesseract; this can take a while)")
                try:
                    ocr_text = ocr_pdf_to_text(
                        pdf_path=pdf_path, language=args.ocr_language
                    )
                except Exception as exc:
                    print(
                        f"  [WARN] OCR failed for {pdf_path.name} — skipping.\n"
                        f"         {exc}"
                    )
                    continue

                if not ocr_text.strip():
                    print(
                        f"  [WARN] OCR produced no text for {pdf_path.name} — skipping."
                    )
                    continue

                # Re-run ingestion path using OCR text directly.
                stem = pdf_path.stem
                source_tag, chapter = parse_filename(stem)
                chunks = chunk_text(ocr_text)
                print(f"         {len(chunks)} OCR chunks created")

                collection = get_collection()
                if source_tag in existing_sources and args.force:
                    print(f"         Removing existing chunks for source='{source_tag}'...")
                    collection.delete(where={"source": {"$eq": source_tag}})

                ids: list[str] = []
                documents: list[str] = []
                metadatas: list[dict] = []
                for i, chunk in enumerate(chunks):
                    chunk_id = stable_chunk_id(source_tag, chapter, i, chunk)
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "source": source_tag,
                            "chapter": chapter,
                            "importance": args.importance,
                            "chunk_index": i,
                            "ocr": True,
                        }
                    )
                for start in range(0, len(ids), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    collection.upsert(
                        ids=ids[start:end],
                        documents=documents[start:end],
                        metadatas=metadatas[start:end],
                    )
                added = len(ids)
                print(f"         ✓ {added} OCR chunks stored  (importance={args.importance})")
                existing_sources.add(source_tag)

        total_chunks += added

    print(f"\nDone. {total_chunks:,} new chunks added to the database.")
    print("Run  python scripts/inspect_db.py  to verify the results.")
    print("Then update books_config.json with the source_tag values shown above.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inspect the ChromaDB vector database and print a summary of its contents.

Usage:
    python scripts/inspect_db.py
    python scripts/inspect_db.py --verbose   # also prints first 200 chars of each chunk

This script is safe to run at any time — it only reads from the database.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHROMA_COLLECTION, CHROMA_DB_PATH
from src.db.chroma_client import (
    collection_size,
    get_all_metadata,
    get_collection,
    list_unique_values,
)

DIVIDER = "─" * 60


def print_section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main(verbose: bool = False) -> None:
    print_section("ChromaDB Inspection Report")
    print(f"  DB path   : {CHROMA_DB_PATH}")
    print(f"  Collection: {CHROMA_COLLECTION}")

    size = collection_size()
    print(f"  Total chunks: {size:,}")

    if size == 0:
        print(
            "\n  [!] The collection is empty.\n"
            "      Drop PDF files into ./data/ and run:\n"
            "          python scripts/ingest_pdfs.py\n"
        )
        return

    # -----------------------------------------------------------------------
    # Unique sources
    # -----------------------------------------------------------------------
    print_section("Unique Sources  (use these as source_tag in books_config.json)")
    sources = list_unique_values("source")
    if sources:
        for s in sources:
            print(f"  • {s}")
    else:
        print("  (none found — metadata may not include a 'source' field)")

    # -----------------------------------------------------------------------
    # Unique chapters
    # -----------------------------------------------------------------------
    print_section("Unique Chapters")
    chapters = list_unique_values("chapter")
    if chapters:
        for c in chapters:
            print(f"  • {c}")
    else:
        print("  (none found)")

    # -----------------------------------------------------------------------
    # Importance distribution
    # -----------------------------------------------------------------------
    print_section("Importance Levels")
    importance_levels = list_unique_values("importance")
    if importance_levels:
        metadatas = get_all_metadata()
        counts: dict[str, int] = defaultdict(int)
        for m in metadatas:
            counts[m.get("importance", "Unset")] += 1
        for level in sorted(counts):
            print(f"  {level:10s}: {counts[level]:,} chunks")
    else:
        print("  (none found — all chunks may use the default importance)")

    # -----------------------------------------------------------------------
    # Per-source chunk counts
    # -----------------------------------------------------------------------
    print_section("Chunks per Source")
    metadatas = get_all_metadata()
    source_counts: dict[str, int] = defaultdict(int)
    for m in metadatas:
        source_counts[m.get("source", "unknown")] += 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:5,}  {src}")

    # -----------------------------------------------------------------------
    # Verbose: sample chunks
    # -----------------------------------------------------------------------
    if verbose:
        print_section("Sample Chunks (first 3 per source)")
        collection = get_collection()
        for source in sources:
            print(f"\n  [{source}]")
            result = collection.get(
                where={"source": {"$eq": source}},
                limit=3,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(result["documents"], result["metadatas"]):
                chapter = meta.get("chapter", "—")
                importance = meta.get("importance", "—")
                preview = doc[:200].replace("\n", " ")
                print(f"    chapter={chapter}  importance={importance}")
                print(f"    {preview}...")
                print()

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB compliance library.")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print sample document chunks for each source.",
    )
    args = parser.parse_args()
    main(verbose=args.verbose)

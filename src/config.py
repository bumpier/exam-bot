"""
Central configuration module.

Loads settings from environment variables (with .env file support) and
provides typed accessors. Also owns the books_config.json loader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db")))
BOOKS_CONFIG_PATH = PROJECT_ROOT / "books_config.json"

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------

# Use a distinct default collection name so we can change embedding backends
# without conflicting with older persisted collections.
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "compliance_books_ollama")

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
OLLAMA_STOP: list[str] = [
    token.strip() for token in os.getenv("OLLAMA_STOP", "```").split(",") if token.strip()
]

# ---------------------------------------------------------------------------
# LLM provider settings
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# OpenAI / ChatGPT settings (used when LLM_PROVIDER=openai)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "700"))
OPENAI_TIMEOUT_SECONDS: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "25"))

# ---------------------------------------------------------------------------
# App login settings
# ---------------------------------------------------------------------------

APP_LOGIN_USERNAME: str = os.getenv("APP_LOGIN_USERNAME", "admin")
APP_LOGIN_PASSWORD: str = os.getenv("APP_LOGIN_PASSWORD", "change-me")

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------

# Default to local Ollama embeddings to avoid network downloads.
# Default to a dedicated embedding model (fast + local).
# Ensure it's available via Ollama:
#   curl http://localhost:11434/api/pull -d '{"name":"nomic-embed-text"}'
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Quiz settings
# ---------------------------------------------------------------------------

DEFAULT_QUESTION_COUNT: int = 10
MIN_QUESTION_COUNT: int = 5
MAX_QUESTION_COUNT: int = 20

# ---------------------------------------------------------------------------
# Books config loader
# ---------------------------------------------------------------------------


def load_books_config() -> list[dict[str, Any]]:
    """Return the list of book entries from books_config.json.

    Each entry has at minimum: display_name, source_tag, description.
    Returns an empty list if the config file is missing or malformed.
    """
    if not BOOKS_CONFIG_PATH.exists():
        return []
    try:
        with BOOKS_CONFIG_PATH.open() as fh:
            data = json.load(fh)
        return data.get("books", [])
    except (json.JSONDecodeError, KeyError):
        return []


def get_source_tag(display_name: str) -> str | None:
    """Look up the ChromaDB source_tag for a given display name."""
    for book in load_books_config():
        if book.get("display_name") == display_name:
            return book.get("source_tag")
    return None

"""
ChromaDB client utilities.

Provides a singleton collection handle and helpers for metadata-filtered
retrieval used by the MCQ generator and the inspect script.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
)


# ---------------------------------------------------------------------------
# Client + collection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    """Return (and cache) a persistent ChromaDB client."""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


@lru_cache(maxsize=1)
def get_embedding_function() -> Any:
    """Return (and cache) the embedding function.

    Supported EMBEDDING_MODEL values:
    - "ollama:<model>"        -> uses local Ollama embeddings
    - "<sentence-transformers model name>" -> uses sentence-transformers (may download)
    """
    if EMBEDDING_MODEL.startswith("ollama:"):
        model = EMBEDDING_MODEL.split("ollama:", 1)[1].strip()
        if not model:
            raise ValueError("EMBEDDING_MODEL is 'ollama:' but no model name provided.")
        # Larger timeout because embedding large batches can be slow.
        return OllamaEmbeddingFunction(url=OLLAMA_BASE_URL, model_name=model, timeout=300)

    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_collection(include_embedding: bool = True) -> chromadb.Collection:
    """Return the compliance_books collection, creating it if absent.

    For read-only operations (count/get metadata), set include_embedding=False to
    avoid initializing the embedding model.
    """
    client = get_client()

    # Prefer opening an existing collection without creating new config,
    # because creating without an embedding function can persist a default
    # embedding backend that later conflicts with the intended embedding.
    if not include_embedding:
        try:
            return client.get_collection(name=CHROMA_COLLECTION)
        except Exception:
            # If it doesn't exist yet, create it using the configured embedding
            # when we can do so without network downloads.
            if EMBEDDING_MODEL.startswith("ollama:"):
                return client.get_or_create_collection(
                    name=CHROMA_COLLECTION,
                    embedding_function=get_embedding_function(),
                    metadata={"hnsw:space": "cosine"},
                )
            # Last resort: create without embedding (may persist a default backend).
            return client.get_or_create_collection(name=CHROMA_COLLECTION)

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------------


def get_all_metadata() -> list[dict[str, Any]]:
    """Return the metadata dicts for every document in the collection."""
    collection = get_collection(include_embedding=False)
    result = collection.get(include=["metadatas"])
    return result.get("metadatas") or []


def list_unique_values(field: str) -> list[str]:
    """Return sorted unique values for a given metadata field."""
    metadatas = get_all_metadata()
    values = {m.get(field) for m in metadatas if m.get(field) is not None}
    return sorted(values)


def collection_size() -> int:
    """Return the total number of chunks stored in the collection."""
    return get_collection(include_embedding=False).count()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_chunks(
    query: str,
    source_tags: list[str],
    top_k: int = RETRIEVAL_TOP_K,
    importance: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for *query* filtered by source_tags.

    Args:
        query:       The semantic query string (e.g. a compliance topic).
        source_tags: List of source metadata values to restrict results to.
        top_k:       Maximum number of chunks to return.
        importance:  Optional importance level filter ("High", "Medium", "Low").

    Returns:
        List of dicts with keys: id, document, source, chapter, importance.
    """
    collection = get_collection(include_embedding=True)

    if not source_tags:
        return []

    # Build ChromaDB $where filter
    if len(source_tags) == 1:
        where: dict[str, Any] = {"source": {"$eq": source_tags[0]}}
    else:
        where = {"$or": [{"source": {"$eq": tag}} for tag in source_tags]}

    if importance:
        importance_clause = {"importance": {"$eq": importance}}
        where = {"$and": [where, importance_clause]}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "id": meta.get("id", ""),
                "document": doc,
                "source": meta.get("source", ""),
                "chapter": meta.get("chapter", ""),
                "importance": meta.get("importance", ""),
                "distance": dist,
            }
        )
    return chunks

"""Make sure a document is in the store. Chunk + embed only on a cache miss."""

import hashlib
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, MIN_CHUNK_CHARS
from .embeddings import embed_texts
from .ingestion import chunk_pages, load_pdf
from .schemas import Chunk
from .vector_store import VectorStore


class NoTextError(ValueError):
    """The PDF has no extractable text (probably scanned). Frontends decide how to report it."""


def cache_key(pdf_path: Path) -> str:
    """Identity of (this file + these settings). Doubles as the doc_id."""
    h = hashlib.sha256(pdf_path.read_bytes())
    # EMBED_MODEL already separates providers (voyage-4 vs nomic-embed-text).
    h.update(f"|{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{MIN_CHUNK_CHARS}".encode())
    return h.hexdigest()[:16]


def prepare_chunks(pdf_path: Path, source: str | None = None) -> tuple[int, list[Chunk]]:
    """PDF -> (page_count, chunks with provenance). No embedding, no store.

    Split out so the eval harness can chunk + embed ONCE and fill several stores.
    """
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise NoTextError("No text could be extracted. Is this a scanned PDF? (OCR is a parked idea.)")
    for c in chunks:  # provenance set here, so ingestion.py stays a pure text concern
        c.source = source or pdf_path.name
    return len(pages), chunks


def build_store(
    pdf_path: Path, store: VectorStore, force: bool = False, source: str | None = None
) -> tuple[str, int]:
    """Return (doc_id, chunk_count). Embeds only if this document isn't stored yet.

    `source` overrides the display filename (the API saves uploads under a hash name).
    """
    doc_id = cache_key(pdf_path)

    if force:
        store.delete(doc_id)

    if store.has(doc_id):
        n = store.count(doc_id)
        print(f"Loaded {n} chunks from {store.name} (doc {doc_id}). No embedding calls.")
        return doc_id, n

    n_pages, chunks = prepare_chunks(pdf_path, source)
    print(f"{n_pages} pages -> {len(chunks)} chunks. Embedding ...")
    vectors = embed_texts([c.text for c in chunks], "document")

    store.add(doc_id, chunks, vectors)
    print(f"Stored {len(chunks)} chunks in {store.name} (doc {doc_id}).")
    return doc_id, len(chunks)

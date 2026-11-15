"""Embedding calls: batching, rate-limit retry, unit-length vectors.

Two providers behind one function. voyage is the real one. ollama exists so the whole
pipeline can be exercised locally for free. Callers never know which one ran.
"""

import time

import httpx
import numpy as np
import voyageai

from .config import EMBED_BATCH, EMBED_MODEL, EMBED_PROVIDER, OLLAMA_URL


def embed_batch_with_retry(vo, batch: list[str], input_type: str, tries: int = 6):
    for attempt in range(tries):
        try:
            return vo.embed(batch, model=EMBED_MODEL, input_type=input_type).embeddings
        except voyageai.error.RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"  Voyage rate limit hit, waiting {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(
        "Voyage kept rate-limiting. Add billing details to your Voyage account or lower EMBED_BATCH."
    )


def _embed_voyage(texts: list[str], input_type: str) -> list[list[float]]:
    vo = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embed_batch_with_retry(vo, texts[i : i + EMBED_BATCH], input_type))
    return vectors


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    """Local, free, for development only. input_type is ignored."""
    vectors: list[list[float]] = []
    with httpx.Client(base_url=OLLAMA_URL, timeout=120.0) as http:
        for i in range(0, len(texts), EMBED_BATCH):
            r = http.post("/api/embed", json={"model": EMBED_MODEL, "input": texts[i : i + EMBED_BATCH]})
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
    return vectors


def normalize(arr: np.ndarray) -> np.ndarray:
    """Unit length, so dot product == cosine similarity. Zero rows stay zero instead of NaN."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_texts(texts: list[str], input_type: str) -> np.ndarray:
    """input_type is 'document' for chunks and 'query' for questions."""
    if not texts:
        raise ValueError("embed_texts() needs at least one text")
    if EMBED_PROVIDER == "voyage":
        vectors = _embed_voyage(texts, input_type)
    elif EMBED_PROVIDER == "ollama":
        vectors = _embed_ollama(texts)
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER {EMBED_PROVIDER!r}. Use voyage or ollama.")
    return normalize(np.asarray(vectors, dtype=np.float32))

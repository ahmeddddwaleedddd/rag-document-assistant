"""Find the chunks most similar to a question."""

from .config import TOP_K
from .embeddings import embed_texts
from .vector_store import Hit, VectorStore

# CHANGED: numpy is no longer imported here. The dot product, the argsort and
# the float() conversion all moved into NumpyStore. This file is now purely
# "turn a question into a vector and ask the store", which is why swapping
# backends needs zero changes below.


def retrieve(question: str, store: VectorStore, k: int = TOP_K, where: dict | None = None) -> list[Hit]:
    """Top-k hits, highest cosine similarity first.

    `where` is an optional metadata filter, e.g. {"doc_id": "ab12...", "page": 7}.
    Each store translates it to its own dialect ($eq for Chroma, SQL for pgvector).
    """
    q = embed_texts([question], "query")[0]  # input_type="query" still matters
    return store.search(q, k, where)

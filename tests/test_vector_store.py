"""ONE contract test, run against every backend.

This is the real deliverable of Phase 3: if a test passes for numpy and fails
for Chroma, the backends are not interchangeable and the abstraction is a lie.
"""

import numpy as np
import pytest

from rag_assistant.schemas import Chunk
from rag_assistant.vector_store import ChromaStore, NumpyStore, PgVectorStore

DOC = "testdoc0000000a"


def unit(vec):
    """Normalise, because every store assumes unit-length vectors."""
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


@pytest.fixture
def chunks():
    return [
        Chunk(id=0, page=1, text="Refunds are processed within 5 business days.", source="t.pdf"),
        Chunk(id=1, page=2, text="Shipping is free above 50 euros.", source="t.pdf"),
    ]


@pytest.fixture
def vectors():
    return np.vstack([unit([1, 0, 0]), unit([0, 1, 0])])


def make_numpy(tmp_path):
    return NumpyStore(cache_dir=tmp_path / "cache")


def make_chroma(tmp_path):
    pytest.importorskip("chromadb")
    return ChromaStore(path=tmp_path / "chroma", collection="test_contract")


def make_pg(tmp_path):
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    try:
        return PgVectorStore(table="test_contract")   # skip cleanly when no DB is running
    except Exception as e:
        pytest.skip(f"no Postgres available: {e}")


@pytest.fixture(params=[make_numpy, make_chroma, make_pg], ids=["numpy", "chroma", "pgvector"])
def store(request, tmp_path):
    s = request.param(tmp_path)
    s.delete(DOC)          # start clean: Chroma and Postgres persist between runs
    yield s
    s.delete(DOC)
    s.close()


def test_empty_store_has_nothing(store):
    assert store.has(DOC) is False
    assert store.search(unit([1, 0, 0]), k=5, where={"doc_id": DOC}) == []


def test_add_then_has_and_count(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    assert store.has(DOC) is True
    assert store.count(DOC) == 2


def test_search_returns_similarity_sorted_desc(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    hits = store.search(unit([1, 0, 0]), k=2, where={"doc_id": DOC})

    assert hits[0][0].id == 0                        # closest chunk first
    assert hits[0][1] == pytest.approx(1.0, abs=0.01)  # SIMILARITY, not distance
    assert hits[0][1] >= hits[1][1]                  # descending
    assert -1.0 <= hits[1][1] <= 1.0                 # same scale as MIN_SCORE


def test_chunk_round_trips_intact(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    chunk, _ = store.search(unit([0, 1, 0]), k=1, where={"doc_id": DOC})[0]
    assert (chunk.id, chunk.page, chunk.source) == (1, 2, "t.pdf")
    assert chunk.text == "Shipping is free above 50 euros."


def test_metadata_filter(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    hits = store.search(unit([1, 0, 0]), k=5, where={"doc_id": DOC, "page": 2})
    assert [c.page for c, _ in hits] == [2]          # page 1 excluded despite scoring higher


def test_add_is_idempotent(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    store.add(DOC, chunks, vectors)                  # must upsert, not duplicate or raise
    assert store.count(DOC) == 2


def test_delete_only_removes_one_document(store, chunks, vectors):
    store.add(DOC, chunks, vectors)
    store.add("otherdoc0000001", chunks, vectors)
    store.delete(DOC)
    assert store.has(DOC) is False
    assert store.count("otherdoc0000001") == 2
    store.delete("otherdoc0000001")


def test_rejects_unknown_filter_key(store):
    with pytest.raises(ValueError):
        store.search(unit([1, 0, 0]), k=1, where={"text": "nope"})

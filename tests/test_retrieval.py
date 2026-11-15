from conftest import fake_embed

from rag_assistant.retrieval import retrieve
from rag_assistant.schemas import Chunk
from rag_assistant.vector_store import NumpyStore


def test_retrieve_ranks_relevant_chunk_first(tmp_path):
    chunks = [Chunk(0, 1, "refunds take five business days"), Chunk(1, 2, "shipping is free above fifty euros")]
    store = NumpyStore(tmp_path)
    store.add("d" * 16, chunks, fake_embed([c.text for c in chunks]))
    hits = retrieve("how long do refunds take", store, where={"doc_id": "d" * 16})
    assert hits[0][0].id == 0 and hits[0][1] > hits[1][1]

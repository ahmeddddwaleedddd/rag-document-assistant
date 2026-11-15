import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from rag_assistant.embeddings import normalize
from rag_assistant.schemas import Chunk
from rag_assistant.vector_store import VectorStore

DIM = 64


def fake_embed(texts, input_type="document"):
    """Deterministic bag-of-words hashing: shared words -> higher cosine. No network."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    return normalize(out)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test: fake embeddings, logs in tmp. A real network call is impossible."""
    monkeypatch.setattr("rag_assistant.retrieval.embed_texts", fake_embed)
    monkeypatch.setattr("rag_assistant.indexing.embed_texts", fake_embed)
    monkeypatch.setattr("rag_assistant.config.LOG_FILE", tmp_path / "queries.jsonl")


def usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def tool_response(answerable=True, answer="Five business days.", citations=(0,)):
    block = SimpleNamespace(type="tool_use", input={"answerable": answerable, "answer": answer,
                                                    "citations": list(citations)})
    return SimpleNamespace(content=[block], usage=usage())


class FakeStream:
    def __init__(self, response, pieces):
        self._response, self._pieces = response, pieces

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        for p in self._pieces:
            yield SimpleNamespace(type="input_json", partial_json=p)

    def get_final_message(self):
        return self._response


class FakeClient:
    """Fake anthropic.Anthropic. Rewrite calls (no tools) get text; answer calls get a tool."""

    def __init__(self, response=None, rewrite="refund processing duration", pieces=None):
        self.response = response or tool_response()
        self.rewrite = rewrite
        self.pieces = pieces or ['{"answerable": true, "answer": "Five ', 'business days.", "citations": [0]}']
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if "tools" not in kw:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.rewrite)], usage=usage())
        return self.response

    def stream(self, **kw):
        self.calls.append(kw)
        return FakeStream(self.response, self.pieces)


class FakeStore(VectorStore):
    """Returns canned hits. Lets generation tests run with zero API calls."""
    name = "fake"

    def __init__(self, hits):
        self._hits = hits

    def has(self, doc_id): return True
    def count(self, doc_id=None): return len(self._hits)
    def add(self, doc_id, chunks, vectors): pass
    def delete(self, doc_id): pass
    def search(self, query_vector, k, where=None): return self._hits[:k]


@pytest.fixture
def chunk():
    return Chunk(id=0, page=4, text="Refunds are processed within 5 business days.", source="t.pdf")

import pytest
from conftest import FakeClient, fake_embed

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag_assistant.api import main as api  # noqa: E402
from rag_assistant.schemas import Chunk  # noqa: E402
from rag_assistant.vector_store import NumpyStore  # noqa: E402

DOC = "ab" * 8


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_assistant.config.API_KEYS", ("secret",))
    monkeypatch.setattr("rag_assistant.config.UPLOAD_DIR", tmp_path / "up")
    chunks = [Chunk(0, 4, "Refunds are processed within 5 business days.", "t.pdf")]
    NumpyStore(tmp_path / "s").add(DOC, chunks, fake_embed([c.text for c in chunks]))
    monkeypatch.setattr(api, "open_store", lambda: NumpyStore(tmp_path / "s"))
    api.app.dependency_overrides[api.get_client] = lambda: FakeClient()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


BODY = {"question": "How long do refunds take?", "doc_id": DOC}


def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_missing_key_401(client):
    assert client.post("/ask", json=BODY).status_code == 401


def test_wrong_key_403(client):
    assert client.post("/ask", json=BODY, headers={"X-API-Key": "nope"}).status_code == 403


def test_ask_ok(client):
    r = client.post("/ask", json=BODY, headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json()["sources"][0]["page"] == 4


def test_unknown_doc_404(client):
    r = client.post("/ask", json={**BODY, "doc_id": "0" * 16}, headers={"X-API-Key": "secret"})
    assert r.status_code == 404


def test_rejects_non_pdf(client):
    r = client.post("/ingest", files={"file": ("x.txt", b"hello", "text/plain")}, headers={"X-API-Key": "secret"})
    assert r.status_code == 415


def test_stream_sends_tokens_and_done(client):
    r = client.post("/ask/stream", json=BODY, headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    assert "event: token" in r.text and "event: done" in r.text

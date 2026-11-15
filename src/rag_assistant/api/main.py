"""FastAPI app: /health, /ingest, /ask, /ask/stream (SSE)."""

import hashlib
from collections.abc import Iterable

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from .. import __version__, config
from ..generation import stream_from_hits
from ..graph import answer_with_retry, finish, retrieve_with_retry
from ..indexing import NoTextError, build_store
from ..querylog import log_query
from ..vector_store import VectorStore, get_store
from .auth import require_api_key

app = FastAPI(title="RAG Document Assistant", version=__version__)


def open_store() -> VectorStore:
    """One store per request. Tests monkeypatch this. Pooling is a parked idea."""
    return get_store(config.VECTOR_STORE)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    page: int | None = Field(default=None, ge=1)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    source: str


def _where(req: AskRequest) -> dict:
    where: dict = {"doc_id": req.doc_id}
    if req.page is not None:
        where["page"] = req.page
    return where


def _require_doc(store: VectorStore, doc_id: str) -> None:
    if not store.has(doc_id):
        raise HTTPException(404, f"Unknown doc_id {doc_id}. POST the PDF to /ingest first.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "store": config.VECTOR_STORE}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    data = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "Only PDF files are accepted")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}.pdf"  # never trust the filename
    path.write_bytes(data)
    source = file.filename or path.name
    with open_store() as store:
        try:
            doc_id, n = build_store(path, store, source=source)
        except NoTextError as e:
            raise HTTPException(422, str(e)) from e
    return IngestResponse(doc_id=doc_id, chunks=n, source=source)


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(req: AskRequest, client: anthropic.Anthropic = Depends(get_client)) -> dict:
    with open_store() as store:
        _require_doc(store, req.doc_id)
        result = answer_with_retry(client, req.question, store, where=_where(req))
        log_query(result, store.name, req.doc_id)
    return result


@app.post("/ask/stream", response_class=EventSourceResponse, dependencies=[Depends(require_api_key)])
def ask_stream(req: AskRequest, client: anthropic.Anthropic = Depends(get_client)) -> Iterable[ServerSentEvent]:
    # The store is opened INSIDE the generator, so it lives exactly as long as the stream.
    with open_store() as store:
        if not store.has(req.doc_id):
            yield ServerSentEvent(data={"detail": f"Unknown doc_id {req.doc_id}"}, event="error")
            return
        hits, state = retrieve_with_retry(client, req.question, store, where=_where(req))
        for kind, payload in stream_from_hits(client, req.question, hits):
            if kind == "token":
                yield ServerSentEvent(data=payload, event="token")
            else:
                result = finish(payload, state)
                log_query(result, store.name, req.doc_id)
                yield ServerSentEvent(data=result, event="done")

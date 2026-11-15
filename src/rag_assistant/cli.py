"""Terminal interface."""

import argparse
import os
import sys
from pathlib import Path

import anthropic

from .config import EMBED_PROVIDER, VECTOR_STORE
from .generation import answer_from_hits, stream_from_hits
from .graph import answer_with_retry, finish, retrieve_with_retry
from .indexing import NoTextError, build_store
from .ingestion import chunk_pages, load_pdf
from .querylog import log_query
from .retrieval import retrieve
from .schemas import Chunk
from .vector_store import VectorStore, get_store


def print_result(result: dict, debug: bool, show_answer: bool = True) -> None:
    if show_answer:
        print(f"\nAssistant: {result['answer']}")
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - page {s['page']} (chunk {s['chunk_id']}, score {s['score']}): {s['preview']}")
    if debug:
        print(f"\n[debug] reason: {result['reason'] or 'answered'}")
        print(f"[debug] queries: {result.get('queries')}")
        print(f"[debug] retrieved: {result['retrieved']}")
        print(f"[debug] tokens: {result['usage']}")


def dump_chunks(chunks: list[Chunk], out: Path = Path("chunks_preview.txt")) -> None:
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"=== chunk {c.id} | page {c.page} | {len(c.text)} chars ===\n{c.text}\n\n")
    print(f"Wrote {len(chunks)} chunks to {out}. Open it and READ them.")


def ask_blocking(client, question: str, store: VectorStore, where: dict, retry: bool) -> dict:
    if retry:
        return answer_with_retry(client, question, store, where=where)
    return answer_from_hits(client, question, retrieve(question, store, where=where))


def ask_streaming(client, question: str, store: VectorStore, where: dict, retry: bool) -> dict:
    if retry:
        hits, state = retrieve_with_retry(client, question, store, where=where)
    else:
        hits, state = retrieve(question, store, where=where), None
    print("\nAssistant: ", end="", flush=True)
    streamed: list[str] = []
    result: dict = {}
    for kind, payload in stream_from_hits(client, question, hits):
        if kind == "token":
            print(payload, end="", flush=True)
            streamed.append(payload)
        else:
            result = finish(payload, state)
    text = "".join(streamed)
    print("" if text else result["answer"])
    if text and text != result["answer"]:  # streamed tokens were provisional and got rejected
        print(f"[not accepted: {result['reason']}] {result['answer']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(prog="rag-assistant", description="Chat with one PDF.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--dump-chunks", action="store_true", help="write chunks_preview.txt and exit")
    ap.add_argument("--debug", action="store_true", help="show retrieval scores, queries and token usage")
    ap.add_argument("--store", choices=["chroma", "pgvector", "numpy"], default=VECTOR_STORE,
                    help=f"vector backend (default from VECTOR_STORE env: {VECTOR_STORE})")
    ap.add_argument("--reset", action="store_true", help="delete this document from the store and re-index")
    ap.add_argument("--page", type=int, help="metadata filter: only search this page")
    ap.add_argument("--no-retry", action="store_true", help="disable the Phase 4 rewrite-and-retry loop")
    ap.add_argument("--stream", action="store_true", help="print the answer while it is generated")
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.exit(f"File not found: {args.pdf}")

    if args.dump_chunks:  # a text problem: no store, no keys, no cost
        try:
            dump_chunks(chunk_pages(load_pdf(args.pdf)))
        except Exception as e:  # noqa: BLE001 - CLI boundary
            sys.exit(f"Could not read PDF: {e}")
        return

    required = ["ANTHROPIC_API_KEY"] + (["VOYAGE_API_KEY"] if EMBED_PROVIDER == "voyage" else [])
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in.")

    try:
        store = get_store(args.store)
    except ImportError as e:
        sys.exit(f"Store {args.store!r} needs an extra dependency: {e}\n"
                 f"  chroma   -> pip install -e '.[chroma]'\n"
                 f"  pgvector -> pip install -e '.[pg]'  (and docker compose up -d postgres)")

    with store:
        try:
            doc_id, n = build_store(args.pdf, store, force=args.reset)
        except NoTextError as e:
            sys.exit(str(e))

        where: dict = {"doc_id": doc_id}
        if args.page is not None:
            where["page"] = args.page

        client = anthropic.Anthropic()
        page_note = f", page={args.page}" if args.page is not None else ""
        print(f"\nReady: {args.pdf.name} ({n} chunks, store={store.name}{page_note}). "
              f"Ask a question, or type 'exit'.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break
            if args.stream:
                result = ask_streaming(client, question, store, where, retry=not args.no_retry)
            else:
                result = ask_blocking(client, question, store, where, retry=not args.no_retry)
            log_query(result, store.name, doc_id)
            print_result(result, args.debug, show_answer=not args.stream)

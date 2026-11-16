# Changelog

## 1.0.0 — 2026-11-14

First release with measured, published metrics.

- Evaluated against a 42-question golden set (direct facts, tables, multi-chunk answers, confusable sections, vague phrasing, unanswerable questions) on a real 15-page PRD.
- Results: recall@3 0.941, recall@5 0.971, MRR 0.86, Chroma-vs-NumPy agreement 1.00, refusal accuracy 0.976. Full numbers in `eval/results/`.
- Verified `pgvector` backend end-to-end against a real Dockerized Postgres instance: tests pass (not skipped), manual Q&A matches NumPy/Chroma answers, full `docker compose up --build` stack reachable at `/health` and `/docs`.
- Documented a real port-conflict issue (another local Postgres instance competing for 5432) and its fix, so the same problem doesn't cost time twice.
- `MAX_TOKENS` raised to 4000 for small local models (`qwen3:4b` via Ollama): confirmed via raw SDK testing that this model spends a large share of its token budget on internal reasoning before emitting the actual tool call, and the previous default (1000) could silently truncate before an answer was produced.
- Added `docs/architecture.md` and `LICENSE`.

## 0.9.0

- Phase 4: LangGraph rewrite-and-retry when top score < RETRY_BELOW; hits merged across attempts.
- Phase 6: eval harness (recall@k, MRR, top-k agreement vs numpy, p95 latency, refusal accuracy, faithfulness).
- Phase 7: FastAPI /ingest, /ask, /ask/stream (SSE), API-key auth (401/403, fail closed).
- Phase 8: Dockerfile + compose api service. Phase 9: Streamlit UI. Phase 10: CI.
- Fixes: pgvector filtered-search underfill (iterative_scan), per-provider collections,
  NumpyStore.count(None), no sys.exit in library code, --page 0.

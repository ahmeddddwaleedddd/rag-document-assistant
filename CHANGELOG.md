## 0.9.0
- Phase 4: LangGraph rewrite-and-retry when top score < RETRY_BELOW; hits merged across attempts.
- Phase 6: eval harness (recall@k, MRR, top-k agreement vs numpy, p95 latency, refusal accuracy, faithfulness).
- Phase 7: FastAPI /ingest, /ask, /ask/stream (SSE), API-key auth (401/403, fail closed).
- Phase 8: Dockerfile + compose api service. Phase 9: Streamlit UI. Phase 10: CI.
- Fixes: pgvector filtered-search underfill (iterative_scan), per-provider collections,
  NumpyStore.count(None), no sys.exit in library code, --page 0.

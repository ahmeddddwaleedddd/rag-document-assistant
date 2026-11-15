"""Phase 6 harness.

    python eval/run_eval.py data/doc.pdf --stores numpy chroma pgvector
    python eval/run_eval.py data/doc.pdf --stores chroma --generate   # costs LLM tokens

Cost design: the document is chunked + embedded ONCE and the questions are embedded
ONCE; those same vectors fill and query every store. Store differences are then
purely ANN effects, never embedding noise.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from rag_assistant.config import CLAUDE_MODEL, TOP_K
from rag_assistant.embeddings import embed_texts
from rag_assistant.evaluation import mean, overlap, percentile, recall_at_k, reciprocal_rank, refusal_accuracy
from rag_assistant.generation import answer_from_hits, build_context
from rag_assistant.graph import retrieve_with_retry
from rag_assistant.indexing import cache_key, prepare_chunks
from rag_assistant.prompts import JUDGE_PROMPT, JUDGE_TOOL
from rag_assistant.vector_store import get_store


def judge(client, answer: str, hits) -> bool | None:
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=500, system=JUDGE_PROMPT, tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
        messages=[{"role": "user", "content": f"EXCERPTS:\n{build_context(hits)}\n\nANSWER:\n{answer}"}],
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    return None if block is None else bool(block.input.get("faithful"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    ap.add_argument("--stores", nargs="+", default=["numpy", "chroma"])
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--generate", action="store_true", help="also measure refusal accuracy + faithfulness")
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    doc_id = cache_key(args.pdf)
    _, chunks = prepare_chunks(args.pdf)
    doc_vectors = None
    q_vectors = embed_texts([q["question"] for q in questions], "query")

    report: dict = {"time": datetime.now(UTC).isoformat(), "doc_id": doc_id,
                    "k": args.k, "n_questions": len(questions), "stores": {}}
    baseline: dict[str, list[int]] = {}

    for name in args.stores:
        with get_store(name) as store:
            if not store.has(doc_id):
                if doc_vectors is None:
                    doc_vectors = embed_texts([c.text for c in chunks], "document")
                store.add(doc_id, chunks, doc_vectors)
            r3, rk, rr, lat, agree = [], [], [], [], []
            for q, qv in zip(questions, q_vectors, strict=True):
                t0 = time.perf_counter()
                hits = store.search(qv, args.k, where={"doc_id": doc_id})
                lat.append((time.perf_counter() - t0) * 1000)
                ids = [c.id for c, _ in hits]
                if name == "numpy":
                    baseline[q["id"]] = ids
                elif q["id"] in baseline:
                    agree.append(overlap(ids, baseline[q["id"]]))
                if q["expected_pages"]:
                    pages = [c.page for c, _ in hits]
                    r3.append(recall_at_k(pages, q["expected_pages"], 3))
                    rk.append(recall_at_k(pages, q["expected_pages"], args.k))
                    rr.append(reciprocal_rank(pages, q["expected_pages"]))
            report["stores"][name] = {
                "recall@3": round(mean(r3), 3), f"recall@{args.k}": round(mean(rk), 3),
                "mrr": round(mean(rr), 3), "p95_ms": round(percentile(lat, 95), 2),
                "agreement_vs_numpy": round(mean(agree), 3) if agree else None,
            }

    if args.generate:
        client = anthropic.Anthropic()
        with get_store(args.stores[0]) as store:
            pairs, faithful = [], []
            for q in questions:
                hits, _ = retrieve_with_retry(client, q["question"], store, where={"doc_id": doc_id})
                res = answer_from_hits(client, q["question"], hits)
                pairs.append((q["answerable"], res["answerable"]))
                if res["answerable"]:
                    verdict = judge(client, res["answer"], hits)
                    if verdict is not None:
                        faithful.append(1.0 if verdict else 0.0)
        report["generation"] = {"store": args.stores[0], "refusal_accuracy": round(refusal_accuracy(pairs), 3),
                                "faithfulness": round(mean(faithful), 3) if faithful else None}

    out = Path("eval/results") / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

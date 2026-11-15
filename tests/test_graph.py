from conftest import FakeClient, FakeStore

from rag_assistant.graph import answer_with_retry, merge_hits
from rag_assistant.schemas import Chunk


def test_strong_retrieval_no_rewrite(chunk):
    client = FakeClient()
    r = answer_with_retry(client, "refund?", FakeStore([(chunk, 0.9)]))
    assert r["attempts"] == 1 and len(r["queries"]) == 1
    assert all("tools" in c for c in client.calls)  # no rewrite call was made


def test_weak_retrieval_rewrites_once(chunk):
    client = FakeClient()
    r = answer_with_retry(client, "money back?", FakeStore([(chunk, 0.30)]))
    assert r["attempts"] == 2
    assert r["queries"] == ["money back?", "refund processing duration"]
    assert r["usage"]["input_tokens"] == 20  # answer + rewrite tokens both counted


def test_empty_store_does_not_retry():
    client = FakeClient()
    r = answer_with_retry(client, "q", FakeStore([]))
    assert r["attempts"] == 1 and client.calls == []


def test_merge_keeps_best_score_and_dedupes():
    a, b = Chunk(0, 1, "a"), Chunk(1, 2, "b")
    merged = merge_hits([(a, 0.3)], [(a, 0.5), (b, 0.4)])
    assert [(c.id, s) for c, s in merged] == [(0, 0.5), (1, 0.4)]

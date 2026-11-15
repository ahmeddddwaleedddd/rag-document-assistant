from conftest import FakeClient, FakeStore, tool_response

from rag_assistant.config import NOT_IN_DOC
from rag_assistant.generation import answer_from_hits, answer_question, stream_from_hits


def test_answers_with_valid_citation(chunk):
    r = answer_from_hits(FakeClient(), "refund time?", [(chunk, 0.9)])
    assert r["answerable"] and r["sources"][0]["page"] == 4


def test_guard0_no_hits_skips_llm():
    client = FakeClient()
    r = answer_from_hits(client, "q", [])
    assert not r["answerable"] and client.calls == []


def test_guard1_low_score_skips_llm(chunk):
    client = FakeClient()
    r = answer_from_hits(client, "q", [(chunk, 0.05)])
    assert r["answer"] == NOT_IN_DOC and client.calls == []


def test_guard2_model_refuses(chunk):
    r = answer_from_hits(FakeClient(tool_response(answerable=False, answer="Not covered.")), "q", [(chunk, 0.9)])
    assert not r["answerable"] and r["reason"] == "model said not answerable"


def test_guard3_invented_citation_rejected(chunk):
    r = answer_from_hits(FakeClient(tool_response(citations=[99])), "q", [(chunk, 0.9)])
    assert not r["answerable"] and r["answer"] == NOT_IN_DOC


def test_answer_question_still_works_with_store(chunk):
    assert answer_question(FakeClient(), "refund?", FakeStore([(chunk, 0.9)]))["answerable"]


def test_stream_yields_tokens_then_done(chunk):
    events = list(stream_from_hits(FakeClient(), "q", [(chunk, 0.9)]))
    text = "".join(p for k, p in events if k == "token")
    assert text == "Five business days."
    assert events[-1][0] == "done" and events[-1][1]["answerable"]


def test_stream_guard_overrides_tokens(chunk):
    events = list(stream_from_hits(FakeClient(tool_response(citations=[99])), "q", [(chunk, 0.9)]))
    assert events[-1][1]["answer"] == NOT_IN_DOC  # streamed text was provisional

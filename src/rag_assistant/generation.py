"""Generate the answer: Claude must call the tool (answer + citations) or refuse.

Two entry points share every guard:
  answer_from_hits()  one blocking call   (CLI default, /ask, eval harness)
  stream_from_hits()  yields answer text as it is generated, then the final result
"""

from collections.abc import Iterator
from typing import Any

import anthropic
import jiter
from pydantic import ValidationError

from .config import CLAUDE_MODEL, FORCE_TOOL, MAX_TOKENS, MIN_SCORE, NOT_IN_DOC
from .prompts import ANSWER_TOOL, SYSTEM_PROMPT
from .retrieval import retrieve
from .schemas import AnswerResult
from .vector_store import Hit, VectorStore

# ("token", str) zero or more times, then exactly one ("done", result_dict)
StreamEvent = tuple[str, Any]


def build_context(hits: list[Hit]) -> str:
    parts = [f'<chunk id="{c.id}" page="{c.page}">\n{c.text}\n</chunk>' for c, _ in hits]
    return "<chunks>\n" + "\n".join(parts) + "\n</chunks>"


def new_result(question: str, hits: list[Hit]) -> dict:
    return {
        "question": question,
        "answerable": False,
        "answer": NOT_IN_DOC,
        "sources": [],
        "retrieved": [{"chunk_id": c.id, "page": c.page, "score": round(s, 3)} for c, s in hits],
        "queries": [question],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "reason": "",
    }


def passes_precheck(result: dict, hits: list[Hit]) -> bool:
    """Guards 0 and 1, both free. False means: don't pay for an LLM call."""
    if not hits:
        result["reason"] = "no chunks retrieved (empty store or filter matched nothing)"
        return False
    best = max(s for _, s in hits)
    if best < MIN_SCORE:
        result["reason"] = f"best score {best:.2f} < MIN_SCORE {MIN_SCORE}"
        return False
    return True


def request_kwargs(question: str, hits: list[Hit]) -> dict:
    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": [ANSWER_TOOL],
        "messages": [{"role": "user", "content": f"{build_context(hits)}\n\nQuestion: {question}"}],
    }
    if FORCE_TOOL:
        kwargs["tool_choice"] = {"type": "tool", "name": ANSWER_TOOL["name"]}
    return kwargs


def finalize(result: dict, hits: list[Hit], response) -> dict:
    """Guards 2 and 3 on a complete response. Identical for blocking and streaming."""
    result["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        result["reason"] = "model returned no tool call"
        return result
    try:
        parsed = AnswerResult(**tool_block.input)
    except (ValidationError, TypeError):
        result["reason"] = "model output failed validation"
        return result

    # Guard 2: the model itself says the chunks don't contain the answer.
    if not parsed.answerable:
        result["answer"] = parsed.answer or NOT_IN_DOC
        result["reason"] = "model said not answerable"
        return result

    # Guard 3: cite or refuse. by_id is built from the hits we actually sent.
    by_id = {c.id: (c, s) for c, s in hits}
    valid = [cid for cid in dict.fromkeys(parsed.citations) if cid in by_id]
    if not valid:
        result["reason"] = "answer had no valid citations"
        return result

    result["answerable"] = True
    result["answer"] = parsed.answer
    result["sources"] = [
        {
            "chunk_id": cid,
            "page": by_id[cid][0].page,
            "source": by_id[cid][0].source,
            "score": round(by_id[cid][1], 3),
            "preview": by_id[cid][0].text[:140].replace("\n", " ") + "...",
        }
        for cid in valid
    ]
    return result


def answer_from_hits(client: anthropic.Anthropic, question: str, hits: list[Hit]) -> dict:
    result = new_result(question, hits)
    if not passes_precheck(result, hits):
        return result
    try:
        response = client.messages.create(**request_kwargs(question, hits))
    except anthropic.APIError as e:
        result["answer"] = f"(API error: {e})"
        result["reason"] = "api_error"
        return result
    return finalize(result, hits, response)


def stream_from_hits(client: anthropic.Anthropic, question: str, hits: list[Hit]) -> Iterator[StreamEvent]:
    """Stream the `answer` field of the tool call as it is generated.

    The answer is a tool ARGUMENT, so it arrives as input_json deltas, not text deltas.
    We re-parse the accumulated JSON with jiter's "trailing-strings" mode (the same
    parser the SDK uses), which exposes the half-written answer string.

    IMPORTANT for clients: tokens are provisional. Guards 2-3 run on the complete
    response, and the final "done" result is the authority. If the answer is
    rejected, the frontend must replace what it streamed with result["answer"].
    """
    result = new_result(question, hits)
    if not passes_precheck(result, hits):
        yield "done", result
        return

    buf, sent = "", 0
    try:
        with client.messages.stream(**request_kwargs(question, hits)) as stream:
            for event in stream:
                if event.type != "input_json":
                    continue
                buf += event.partial_json
                try:
                    partial = jiter.from_json(buf.encode("utf-8"), partial_mode="trailing-strings")
                except ValueError:
                    continue  # e.g. buffer ends mid-escape; next delta completes it
                answer = partial.get("answer") if isinstance(partial, dict) else None
                if isinstance(answer, str) and len(answer) > sent:
                    yield "token", answer[sent:]
                    sent = len(answer)
            response = stream.get_final_message()
    except anthropic.APIError as e:
        result["answer"] = f"(API error: {e})"
        result["reason"] = "api_error"
        yield "done", result
        return

    yield "done", finalize(result, hits, response)


def answer_question(
    client: anthropic.Anthropic, question: str, store: VectorStore, where: dict | None = None
) -> dict:
    """v1/v2-compatible entry point: retrieve once, answer once (no retry)."""
    return answer_from_hits(client, question, retrieve(question, store, where=where))

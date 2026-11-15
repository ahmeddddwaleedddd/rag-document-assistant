"""Phase 4: retrieval with one smart retry, as a LangGraph state machine.

    START -> retrieve -> route --strong | no hits | out of attempts--> generate -> END
                ^          |
                |        weak
                +-- rewrite

"Weak" (top score < RETRY_BELOW) is a HIGHER bar than "hopeless" (MIN_SCORE):
between the two we still answer, but only after trying a better query.
Hits from all attempts are MERGED, so a bad rewrite can never make retrieval worse.
The user's original question is what Claude answers. The rewrite only drives search.
"""

from typing import TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from .config import CLAUDE_MODEL, MAX_ATTEMPTS, RETRY_BELOW, TOP_K
from .generation import answer_from_hits
from .prompts import REWRITE_PROMPT
from .retrieval import retrieve
from .vector_store import Hit, VectorStore


def _zero_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0}


class RAGState(TypedDict, total=False):
    question: str               # user's words, never modified
    query: str                  # what we embed on this attempt
    queries: list[str]          # every query tried (logs, debugging, interviews)
    where: dict | None
    attempts: int
    hits: list[Hit]             # merged across attempts, best first
    rewrite_usage: dict[str, int]
    result: dict


def merge_hits(old: list[Hit], new: list[Hit], k: int = TOP_K) -> list[Hit]:
    """Union by chunk identity, keep each chunk's best score, return top-k."""
    best: dict[tuple, Hit] = {}
    for chunk, score in [*old, *new]:
        key = (chunk.source, chunk.page, chunk.id)
        if key not in best or score > best[key][1]:
            best[key] = (chunk, score)
    return sorted(best.values(), key=lambda h: -h[1])[:k]


def rewrite_query(client: anthropic.Anthropic, question: str) -> tuple[str, dict[str, int]]:
    """One cheap LLM call. Any failure falls back to the original question."""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.APIError:
        return question, _zero_usage()
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    line = text.splitlines()[0].strip().strip("\"'") if text else ""
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return (line or question), usage


def build_graph(client: anthropic.Anthropic, store: VectorStore, *, generate: bool = True):
    """Dependencies are bound by closure, so tests can inject fakes. generate=False is
    used by streaming frontends: the graph decides WHAT to retrieve, they stream the answer."""

    def retrieve_node(state: RAGState) -> dict:
        new = retrieve(state["query"], store, where=state.get("where"))
        return {"hits": merge_hits(state.get("hits", []), new), "attempts": state.get("attempts", 0) + 1}

    def rewrite_node(state: RAGState) -> dict:
        new_query, usage = rewrite_query(client, state["question"])
        prev = state.get("rewrite_usage") or _zero_usage()
        return {
            "query": new_query,
            "queries": [*state.get("queries", []), new_query],
            "rewrite_usage": {k: prev[k] + usage[k] for k in prev},
        }

    def generate_node(state: RAGState) -> dict:
        return {"result": answer_from_hits(client, state["question"], state.get("hits", []))}

    done = "generate" if generate else END

    def route(state: RAGState) -> str:
        hits = state.get("hits") or []
        if not hits:  # empty store / filter: a rewrite cannot fix that, save the money
            return done
        if hits[0][1] >= RETRY_BELOW or state.get("attempts", 0) >= MAX_ATTEMPTS:
            return done
        return "rewrite"

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite_node)
    g.add_edge(START, "retrieve")
    g.add_edge("rewrite", "retrieve")
    if generate:
        g.add_node("generate", generate_node)
        g.add_edge("generate", END)
    g.add_conditional_edges("retrieve", route, ["rewrite", done])
    return g.compile()


def _initial(question: str, where: dict | None) -> RAGState:
    return {
        "question": question,
        "query": question,
        "queries": [question],
        "where": where,
        "attempts": 0,
        "hits": [],
        "rewrite_usage": _zero_usage(),
    }


def finish(result: dict, state: dict | None) -> dict:
    """Attach the retry trace and add rewrite tokens to the cost."""
    if not state:
        return result
    result["queries"] = state.get("queries", result.get("queries", []))
    result["attempts"] = state.get("attempts", 1)
    extra = state.get("rewrite_usage") or _zero_usage()
    for key in ("input_tokens", "output_tokens"):
        result["usage"][key] = result["usage"].get(key, 0) + extra.get(key, 0)
    return result


def answer_with_retry(
    client: anthropic.Anthropic, question: str, store: VectorStore, where: dict | None = None
) -> dict:
    state = build_graph(client, store).invoke(_initial(question, where))
    return finish(state["result"], state)


def retrieve_with_retry(
    client: anthropic.Anthropic, question: str, store: VectorStore, where: dict | None = None
) -> tuple[list[Hit], dict]:
    state = build_graph(client, store, generate=False).invoke(_initial(question, where))
    return state.get("hits", []), state

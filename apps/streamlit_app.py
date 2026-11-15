"""Streamlit UI. A separate consumer of the API: it never imports the package core.

    API_URL=http://localhost:8000 API_KEY=... streamlit run apps/streamlit_app.py
"""

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Ask this document", page_icon="📄")
st.title("📄 Ask this document")
st.caption("Answers come only from your PDF, with page citations, or an honest refusal.")


def sse_events(resp: httpx.Response):
    """Minimal SSE parser: yields (event, data) pairs."""
    event, data = "message", []
    for line in resp.iter_lines():
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


with st.sidebar:
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf is not None and st.session_state.get("file_name") != pdf.name:
        with st.spinner("Indexing (only the first time) ..."):
            r = httpx.post(f"{API_URL}/ingest", headers=HEADERS, timeout=600,
                           files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
        if r.status_code == 200:
            st.session_state.update(doc_id=r.json()["doc_id"], file_name=pdf.name, messages=[])
            st.success(f"Ready: {r.json()['chunks']} chunks")
        else:
            st.error(f"{r.status_code}: {r.text}")

if "doc_id" not in st.session_state:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        final: dict = {}

        def tokens():
            body = {"question": question, "doc_id": st.session_state.doc_id}
            with httpx.stream("POST", f"{API_URL}/ask/stream", json=body, headers=HEADERS, timeout=120) as r:
                if r.status_code != 200:
                    final["answer"] = f"Error {r.status_code}: {r.read().decode()}"
                    return
                for event, data in sse_events(r):
                    if event == "token":
                        yield data
                    elif event in {"done", "error"}:
                        final.update(data if event == "done" else {"answer": data["detail"]})

        streamed = st.write_stream(tokens())
        answer = final.get("answer", "")
        if answer and answer != streamed:  # provisional tokens rejected by the guards
            st.warning(answer)
        sources = final.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**Page {s['page']}** (score {s['score']}): {s['preview']}")
        text = answer + "".join(f"\n\n- page {s['page']}" for s in sources)
        st.session_state.messages.append({"role": "assistant", "content": text})

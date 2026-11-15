"""Load a PDF, clean the text, split it into chunks (never crossing a page)."""

import re
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS
from .schemas import Chunk


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def split_units(text: str) -> list[str]:
    """Split page text into paragraph/sentence-sized units, each <= CHUNK_SIZE."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= CHUNK_SIZE:
            units.append(para)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > CHUNK_SIZE:  # last resort for giant unbroken text
                units.append(sentence[:CHUNK_SIZE])
                sentence = sentence[CHUNK_SIZE:]
            if sentence:
                units.append(sentence)
    return units


def overlap_tail(text: str) -> str:
    tail = text[-CHUNK_OVERLAP:]
    if " " in tail:
        tail = tail[tail.find(" ") + 1 :]  # start on a word boundary
    return tail


def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        current = ""
        page_chunks = []
        for unit in split_units(text):
            if current and len(current) + 1 + len(unit) > CHUNK_SIZE:
                page_chunks.append(current)
                current = f"{overlap_tail(current)} {unit}"
            else:
                current = f"{current} {unit}".strip()
        if current.strip():
            page_chunks.append(current)
        for c in page_chunks:
            if len(c) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(id=len(chunks), page=page_no, text=c))
    return chunks
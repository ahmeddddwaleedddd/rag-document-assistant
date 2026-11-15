from dataclasses import dataclass

from pydantic import BaseModel, Field


class AnswerResult(BaseModel):
    # UNCHANGED. citations stay plain per-document ints: the store adds doc_id
    # around them, so Claude's contract never had to learn about databases.
    answerable: bool
    answer: str
    citations: list[int] = Field(default_factory=list)


@dataclass
class Chunk:
    id: int
    page: int
    text: str
    # NEW in v2: the filename, so a shared collection can say where a chunk came
    # from without a second lookup. Default "" on purpose, for two reasons:
    #   1. ingestion.py keeps creating Chunk(id=, page=, text=) unchanged
    #   2. old .cache JSON files have no "source" key, and Chunk(**d) still works
    source: str = ""

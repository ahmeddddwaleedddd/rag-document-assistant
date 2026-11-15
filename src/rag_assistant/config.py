"""Settings and constants. Every one of these is a decision to record in the decision log."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# usecwd=True: look for .env in the folder you run the command from
load_dotenv(find_dotenv(usecwd=True))


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, "").split(",") if part.strip())


# --- LLM -------------------------------------------------------------------
# The Anthropic SDK also reads ANTHROPIC_BASE_URL, which is how the same code
# talks to a local Ollama server for free testing (see README).
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
# Ollama's Anthropic-compatible API does not support tool_choice. FORCE_TOOL=false
# drops it and relies on the system prompt ("Always respond by calling the tool").
FORCE_TOOL = _bool("FORCE_TOOL", "true")

# --- Embeddings --------------------------------------------------------------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "voyage").lower()  # voyage | ollama
EMBED_MODEL = os.getenv("EMBED_MODEL", "voyage-4" if EMBED_PROVIDER == "voyage" else "nomic-embed-text")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# --- Chunking and retrieval -------------------------------------------------
CHUNK_SIZE = 1000      # max characters per chunk (roughly 200-250 tokens)
CHUNK_OVERLAP = 150    # characters repeated at the start of the next chunk
MIN_CHUNK_CHARS = 30   # drop tiny fragments (page numbers, stray headers)
TOP_K = 5              # chunks sent to Claude
# Below MIN_SCORE: hopeless, refuse without calling the LLM.
# Below RETRY_BELOW: weak, rewrite the query and retry once (Phase 4).
# Both are STARTING GUESSES calibrated for voyage-4. Tune with --debug and the eval harness.
# Every store returns cosine similarity, so the same numbers hold on all backends.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))
RETRY_BELOW = float(os.getenv("RETRY_BELOW", "0.40"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))  # 1 original + 1 rewrite

# --- Vector store ----------------------------------------------------------
# Different providers produce different dimensions (voyage-4: 1024, nomic: 768),
# and one Chroma collection / pgvector table cannot mix them. Suffix by provider.
_SUFFIX = "" if EMBED_PROVIDER == "voyage" else f"_{EMBED_PROVIDER}"
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")              # chroma | pgvector | numpy
COLLECTION = os.getenv("COLLECTION", f"rag_chunks{_SUFFIX}")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ".chroma"))
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5432/rag")
PG_TABLE = os.getenv("PG_TABLE", f"rag_chunks{_SUFFIX}")
PG_EF_SEARCH = int(os.getenv("PG_EF_SEARCH", "100"))  # pgvector default is 40
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "500"))  # local DB write is cheap, network embed is not

CACHE_DIR = Path(".cache")     # only used by NumpyStore
LOG_FILE = Path("logs/queries.jsonl")

# --- API (Phase 7) ---------------------------------------------------------
API_KEYS = _csv("API_KEYS")    # comma-separated. Empty = API refuses every request (fail closed).
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))

NOT_IN_DOC = "I couldn't find that in the document."

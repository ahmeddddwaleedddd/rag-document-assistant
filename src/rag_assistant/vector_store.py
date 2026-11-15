"""One interface, three backends.

THE CONTRACT every backend must honour:
  - search() returns COSINE SIMILARITY in [-1, 1], higher = better, sorted desc.
    Chroma and pgvector natively return DISTANCE (0 = identical), so each one
    converts with `1 - distance` before returning. This is what lets MIN_SCORE
    and the three guards in generation.py stay backend-independent.
  - (doc_id, chunk_id) is the globally unique key. Chunk.id restarts at 0 per
    document, so chunk_id alone is NOT unique in a shared collection/table.
  - add() is idempotent (upsert), so a crashed run can be re-run safely.

Layer 2: imports config + schemas only. Backend libraries are imported inside
__init__, so missing optional extras only break the backend you asked for.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import (
    CACHE_DIR,
    CHROMA_DIR,
    COLLECTION,
    PG_DSN,
    PG_EF_SEARCH,
    PG_TABLE,
    UPSERT_BATCH,
    VECTOR_STORE,
)
from .schemas import Chunk

# (chunk, similarity) — same shape v1's retrieve() returned, so callers barely change.
Hit = tuple[Chunk, float]

# Only these keys may be filtered on. A whitelist, because for pgvector these
# strings become SQL identifiers.
FILTERABLE = ("doc_id", "page", "source")


class VectorStore:
    """Abstract base. Subclasses must implement the five abstract methods."""

    name = "base"

    def has(self, doc_id: str) -> bool:
        """True if this document is already indexed (skip embedding = save money)."""
        raise NotImplementedError

    def add(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Upsert chunks + their unit-length vectors. vectors[i] belongs to chunks[i]."""
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, k: int, where: dict | None = None) -> list[Hit]:
        """Top-k by cosine SIMILARITY, descending. May return fewer than k."""
        raise NotImplementedError

    def delete(self, doc_id: str) -> None:
        """Remove one document. Must not touch other documents."""
        raise NotImplementedError

    def count(self, doc_id: str | None = None) -> int:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. No-op for file-based backends."""
        return None

    # Lets callers write `with get_store() as store:` and never leak a connection.
    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _check_where(where: dict | None) -> dict:
        where = where or {}
        bad = set(where) - set(FILTERABLE)
        if bad:
            raise ValueError(f"Cannot filter on {sorted(bad)}. Allowed: {list(FILTERABLE)}")
        return where


class NumpyStore(VectorStore):
    """v1's behaviour, now behind the interface: JSON + .npy files, brute-force search.

    Why keep it: it is exact (no ANN approximation), so it is the ground-truth
    baseline for recall@k, and it needs no server, so tests run offline and free.
    Not for production: every vector is loaded into RAM and compared.
    """

    name = "numpy"

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.dir = Path(cache_dir)
        self._docs: dict[str, tuple[list[Chunk], np.ndarray]] = {}  # loaded documents

    def _files(self, doc_id: str) -> tuple[Path, Path]:
        return self.dir / f"{doc_id}.chunks.json", self.dir / f"{doc_id}.vectors.npy"

    def has(self, doc_id: str) -> bool:
        cf, vf = self._files(doc_id)
        return cf.exists() and vf.exists()  # BOTH, so a half-written cache is a miss

    def add(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        cf, vf = self._files(doc_id)
        cf.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8")
        np.save(vf, vectors)
        self._docs[doc_id] = (chunks, vectors)

    def _load(self, doc_id: str) -> None:
        if doc_id in self._docs or not self.has(doc_id):
            return
        cf, vf = self._files(doc_id)
        # Chunk(**d) tolerates old cache files without "source" because the
        # dataclass field has a default.
        chunks = [Chunk(**d) for d in json.loads(cf.read_text(encoding="utf-8"))]
        self._docs[doc_id] = (chunks, np.load(vf))

    def search(self, query_vector: np.ndarray, k: int, where: dict | None = None) -> list[Hit]:
        where = self._check_where(where)
        if "doc_id" in where:
            self._load(str(where["doc_id"]))
        else:
            for f in sorted(self.dir.glob("*.chunks.json")):  # no filter: search everything
                self._load(f.name.removesuffix(".chunks.json"))

        hits: list[Hit] = []
        for doc_id, (chunks, vectors) in self._docs.items():
            if "doc_id" in where and where["doc_id"] != doc_id:
                continue
            scores = vectors @ query_vector  # unit vectors -> dot product IS cosine similarity
            for i, chunk in enumerate(chunks):
                if any(getattr(chunk, key) != val for key, val in where.items() if key != "doc_id"):
                    continue
                hits.append((chunk, float(scores[i])))  # float(): np.float32 is not JSON-serialisable

        hits.sort(key=lambda h: -h[1])
        return hits[:k]

    def delete(self, doc_id: str) -> None:
        for f in self._files(doc_id):
            f.unlink(missing_ok=True)
        self._docs.pop(doc_id, None)

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:  # v2 bug: only counted docs already in memory
            return sum(self.count(f.name.removesuffix(".chunks.json")) for f in self.dir.glob("*.chunks.json"))
        self._load(doc_id)
        return len(self._docs.get(doc_id, ([], None))[0])



class ChromaStore(VectorStore):
    """Chroma, embedded + persistent. Local default."""

    name = "chroma"

    def __init__(self, path: Path = CHROMA_DIR, collection: str = COLLECTION):
        import chromadb  # imported here so `pip install .` without [chroma] still works

        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(
            name=collection,
            # CRITICAL: Chroma's default space is l2 (squared Euclidean), not cosine.
            # Modern syntax; the old metadata={"hnsw:space": "cosine"} is deprecated.
            configuration={"hnsw": {"space": "cosine"}},
            # We always pass our own Voyage vectors, so Chroma must not try to
            # embed anything itself (its default downloads an ONNX model).
            # If your chromadb version rejects None here, delete this line:
            # it is never called as long as we only pass embeddings.
            embedding_function=None,
        )

    def has(self, doc_id: str) -> bool:
        return self.count(doc_id) > 0

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self.col.count()
        return len(self.col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def add(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):  # Chroma caps records per call
            batch = chunks[i : i + UPSERT_BATCH]
            self.col.upsert(  # upsert, not add: re-running never raises "ID already exists"
                ids=[f"{doc_id}:{c.id}" for c in batch],  # globally unique key
                embeddings=[v.tolist() for v in vectors[i : i + UPSERT_BATCH]],
                documents=[c.text for c in batch],
                metadatas=[
                    {"doc_id": doc_id, "chunk_id": c.id, "page": c.page, "source": c.source}
                    for c in batch
                ],
            )

    @staticmethod
    def _to_chroma_where(where: dict) -> dict | None:
        """Chroma needs an explicit $and once there is more than one condition."""
        if not where:
            return None
        clauses = [{key: {"$eq": val}} for key, val in where.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, query_vector: np.ndarray, k: int, where: dict | None = None) -> list[Hit]:
        where = self._check_where(where)
        res = self.col.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=self._to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        # query() is a BATCH api: results are nested one level per query vector.
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True):
            chunk = Chunk(
                id=int(meta["chunk_id"]),
                page=int(meta["page"]),
                text=text,
                source=str(meta.get("source", "")),
            )
            hits.append((chunk, 1.0 - float(dist)))  # cosine DISTANCE -> similarity
        return hits

    def delete(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})


class PgVectorStore(VectorStore):
    """Postgres + pgvector. The 'you already run a database' option."""

    name = "pgvector"

    def __init__(self, dsn: str = PG_DSN, table: str = PG_TABLE):
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        self._psycopg = psycopg
        self.table = table
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)  # must run AFTER the extension exists
        # HNSW applies WHERE filters AFTER the index scan. With the default ef_search=40,
        # `WHERE doc_id = ...` on a shared table can return fewer than k rows.
        # iterative_scan (pgvector >= 0.8) keeps scanning until k rows pass the filter;
        # strict_order keeps results exactly sorted by distance.
        self.conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(PG_EF_SEARCH)))
        self.conn.execute("SET hnsw.iterative_scan = strict_order")

    def _sql(self):
        from psycopg import sql

        return sql

    def _table_exists(self) -> bool:
        # to_regclass returns NULL instead of raising when the table is absent.
        return self.conn.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()[0] is not None

    def _ensure_table(self, dim: int) -> None:
        """Created lazily so the embedding dimension is read from the data, never hardcoded."""
        sql = self._sql()
        t = sql.Identifier(self.table)
        self.conn.execute(
            sql.SQL(
                """CREATE TABLE IF NOT EXISTS {t} (
                       doc_id    text   NOT NULL,
                       chunk_id  int    NOT NULL,
                       page      int    NOT NULL,
                       source    text   NOT NULL DEFAULT '',
                       text      text   NOT NULL,
                       embedding vector({dim}) NOT NULL,
                       PRIMARY KEY (doc_id, chunk_id)
                   )"""
            ).format(t=t, dim=sql.Literal(dim))
        )
        self.conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} (doc_id)").format(
                i=sql.Identifier(f"{self.table}_doc_id_idx"), t=t
            )
        )
        # vector_cosine_ops MUST match the <=> operator used in search().
        # Mismatched opclass = the index is silently ignored.
        self.conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)").format(
                i=sql.Identifier(f"{self.table}_embedding_idx"), t=t
            )
        )

    def has(self, doc_id: str) -> bool:
        return self._table_exists() and self.count(doc_id) > 0

    def count(self, doc_id: str | None = None) -> int:
        if not self._table_exists():
            return 0
        sql = self._sql()
        t = sql.Identifier(self.table)
        if doc_id is None:
            q, params = sql.SQL("SELECT count(*) FROM {t}").format(t=t), ()
        else:
            q, params = sql.SQL("SELECT count(*) FROM {t} WHERE doc_id = %s").format(t=t), (doc_id,)
        return int(self.conn.execute(q, params).fetchone()[0])

    def add(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from pgvector import Vector

        sql = self._sql()
        self._ensure_table(int(vectors.shape[1]))  # dimension from the data
        rows = [
            (doc_id, c.id, c.page, c.source, c.text, Vector(vectors[i]))
            for i, c in enumerate(chunks)
        ]
        q = sql.SQL(
            """INSERT INTO {t} (doc_id, chunk_id, page, source, text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (doc_id, chunk_id) DO UPDATE
                 SET page = EXCLUDED.page, source = EXCLUDED.source,
                     text = EXCLUDED.text, embedding = EXCLUDED.embedding"""
        ).format(t=sql.Identifier(self.table))
        with self.conn.cursor() as cur:
            for i in range(0, len(rows), UPSERT_BATCH):
                cur.executemany(q, rows[i : i + UPSERT_BATCH])

    def search(self, query_vector: np.ndarray, k: int, where: dict | None = None) -> list[Hit]:
        from pgvector import Vector

        where = self._check_where(where)
        if not self._table_exists():
            return []
        sql = self._sql()

        conditions = [sql.SQL("TRUE")]
        params: list = []
        for key, val in where.items():  # key is whitelisted, so Identifier() is safe
            conditions.append(sql.SQL("{c} = %s").format(c=sql.Identifier(key)))
            params.append(val)

        vec = Vector(query_vector)
        q = sql.SQL(
            """SELECT chunk_id, page, source, text, 1 - (embedding <=> %s) AS similarity
               FROM {t}
               WHERE {w}
               ORDER BY embedding <=> %s
               LIMIT %s"""
        ).format(t=sql.Identifier(self.table), w=sql.SQL(" AND ").join(conditions))
        # <=> is cosine DISTANCE; `1 - distance` gives similarity. ORDER BY uses the
        # raw distance (ascending) so the HNSW index can actually be used.
        rows = self.conn.execute(q, [vec, *params, vec, k]).fetchall()
        return [
            (Chunk(id=int(cid), page=int(page), text=text, source=source), float(sim))
            for cid, page, source, text, sim in rows
        ]

    def delete(self, doc_id: str) -> None:
        if not self._table_exists():
            return
        sql = self._sql()
        self.conn.execute(
            sql.SQL("DELETE FROM {t} WHERE doc_id = %s").format(t=sql.Identifier(self.table)),
            (doc_id,),
        )

    def close(self) -> None:
        self.conn.close()


def get_store(name: str | None = None) -> VectorStore:
    """Factory: the ONE place that knows which backends exist."""
    choice = (name or VECTOR_STORE).lower()
    if choice == "numpy":
        return NumpyStore()
    if choice == "chroma":
        return ChromaStore()
    if choice in {"pgvector", "pg", "postgres"}:
        return PgVectorStore()
    raise ValueError(f"Unknown store {choice!r}. Use numpy, chroma or pgvector.")

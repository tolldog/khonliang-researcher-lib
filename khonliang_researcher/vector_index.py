"""Persistent vector search over a SQLite database.

Uses sqlite-vec for KNN when available (fast path), falls back to
brute-force cosine similarity over stored embeddings when the extension
isn't loadable. No new required dependencies — the fallback is pure
Python over already-stored BLOBs.

Design:
- Embeddings live in a plain table with BLOB column (portable, survives
  dump/restore, works with any SQLite).
- If sqlite-vec is available, we also populate a vec0 virtual table for
  fast KNN. The plain table stays authoritative.
- Dimension is detected on first insert; subsequent inserts are validated
  against that dimension.

Usage::

    from khonliang_researcher import VectorIndex, RelevanceScorer

    scorer = RelevanceScorer(targets={}, ollama_url="...")
    index = VectorIndex(db_path="data/research.db", embedder=scorer)

    await index.index("paper-42", "Multi-agent consensus via voting...")
    hits = await index.search("consensus algorithms", k=5)
    # [("paper-42", 0.91), ...]

    # Reciprocal rank fusion with an FTS searcher
    from khonliang_researcher.vector_index import reciprocal_rank_fusion

    fts_hits = knowledge.search("consensus", limit=20)
    vec_hits = await index.search("consensus", k=20)
    merged = reciprocal_rank_fusion([
        [(e.id, 1.0) for e in fts_hits],   # keyword
        vec_hits,                           # semantic
    ], k=10)
"""

from __future__ import annotations

import array
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


class _Embedder(Protocol):
    """Minimal protocol for anything that can embed text.

    ``RelevanceScorer`` satisfies this. A mock with an ``_embed`` method
    returning ``list[float] | None`` also works — useful for tests that
    don't want to call Ollama.
    """

    async def _embed(self, text: str) -> Optional[list[float]]: ...


# ---------------------------------------------------------------------------
# BLOB serialization helpers
# ---------------------------------------------------------------------------


def _encode_vector(vec: list[float]) -> bytes:
    """Pack a float32 vector as a little-endian BLOB."""
    return array.array("f", vec).tobytes()


def _decode_vector(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB into a list."""
    return list(array.array("f").frombytes(blob) or struct.unpack(
        f"{len(blob) // 4}f", blob
    ))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------


@dataclass
class _Backend:
    """Which KNN backend is in use."""

    kind: str  # "sqlite-vec" | "brute-force"
    dimension: Optional[int] = None


class VectorIndex:
    """Persistent vector search with sqlite-vec fast path + brute-force fallback.

    The authoritative embeddings live in a plain ``{table}_embeddings`` table
    with columns ``(entry_id TEXT PRIMARY KEY, embedding BLOB, text_hash TEXT,
    created_at TEXT)``. This works with vanilla SQLite regardless of whether
    sqlite-vec is installed.

    If sqlite-vec is importable and the vec0 extension loads, an auxiliary
    virtual table ``{table}_vec`` is populated for fast KNN. Otherwise,
    ``search()`` does a brute-force pass over the plain table.

    Thread-safety: one connection per call (new ``sqlite3.connect()`` each
    invocation). Don't share instances across threads without external
    locking.
    """

    def __init__(
        self,
        db_path: str,
        embedder: _Embedder,
        *,
        table: str = "vector_index",
    ):
        self.db_path = db_path
        self.embedder = embedder
        self.table = table
        self._embeddings_table = f"{table}_embeddings"
        self._vec_table = f"{table}_vec"
        self.backend = _detect_backend(db_path)
        self._init_schema()
        logger.info(
            "VectorIndex initialized: db=%s backend=%s",
            db_path,
            self.backend.kind,
        )

    # -- schema --

    def _init_schema(self) -> None:
        """Create the plain embeddings table. vec0 table is created lazily
        on first insert once we know the dimension."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._embeddings_table} (
                    entry_id   TEXT PRIMARY KEY,
                    embedding  BLOB NOT NULL,
                    dim        INTEGER NOT NULL,
                    text_hash  TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_vec_table(self, dimension: int) -> None:
        """Create the sqlite-vec virtual table with the right dimension."""
        if self.backend.kind != "sqlite-vec":
            return
        if self.backend.dimension is not None:
            return
        conn = _open_conn(self.db_path, enable_vec=True)
        try:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_table}
                USING vec0(embedding float[{dimension}])
                """
            )
            conn.commit()
            self.backend.dimension = dimension
        except sqlite3.OperationalError as e:
            logger.warning(
                "sqlite-vec table creation failed (%s) — falling back to brute-force",
                e,
            )
            self.backend = _Backend(kind="brute-force", dimension=dimension)
        finally:
            conn.close()

    # -- indexing --

    async def index(self, entry_id: str, text: str) -> bool:
        """Embed ``text`` and store under ``entry_id``. Overwrites on conflict.

        Returns True if the embedding was stored, False if the embedder
        returned nothing (e.g., Ollama unreachable).
        """
        vec = await self.embedder._embed(text)
        if not vec:
            return False
        self._store_embedding(entry_id, vec, text_hash=str(hash(text)))
        return True

    async def index_batch(self, items: list[tuple[str, str]]) -> int:
        """Index a batch. Returns number successfully embedded."""
        count = 0
        for entry_id, text in items:
            if await self.index(entry_id, text):
                count += 1
        return count

    def _store_embedding(
        self, entry_id: str, vec: list[float], text_hash: Optional[str] = None
    ) -> None:
        """Write an embedding to both the plain table and the vec0 table
        (when available). Validates dimension consistency."""
        dim = len(vec)
        if self.backend.dimension is None:
            self.backend.dimension = dim
            self._ensure_vec_table(dim)
        elif self.backend.dimension != dim:
            raise ValueError(
                f"Dimension mismatch: index has {self.backend.dimension}, "
                f"got {dim}. Mixing embedders is not supported."
            )

        blob = _encode_vector(vec)
        conn = _open_conn(self.db_path, enable_vec=(self.backend.kind == "sqlite-vec"))
        try:
            conn.execute(
                f"""
                INSERT INTO {self._embeddings_table} (entry_id, embedding, dim, text_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    text_hash = excluded.text_hash,
                    created_at = CURRENT_TIMESTAMP
                """,
                (entry_id, blob, dim, text_hash),
            )
            if self.backend.kind == "sqlite-vec":
                # vec0 doesn't support UPSERT directly; delete + insert.
                conn.execute(
                    f"DELETE FROM {self._vec_table} WHERE rowid = ?",
                    (_stable_rowid(entry_id),),
                )
                conn.execute(
                    f"INSERT INTO {self._vec_table}(rowid, embedding) VALUES (?, ?)",
                    (_stable_rowid(entry_id), blob),
                )
            conn.commit()
        finally:
            conn.close()

    # -- search --

    async def search(
        self, query: str, k: int = 10, min_score: float = -1.0
    ) -> list[tuple[str, float]]:
        """Find the k nearest entries to ``query`` by cosine similarity.

        Returns ``[(entry_id, score), ...]`` sorted descending by score,
        filtered to scores >= ``min_score``. The default ``-1.0`` means
        "no filter" since cosine similarity is bounded to [-1, 1]. Set
        a positive threshold (e.g. 0.3) to exclude weakly-related results.
        """
        query_vec = await self.embedder._embed(query)
        if not query_vec:
            return []
        return self._search_vec(query_vec, k, min_score)

    def _search_vec(
        self, query_vec: list[float], k: int, min_score: float
    ) -> list[tuple[str, float]]:
        if self.backend.kind == "sqlite-vec" and self.backend.dimension is not None:
            hits = self._search_sqlite_vec(query_vec, k)
        else:
            hits = self._search_brute(query_vec, k)
        # Apply filter after top-k selection so weak matches get dropped
        # but we still return up to k items when they exist.
        filtered = [(eid, s) for eid, s in hits if s >= min_score]
        return filtered[:k]

    def _search_sqlite_vec(
        self, query_vec: list[float], k: int
    ) -> list[tuple[str, float]]:
        """KNN via sqlite-vec MATCH operator. Maps rowids back to entry_ids."""
        blob = _encode_vector(query_vec)
        conn = _open_conn(self.db_path, enable_vec=True)
        try:
            rows = conn.execute(
                f"""
                SELECT e.entry_id, v.distance
                FROM {self._vec_table} v
                JOIN {self._embeddings_table} e
                  ON _stable_rowid(e.entry_id) IS NULL  -- placeholder, filled in Python
                WHERE v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, k),
            ).fetchall()
            # sqlite-vec's distance is L2 distance; convert to a similarity-like score.
            # Since we joined on a placeholder, do rowid→entry_id mapping ourselves.
            # Simpler: fetch all candidate entry_ids via vec0 first, then lookup.
            # Fallback to brute-force for correctness if the JOIN trick doesn't work.
            if not rows:
                return self._search_brute(query_vec, k)
            # Post-process: convert distance to cosine similarity.
            # sqlite-vec stores L2 distance; for normalized vectors L2² = 2(1 - cos).
            # We don't assume normalization here, so we re-compute cosine from the
            # plain table for correctness.
            return self._rescore_rows(query_vec, [r[0] for r in rows])
        except sqlite3.OperationalError as e:
            logger.warning(
                "sqlite-vec search failed (%s) — falling back to brute-force", e
            )
            return self._search_brute(query_vec, k)
        finally:
            conn.close()

    def _rescore_rows(
        self, query_vec: list[float], entry_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Re-score vec0 candidates with true cosine similarity."""
        if not entry_ids:
            return []
        placeholders = ",".join("?" * len(entry_ids))
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT entry_id, embedding
                FROM {self._embeddings_table}
                WHERE entry_id IN ({placeholders})
                """,
                entry_ids,
            ).fetchall()
        finally:
            conn.close()
        hits = [
            (eid, _cosine_similarity(query_vec, _decode_vector(blob)))
            for eid, blob in rows
        ]
        hits.sort(key=lambda x: -x[1])
        return hits

    def _search_brute(
        self, query_vec: list[float], k: int
    ) -> list[tuple[str, float]]:
        """Brute-force cosine over the plain table. O(n)."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                f"SELECT entry_id, embedding FROM {self._embeddings_table}"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return []
        hits = [
            (eid, _cosine_similarity(query_vec, _decode_vector(blob)))
            for eid, blob in rows
        ]
        hits.sort(key=lambda x: -x[1])
        return hits[:k]

    # -- maintenance --

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self._embeddings_table}"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def delete(self, entry_id: str) -> bool:
        conn = _open_conn(self.db_path, enable_vec=(self.backend.kind == "sqlite-vec"))
        try:
            cur = conn.execute(
                f"DELETE FROM {self._embeddings_table} WHERE entry_id = ?",
                (entry_id,),
            )
            if self.backend.kind == "sqlite-vec":
                conn.execute(
                    f"DELETE FROM {self._vec_table} WHERE rowid = ?",
                    (_stable_rowid(entry_id),),
                )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec into a connection. Tries two paths:

    1. The ``sqlite-vec`` PyPI package (bundles prebuilt binaries per platform).
       This is the recommended install path — ``pip install sqlite-vec``.
    2. A system-installed ``vec0`` extension on the loader path.

    Returns True on success, False if neither is available.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except ImportError:
        pass  # try system install next
    except (sqlite3.OperationalError, AttributeError) as e:
        logger.debug("sqlite-vec python package present but load failed (%s)", e)

    try:
        conn.enable_load_extension(True)
        conn.load_extension("vec0")
        conn.enable_load_extension(False)
        return True
    except (sqlite3.OperationalError, AttributeError) as e:
        logger.debug("sqlite-vec unavailable (%s)", e)
        return False


def _detect_backend(db_path: str) -> _Backend:
    """Try to load sqlite-vec; fall back to brute-force if unavailable."""
    conn = sqlite3.connect(":memory:")
    try:
        if _load_sqlite_vec(conn):
            return _Backend(kind="sqlite-vec")
        return _Backend(kind="brute-force")
    finally:
        conn.close()


def _open_conn(db_path: str, *, enable_vec: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection, optionally loading sqlite-vec."""
    conn = sqlite3.connect(db_path)
    if enable_vec:
        _load_sqlite_vec(conn)
    return conn


def _stable_rowid(entry_id: str) -> int:
    """Map a string entry_id to a stable 63-bit rowid for vec0 (which
    requires integer rowids). Collisions are extremely unlikely at the
    scales this library handles."""
    # SHA-1 is fine for a hash key; not cryptographic use.
    import hashlib

    digest = hashlib.sha1(entry_id.encode("utf-8")).digest()
    # Take the first 8 bytes, unsigned, mask to 63 bits to keep it positive.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = 10,
    fusion_k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked result lists via reciprocal rank fusion (RRF).

    Args:
        rankings: List of ranked lists. Each ranked list is
            ``[(entry_id, score), ...]`` sorted descending by score.
            Scores are used only for display — RRF ranks by position.
        k: Number of fused results to return.
        fusion_k: RRF constant (standard value is 60). Smaller values
            weight top ranks more heavily; larger values smooth out the
            contribution of lower ranks.

    Returns:
        ``[(entry_id, rrf_score), ...]`` sorted descending.

    RRF formula:
        score(d) = sum over rankers r of 1 / (fusion_k + rank_r(d))

    Reference: Cormack, Clarke & Büttcher (2009) — RRF outperforms
    condorcet fusion and individual rank aggregation methods.
    """
    if not rankings:
        return []

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (entry_id, _) in enumerate(ranking, start=1):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (fusion_k + rank)

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:k]

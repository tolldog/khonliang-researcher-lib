"""Tests for VectorIndex + reciprocal_rank_fusion.

Uses a deterministic MockEmbedder so tests don't need Ollama. Exercises
both the brute-force fallback path (the default when sqlite-vec is not
installed) and sqlite-vec when available.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from khonliang_researcher.vector_index import (
    VectorIndex,
    _cosine_similarity,
    _decode_vector,
    _encode_vector,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Deterministic embedder for tests — no Ollama dependency.

    Hashes the text into a small vector. Similar strings get similar
    vectors because the first few chars dominate. Different strings get
    different vectors. All vectors are 16-dimensional.
    """

    DIM = 16

    async def _embed(self, text: str) -> list[float] | None:
        if not text:
            return None
        # Seed from hash, generate deterministic floats
        h = hashlib.sha256(text.lower().encode()).digest()
        # 16 floats in [-1, 1] derived from the hash
        vec = [((b - 128) / 128.0) for b in h[: self.DIM]]
        # Bias toward first char so semantically-close mocks cluster
        first_char_bias = ord(text[0]) / 255.0
        vec[0] = first_char_bias
        return vec


class NoneEmbedder:
    """Returns None — simulates Ollama being unreachable."""

    async def _embed(self, text: str) -> list[float] | None:
        return None


class BadDimensionEmbedder:
    """Returns a vector of configurable dimension (for dim-mismatch tests)."""

    def __init__(self, dim: int):
        self.dim = dim

    async def _embed(self, text: str) -> list[float]:
        return [0.1] * self.dim


# ---------------------------------------------------------------------------
# BLOB round-trip
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip():
    vec = [0.1, -0.5, 3.14, -2.71, 0.0, 1e-6]
    blob = _encode_vector(vec)
    decoded = _decode_vector(blob)
    assert len(decoded) == len(vec)
    for a, b in zip(decoded, vec):
        assert abs(a - b) < 1e-5


def test_cosine_identical():
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    """Zero vectors give zero similarity (no division by zero)."""
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_mismatched_length_returns_zero():
    """Length mismatch returns 0 rather than raising."""
    assert _cosine_similarity([1.0, 0.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# VectorIndex — init and schema
# ---------------------------------------------------------------------------


def test_init_creates_schema(tmp_path):
    db = str(tmp_path / "index.db")
    idx = VectorIndex(db, MockEmbedder())
    assert idx.count() == 0
    # Backend should be brute-force (sqlite-vec not expected in test env)
    assert idx.backend.kind in ("sqlite-vec", "brute-force")


def test_init_is_idempotent(tmp_path):
    db = str(tmp_path / "index.db")
    VectorIndex(db, MockEmbedder())
    # Reopening an existing index must not error
    idx2 = VectorIndex(db, MockEmbedder())
    assert idx2.count() == 0


# ---------------------------------------------------------------------------
# VectorIndex — indexing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_stores_embedding(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    ok = await idx.index("a", "hello world")
    assert ok is True
    assert idx.count() == 1


@pytest.mark.asyncio
async def test_index_upsert_overwrites(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    await idx.index("a", "first text")
    await idx.index("a", "second text")
    assert idx.count() == 1  # same id, upserted


@pytest.mark.asyncio
async def test_index_skips_when_embedder_returns_none(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), NoneEmbedder())
    ok = await idx.index("a", "whatever")
    assert ok is False
    assert idx.count() == 0


@pytest.mark.asyncio
async def test_index_batch_returns_count(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    count = await idx.index_batch([
        ("a", "apple"),
        ("b", "banana"),
        ("c", "cherry"),
    ])
    assert count == 3
    assert idx.count() == 3


@pytest.mark.asyncio
async def test_dimension_mismatch_raises(tmp_path):
    """Once a dimension is established, mixing embedders fails loudly."""
    idx = VectorIndex(str(tmp_path / "i.db"), BadDimensionEmbedder(dim=8))
    await idx.index("a", "first")  # establishes dim=8
    # Swap embedder to one with different dim
    idx.embedder = BadDimensionEmbedder(dim=16)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        await idx.index("b", "second")


# ---------------------------------------------------------------------------
# VectorIndex — search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_self_first(tmp_path):
    """An indexed entry should be its own top match when queried with
    the same text."""
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    await idx.index("a", "unique text alpha")
    await idx.index("b", "completely different beta")
    await idx.index("c", "yet another gamma")

    hits = await idx.search("unique text alpha", k=3)
    assert len(hits) == 3
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_search_respects_k(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    for i in range(5):
        await idx.index(f"e{i}", f"entry number {i}")
    hits = await idx.search("query", k=3)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_search_empty_index_returns_empty(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    assert await idx.search("anything") == []


@pytest.mark.asyncio
async def test_search_min_score_filter(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    await idx.index("a", "hello")
    # Set min_score high enough that nothing passes
    hits = await idx.search("totally unrelated query xyz", k=10, min_score=0.999)
    # The self-similarity is ~1.0, but no item here was indexed as the query
    # So nothing should pass 0.999
    assert all(s >= 0.999 for _, s in hits)


@pytest.mark.asyncio
async def test_search_returns_sorted_descending(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    for i in range(10):
        await idx.index(f"e{i}", f"text {i}")
    hits = await idx.search("query", k=10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_search_returns_empty_when_embedder_fails(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), NoneEmbedder())
    assert await idx.search("query") == []


# ---------------------------------------------------------------------------
# VectorIndex — delete / count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_entry(tmp_path):
    idx = VectorIndex(str(tmp_path / "i.db"), MockEmbedder())
    await idx.index("a", "text")
    await idx.index("b", "text")
    assert idx.count() == 2
    assert idx.delete("a") is True
    assert idx.count() == 1
    # Deleting again is idempotent (returns False)
    assert idx.delete("a") is False


# ---------------------------------------------------------------------------
# Persistence across VectorIndex instances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embeddings_persist_across_instances(tmp_path):
    db = str(tmp_path / "i.db")
    idx = VectorIndex(db, MockEmbedder())
    await idx.index("a", "persistent text")

    # New instance pointing at the same DB
    idx2 = VectorIndex(db, MockEmbedder())
    assert idx2.count() == 1
    hits = await idx2.search("persistent text", k=1)
    assert hits[0][0] == "a"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_single_ranking_preserves_order():
    ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = reciprocal_rank_fusion([ranking], k=3)
    assert [eid for eid, _ in fused] == ["a", "b", "c"]


def test_rrf_unified_top_item():
    """Items that appear in the top positions of multiple rankings get
    boosted scores relative to items in only one ranking."""
    r1 = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    r2 = [("b", 1.0), ("a", 0.9), ("d", 0.8)]
    fused = reciprocal_rank_fusion([r1, r2], k=4)
    top_ids = [eid for eid, _ in fused[:2]]
    # a and b appear in both — they should outrank c and d
    assert set(top_ids) == {"a", "b"}


def test_rrf_respects_k():
    r1 = [(f"e{i}", 1.0 - i * 0.1) for i in range(10)]
    r2 = [(f"e{i}", 1.0 - i * 0.1) for i in range(5, 15)]
    fused = reciprocal_rank_fusion([r1, r2], k=3)
    assert len(fused) == 3


def test_rrf_fusion_k_parameter_tunes_weight():
    """fusion_k controls how much top ranks dominate. Smaller values
    make rank-1 contributions much larger than later ranks; larger
    values smooth the distribution."""
    r1 = [("a", 1.0), ("b", 0.9), ("c", 0.8)]

    small = dict(reciprocal_rank_fusion([r1], k=3, fusion_k=1))
    large = dict(reciprocal_rank_fusion([r1], k=3, fusion_k=1000))

    # With fusion_k=1: a=1/2=0.5, c=1/4=0.25 → ratio 2x
    # With fusion_k=1000: a=1/1001≈0.001, c=1/1003≈0.000997 → ratio ~1x
    small_ratio = small["a"] / small["c"]
    large_ratio = large["a"] / large["c"]

    # Small fusion_k gives rank 1 a much bigger advantage over rank 3
    assert small_ratio > large_ratio
    assert small_ratio == pytest.approx(2.0, abs=0.01)
    assert large_ratio == pytest.approx(1.0, abs=0.01)


def test_rrf_disjoint_rankings():
    """No overlap — each item appears in exactly one ranking."""
    r1 = [("a", 1.0), ("b", 0.5)]
    r2 = [("c", 1.0), ("d", 0.5)]
    fused = reciprocal_rank_fusion([r1, r2], k=4)
    # All four should appear
    assert {eid for eid, _ in fused} == {"a", "b", "c", "d"}


def test_rrf_returns_descending_scores():
    r1 = [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)]
    r2 = [("b", 1.0), ("a", 0.9), ("c", 0.8), ("e", 0.7)]
    fused = reciprocal_rank_fusion([r1, r2], k=5)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)

"""Tests for khonliang_researcher.relevance."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khonliang_researcher.relevance import RelevanceScorer, cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_known_value():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    dot = 1*4 + 2*5 + 3*6  # 32
    na = math.sqrt(14)
    nb = math.sqrt(77)
    assert cosine_similarity(a, b) == pytest.approx(dot / (na * nb))


# ---------------------------------------------------------------------------
# RelevanceScorer — construction and initialization
# ---------------------------------------------------------------------------

def test_scorer_init_stores_config():
    targets = {"TSLA": {"description": "Tesla"}}
    scorer = RelevanceScorer(targets=targets, threshold=0.7, content_prefix_len=500)
    assert scorer.targets == targets
    assert scorer.threshold == 0.7
    assert scorer.content_prefix_len == 500
    assert not scorer._ready


def test_scorer_defaults():
    scorer = RelevanceScorer(targets={})
    assert scorer.threshold == 0.6
    assert scorer.content_prefix_len == 1500
    assert scorer.model == "nomic-embed-text"


@pytest.mark.asyncio
async def test_scorer_initialize_embeds_targets():
    targets = {
        "TSLA": {"description": "Tesla electric vehicles"},
        "NVDA": {"description": "NVIDIA GPU computing"},
        "empty": {},  # no description — should be skipped
    }
    scorer = RelevanceScorer(targets=targets)

    fake_embeddings = {
        "Tesla electric vehicles": [0.1, 0.2, 0.3],
        "NVIDIA GPU computing": [0.4, 0.5, 0.6],
    }

    async def mock_embed(text):
        return fake_embeddings.get(text)

    scorer._embed = mock_embed
    await scorer.initialize()

    assert scorer._ready
    assert "TSLA" in scorer._target_embeddings
    assert "NVDA" in scorer._target_embeddings
    assert "empty" not in scorer._target_embeddings


@pytest.mark.asyncio
async def test_scorer_initialize_not_ready_if_no_embeddings():
    scorer = RelevanceScorer(targets={"x": {"description": "test"}})
    scorer._embed = AsyncMock(return_value=None)
    await scorer.initialize()
    assert not scorer._ready


@pytest.mark.asyncio
async def test_scorer_initialize_idempotent():
    scorer = RelevanceScorer(targets={"x": {"description": "test"}})
    scorer._embed = AsyncMock(return_value=[1.0, 2.0])
    await scorer.initialize()
    assert scorer._ready
    scorer._embed.reset_mock()
    await scorer.initialize()
    scorer._embed.assert_not_called()


# ---------------------------------------------------------------------------
# RelevanceScorer — scoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_returns_similarity_per_target():
    scorer = RelevanceScorer(targets={})
    scorer._ready = True
    scorer._target_embeddings = {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
    }
    scorer._embed = AsyncMock(return_value=[1.0, 0.0])

    scores = await scorer.score("test", "content")
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_score_returns_empty_when_embed_fails():
    scorer = RelevanceScorer(targets={})
    scorer._ready = True
    scorer._target_embeddings = {"A": [1.0]}
    scorer._embed = AsyncMock(return_value=None)

    scores = await scorer.score("test", "content")
    assert scores == {}


@pytest.mark.asyncio
async def test_score_auto_initializes():
    scorer = RelevanceScorer(targets={"A": {"description": "test"}})
    scorer._embed = AsyncMock(return_value=[1.0, 0.0])

    scores = await scorer.score("test", "content")
    # Should have called _embed for initialize + score
    assert scorer._ready


@pytest.mark.asyncio
async def test_score_truncates_content():
    scorer = RelevanceScorer(targets={}, content_prefix_len=10)
    scorer._ready = True
    scorer._target_embeddings = {"A": [1.0]}

    captured_text = None
    original_embed = scorer._embed

    async def capture_embed(text):
        nonlocal captured_text
        captured_text = text
        return [1.0]

    scorer._embed = capture_embed
    await scorer.score("title", "x" * 1000)
    assert captured_text is not None
    # title\n\n + 10 chars
    assert len(captured_text) == len("title\n\n") + 10


# ---------------------------------------------------------------------------
# RelevanceScorer — is_relevant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_relevant_above_threshold():
    scorer = RelevanceScorer(targets={}, threshold=0.5)
    scorer._ready = True
    scorer._target_embeddings = {"A": [1.0, 0.0]}
    scorer._embed = AsyncMock(return_value=[1.0, 0.0])

    relevant, scores = await scorer.is_relevant("test", "content")
    assert relevant is True
    assert scores["A"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_is_relevant_below_threshold():
    scorer = RelevanceScorer(targets={}, threshold=0.99)
    scorer._ready = True
    scorer._target_embeddings = {"A": [1.0, 0.0]}
    scorer._embed = AsyncMock(return_value=[0.7, 0.7])

    relevant, scores = await scorer.is_relevant("test", "content")
    assert relevant is False


@pytest.mark.asyncio
async def test_is_relevant_defaults_true_on_failure():
    scorer = RelevanceScorer(targets={})
    scorer._ready = True
    scorer._target_embeddings = {"A": [1.0]}
    scorer._embed = AsyncMock(return_value=None)

    relevant, scores = await scorer.is_relevant("test", "content")
    assert relevant is True
    assert scores == {}


# ---------------------------------------------------------------------------
# RelevanceScorer — signal learning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_signal_no_blackboard():
    scorer = RelevanceScorer(targets={})
    # Should not raise
    await scorer.record_signal("title", "content", "positive")


@pytest.mark.asyncio
async def test_record_signal_posts_to_blackboard():
    board = MagicMock()
    scorer = RelevanceScorer(targets={}, blackboard=board)
    scorer._embed = AsyncMock(return_value=[1.0, 2.0])

    await scorer.record_signal("title", "content", "positive", target="TSLA")

    board.post.assert_called_once()
    call_kwargs = board.post.call_args[1]
    assert call_kwargs["section"] == "relevance_signals"
    assert call_kwargs["content"]["signal"] == "positive"
    assert call_kwargs["content"]["target"] == "TSLA"
    assert call_kwargs["embedding"] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_record_signal_skips_on_embed_failure():
    board = MagicMock()
    scorer = RelevanceScorer(targets={}, blackboard=board)
    scorer._embed = AsyncMock(return_value=None)

    await scorer.record_signal("title", "content", "positive")
    board.post.assert_not_called()


def test_signal_adjustment_no_blackboard():
    scorer = RelevanceScorer(targets={})
    assert scorer._compute_signal_adjustment([1.0]) == 0.0


def test_signal_adjustment_no_matches():
    board = MagicMock()
    board.search_similar.return_value = []
    scorer = RelevanceScorer(targets={}, blackboard=board)
    assert scorer._compute_signal_adjustment([1.0]) == 0.0


def test_signal_adjustment_positive_boosts():
    board = MagicMock()
    entry = MagicMock()
    entry.content = {"signal": "positive"}
    board.search_similar.return_value = [(entry, 0.9)]
    scorer = RelevanceScorer(targets={}, blackboard=board)

    adj = scorer._compute_signal_adjustment([1.0])
    assert adj > 0


def test_signal_adjustment_negative_dampens():
    board = MagicMock()
    entry = MagicMock()
    entry.content = {"signal": "negative"}
    board.search_similar.return_value = [(entry, 0.9)]
    scorer = RelevanceScorer(targets={}, blackboard=board)

    adj = scorer._compute_signal_adjustment([1.0])
    assert adj < 0


def test_signal_adjustment_capped():
    board = MagicMock()
    entries = [(MagicMock(content={"signal": "positive"}), 1.0) for _ in range(20)]
    board.search_similar.return_value = entries
    scorer = RelevanceScorer(targets={}, blackboard=board)

    adj = scorer._compute_signal_adjustment([1.0])
    assert adj <= scorer.SIGNAL_BOOST

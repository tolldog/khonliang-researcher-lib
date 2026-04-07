"""Tests for khonliang_researcher.synthesizer."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang_researcher.synthesizer import BaseSynthesizer, SynthesisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_summary_entry(entry_id, title, parent_id="", assessments=None, tags=None):
    """Create a fake KnowledgeEntry for summary testing."""
    entry = MagicMock()
    entry.id = entry_id
    entry.title = title
    entry.tags = tags or ["summary"]
    entry.content = json.dumps({
        "title": title,
        "abstract": f"Abstract of {title}",
        "key_findings": [f"Finding from {title}"],
        "methods": ["method_a"],
    })
    entry.metadata = {
        "parent_id": parent_id,
        "assessments": assessments or {},
    }
    from khonliang.knowledge.store import Tier
    entry.tier = Tier.DERIVED
    return entry


def make_synthesizer(entries, generate_response="Synthesized content"):
    """Create a BaseSynthesizer with mocked stores."""
    knowledge = MagicMock()
    knowledge.get_by_tier.return_value = entries
    knowledge.search.return_value = entries

    triples = MagicMock()
    triples.build_context.return_value = ""

    pool = MagicMock()
    client = AsyncMock()
    client.generate.return_value = generate_response
    pool.get_client.return_value = client

    return BaseSynthesizer(knowledge, triples, pool)


# ---------------------------------------------------------------------------
# SynthesisResult
# ---------------------------------------------------------------------------

def test_synthesis_result_defaults():
    r = SynthesisResult(query="q", synthesis_type="topic", content="c", document_count=5)
    assert r.document_ids == []
    assert r.success is False


# ---------------------------------------------------------------------------
# _get_summaries
# ---------------------------------------------------------------------------

def test_get_summaries_filters_by_tag():
    good = make_summary_entry("s1", "Good", tags=["summary"])
    bad = make_summary_entry("s2", "Bad", tags=["other"])
    synth = make_synthesizer([good, bad])

    summaries = synth._get_summaries()
    assert len(summaries) == 1
    assert summaries[0]["title"] == "Good"


def test_get_summaries_with_limit():
    entries = [make_summary_entry(f"s{i}", f"Paper {i}") for i in range(10)]
    synth = make_synthesizer(entries)

    summaries = synth._get_summaries(limit=3)
    assert len(summaries) == 3


def test_get_summaries_no_limit():
    entries = [make_summary_entry(f"s{i}", f"Paper {i}") for i in range(10)]
    synth = make_synthesizer(entries)

    summaries = synth._get_summaries(limit=None)
    assert len(summaries) == 10


def test_get_summaries_with_query():
    entries = [make_summary_entry("s1", "Match")]
    synth = make_synthesizer(entries)

    summaries = synth._get_summaries(query="test")
    synth.knowledge.search.assert_called_once()
    assert len(summaries) == 1


def test_get_summaries_parses_json_content():
    entry = make_summary_entry("s1", "Paper", parent_id="p1")
    synth = make_synthesizer([entry])

    summaries = synth._get_summaries()
    assert summaries[0]["summary"]["title"] == "Paper"
    assert summaries[0]["parent_id"] == "p1"


def test_get_summaries_handles_invalid_json():
    entry = make_summary_entry("s1", "Paper")
    entry.content = "not json"
    synth = make_synthesizer([entry])

    summaries = synth._get_summaries()
    assert "raw" in summaries[0]["summary"]


def test_get_summaries_custom_tags():
    entry = make_summary_entry("s1", "News", tags=["news_summary"])
    synth = make_synthesizer([entry])
    synth.summary_tags = ["news_summary"]

    summaries = synth._get_summaries()
    assert len(summaries) == 1


# ---------------------------------------------------------------------------
# _format_summaries
# ---------------------------------------------------------------------------

def test_format_summaries_includes_title():
    synth = make_synthesizer([])
    formatted = synth._format_summaries([{
        "title": "Test Paper",
        "summary": {"title": "Test Paper", "abstract": "An abstract", "key_findings": ["F1"], "methods": ["M1"]},
    }])
    assert "Test Paper" in formatted
    assert "An abstract" in formatted


def test_format_summaries_truncates_at_max_chars():
    synth = make_synthesizer([])
    summaries = [{
        "title": f"Paper {i}",
        "summary": {"title": f"Paper {i}", "abstract": "x" * 200, "key_findings": [], "methods": []},
    } for i in range(100)]

    formatted = synth._format_summaries(summaries, max_chars=500)
    assert len(formatted) < 700  # some overhead allowed
    assert "more documents" in formatted


# ---------------------------------------------------------------------------
# topic_summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_topic_summary_success():
    entries = [make_summary_entry("s1", "RL Paper")]
    synth = make_synthesizer(entries, generate_response="Topic synthesis content")

    result = await synth.topic_summary("reinforcement learning")
    assert result.success
    assert result.synthesis_type == "topic"
    assert result.document_count == 1
    assert "Topic synthesis content" in result.content


@pytest.mark.asyncio
async def test_topic_summary_no_papers():
    synth = make_synthesizer([])

    result = await synth.topic_summary("nothing")
    assert not result.success
    assert result.document_count == 0


# ---------------------------------------------------------------------------
# target_brief
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_target_brief_filters_by_score():
    high = make_summary_entry("s1", "Relevant", assessments={"TSLA": {"score": 0.8}})
    low = make_summary_entry("s2", "Irrelevant", assessments={"TSLA": {"score": 0.1}})
    none = make_summary_entry("s3", "Unscored")
    synth = make_synthesizer([high, low, none], generate_response="Brief content")

    result = await synth.target_brief("TSLA", "Tesla electric vehicles")
    assert result.success
    assert result.document_count == 1
    assert result.document_ids == ["s1"]


@pytest.mark.asyncio
async def test_target_brief_no_scored_summaries():
    entry = make_summary_entry("s1", "Unrelated")
    synth = make_synthesizer([entry])

    result = await synth.target_brief("TSLA", "Tesla")
    # Current behavior: falls back to unfiltered (PR #2 will change this)
    # Either way, should not error
    assert isinstance(result, SynthesisResult)


@pytest.mark.asyncio
async def test_target_brief_sorts_by_score():
    low = make_summary_entry("s1", "Low", assessments={"proj": {"score": 0.4}})
    high = make_summary_entry("s2", "High", assessments={"proj": {"score": 0.9}})
    synth = make_synthesizer([low, high], generate_response="Brief")

    result = await synth.target_brief("proj", "Description", limit=2)
    assert result.success
    # high should be first
    assert result.document_ids[0] == "s2"


# ---------------------------------------------------------------------------
# landscape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_landscape_success():
    entries = [make_summary_entry(f"s{i}", f"Paper {i}") for i in range(3)]
    synth = make_synthesizer(entries, generate_response="Landscape overview")

    result = await synth.landscape()
    assert result.success
    assert result.synthesis_type == "landscape"
    assert result.document_count == 3


@pytest.mark.asyncio
async def test_landscape_no_papers():
    synth = make_synthesizer([])
    result = await synth.landscape()
    assert not result.success


@pytest.mark.asyncio
async def test_landscape_includes_triple_context():
    entries = [make_summary_entry("s1", "Paper")]
    synth = make_synthesizer(entries, generate_response="Overview")
    synth.triples.build_context.return_value = "A → uses → B"

    result = await synth.landscape()
    assert "Known Relationships" in result.content
    assert "A → uses → B" in result.content


# ---------------------------------------------------------------------------
# Prompt customization via subclass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_subclass_prompt_override():
    class FinanceSynthesizer(BaseSynthesizer):
        TOPIC_PROMPT = "Custom finance prompt for {topic} with {count} docs:\n{summaries}"

    entries = [make_summary_entry("s1", "Market Report")]
    knowledge = MagicMock()
    knowledge.get_by_tier.return_value = entries
    knowledge.search.return_value = entries
    triples = MagicMock()
    triples.build_context.return_value = ""
    pool = MagicMock()
    client = AsyncMock()
    client.generate.return_value = "Finance synthesis"
    pool.get_client.return_value = client

    synth = FinanceSynthesizer(knowledge, triples, pool)
    result = await synth.topic_summary("markets")

    prompt_used = client.generate.call_args[1]["prompt"]
    assert "Custom finance prompt" in prompt_used

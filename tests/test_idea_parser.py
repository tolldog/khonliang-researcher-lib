"""Tests for khonliang_researcher.idea_parser."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang_researcher.idea_parser import (
    BaseIdeaParser,
    DEFAULT_IDEA_PROMPT,
    clean_for_json,
)


# ---------------------------------------------------------------------------
# clean_for_json
# ---------------------------------------------------------------------------

def test_clean_for_json_removes_latex():
    text = r"This $x = y^2$ is math and $$\int dx$$ too"
    cleaned = clean_for_json(text)
    assert "[math]" in cleaned
    assert "$" not in cleaned


def test_clean_for_json_removes_unicode():
    text = "α-particles → β decay ✓"
    cleaned = clean_for_json(text)
    assert "α" not in cleaned
    assert "→" not in cleaned


def test_clean_for_json_collapses_whitespace():
    text = "lots\n\n\n\nof\n\n\nnewlines"
    cleaned = clean_for_json(text)
    assert "\n\n\n" not in cleaned


def test_clean_for_json_preserves_ascii():
    text = "Plain English text with numbers 123 and punctuation."
    cleaned = clean_for_json(text)
    assert "Plain English text" in cleaned
    assert "123" in cleaned


# ---------------------------------------------------------------------------
# BaseIdeaParser construction
# ---------------------------------------------------------------------------

def make_parser(generate_response=None, parser_cls=BaseIdeaParser):
    """Create a BaseIdeaParser with a mocked client."""
    pool = MagicMock()
    client = MagicMock()
    client.generate_json = AsyncMock(return_value=generate_response or {})
    pool.get_client.return_value = client
    parser = parser_cls(model_pool=pool)
    return parser, client


def test_default_prompt_used_when_none_passed():
    parser, _ = make_parser()
    assert parser.system_prompt == DEFAULT_IDEA_PROMPT


def test_custom_prompt_overrides_default():
    pool = MagicMock()
    pool.get_client.return_value = MagicMock(generate_json=AsyncMock())
    parser = BaseIdeaParser(model_pool=pool, system_prompt="Custom prompt")
    assert parser.system_prompt == "Custom prompt"


def test_subclass_can_override_default_prompt():
    class SpecParser(BaseIdeaParser):
        DEFAULT_PROMPT = "Spec-specific decomposition prompt"

    pool = MagicMock()
    pool.get_client.return_value = MagicMock(generate_json=AsyncMock())
    parser = SpecParser(model_pool=pool)
    assert parser.system_prompt == "Spec-specific decomposition prompt"


# ---------------------------------------------------------------------------
# Model selection by length
# ---------------------------------------------------------------------------

def test_short_text_uses_short_model():
    parser, _ = make_parser()
    assert parser._select_model_for_text("short input") == parser.SHORT_MODEL


def test_long_text_uses_long_model():
    parser, _ = make_parser()
    long_text = "x" * (parser.SHORT_THRESHOLD + 1)
    assert parser._select_model_for_text(long_text) == parser.LONG_MODEL


def test_subclass_can_override_models():
    class CustomParser(BaseIdeaParser):
        SHORT_MODEL = "tiny"
        LONG_MODEL = "huge"
        SHORT_THRESHOLD = 100

    parser, _ = make_parser(parser_cls=CustomParser)
    assert parser._select_model_for_text("x" * 50) == "tiny"
    assert parser._select_model_for_text("x" * 200) == "huge"


# ---------------------------------------------------------------------------
# handle()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_returns_normalized_dict():
    response = {
        "title": "Test Idea",
        "source_type": "linkedin",
        "claims": ["claim 1", "claim 2"],
        "search_queries": ["query 1"],
        "keywords": ["k1", "k2"],
    }
    parser, client = make_parser(generate_response=response)
    result = await parser.handle("Some text to parse")

    assert result["success"] is True
    assert result["title"] == "Test Idea"
    assert result["claims"] == ["claim 1", "claim 2"]
    assert result["search_queries"] == ["query 1"]
    client.generate_json.assert_called_once()


@pytest.mark.asyncio
async def test_handle_fills_defaults_for_missing_fields():
    parser, _ = make_parser(generate_response={"claims": ["c"]})
    result = await parser.handle("text")
    assert result["success"] is True
    assert result["title"] == "Untitled idea"
    assert result["source_type"] == "freeform"
    assert result["keywords"] == []


@pytest.mark.asyncio
async def test_handle_handles_non_dict_response():
    parser, _ = make_parser(generate_response=["not a dict"])
    result = await parser.handle("text")
    assert result["success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_handle_handles_client_exception():
    pool = MagicMock()
    client = MagicMock()
    client.generate_json = AsyncMock(side_effect=RuntimeError("LLM down"))
    pool.get_client.return_value = client
    parser = BaseIdeaParser(model_pool=pool)

    result = await parser.handle("text")
    assert result["success"] is False
    assert "LLM down" in result["error"]


@pytest.mark.asyncio
async def test_handle_truncates_long_input():
    parser, client = make_parser(generate_response={"claims": []})
    long_input = "x" * 50_000
    await parser.handle(long_input)

    call_kwargs = client.generate_json.call_args[1]
    # The prompt embeds the cleaned input; cleaned length should be <= MAX_INPUT_CHARS
    assert len(call_kwargs["prompt"]) < 20_000


@pytest.mark.asyncio
async def test_handle_passes_system_prompt():
    parser, client = make_parser(generate_response={"claims": []})
    await parser.handle("text")
    assert client.generate_json.call_args[1]["system"] == DEFAULT_IDEA_PROMPT

"""Tests for khonliang_researcher.best_of_n."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from khonliang_researcher.best_of_n import (
    DEFAULT_SELECTION_PROMPT,
    select_best_of_n,
    serialize_candidates,
)


def make_client(generate_responses):
    """Create a mock client whose generate() returns successive responses.

    The first N calls return the candidates, the (N+1)th returns the
    selection. Pass a list of strings.
    """
    client = MagicMock()
    client.generate = AsyncMock(side_effect=list(generate_responses))
    return client


# ---------------------------------------------------------------------------
# n=1 fast path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_n1_short_circuits_to_single_call():
    client = make_client(["only response"])
    result = await select_best_of_n(client, "prompt", n=1)
    assert result == "only response"
    assert client.generate.call_count == 1


@pytest.mark.asyncio
async def test_n0_treated_as_single():
    client = make_client(["fallback"])
    result = await select_best_of_n(client, "prompt", n=0)
    assert result == "fallback"


# ---------------------------------------------------------------------------
# Multi-candidate selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_n3_generates_three_candidates_then_selects():
    client = make_client(["cand A", "cand B", "cand C", "2"])
    result = await select_best_of_n(client, "prompt", n=3)
    # Selection "2" -> 1-indexed, so candidate B
    assert result == "cand B"
    # 3 candidates + 1 selection = 4 calls
    assert client.generate.call_count == 4


@pytest.mark.asyncio
async def test_invalid_selection_falls_back_to_first():
    client = make_client(["A", "B", "C", "not a number"])
    result = await select_best_of_n(client, "prompt", n=3)
    assert result == "A"


@pytest.mark.asyncio
async def test_out_of_range_selection_falls_back_to_first():
    client = make_client(["A", "B", "C", "99"])
    result = await select_best_of_n(client, "prompt", n=3)
    assert result == "A"


@pytest.mark.asyncio
async def test_zero_selection_falls_back_to_first():
    # "0" - 1 = -1, which is out of range; should fall back
    client = make_client(["A", "B", "C", "0"])
    result = await select_best_of_n(client, "prompt", n=3)
    assert result == "A"


# ---------------------------------------------------------------------------
# return_candidates mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_return_candidates_mode_returns_dict():
    client = make_client(["A", "B", "C", "1"])
    result = await select_best_of_n(
        client, "prompt", n=3, return_candidates=True
    )
    assert isinstance(result, dict)
    assert result["selected"] == 1
    assert result["candidates"] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_serialize_candidates_returns_json_string():
    payload = {"selected": 2, "candidates": ["x", "y"]}
    out = serialize_candidates(payload)
    assert isinstance(out, str)
    assert "selected" in out


# ---------------------------------------------------------------------------
# Customization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_custom_selection_prompt_template():
    client = make_client(["A", "B", "1"])
    custom = "Pick the best of {n}: {candidates}"
    await select_best_of_n(
        client, "prompt", n=2, selection_prompt_template=custom
    )

    selection_call = client.generate.call_args_list[2]
    assert "Pick the best of 2" in selection_call[1]["prompt"]


@pytest.mark.asyncio
async def test_default_prompt_template_has_placeholders():
    assert "{n}" in DEFAULT_SELECTION_PROMPT
    assert "{candidates}" in DEFAULT_SELECTION_PROMPT


@pytest.mark.asyncio
async def test_temperature_passed_through():
    client = make_client(["A", "B", "1"])
    await select_best_of_n(
        client, "prompt", n=2, temperature=0.9, selection_temperature=0.05
    )

    # First two calls are candidates at temperature=0.9
    for i in range(2):
        assert client.generate.call_args_list[i][1]["temperature"] == 0.9
    # Third is the selection at lower temperature
    assert client.generate.call_args_list[2][1]["temperature"] == 0.05


@pytest.mark.asyncio
async def test_model_override_passed_through():
    client = make_client(["A", "B", "1"])
    await select_best_of_n(client, "prompt", n=2, model="qwen2.5:7b")

    for call in client.generate.call_args_list:
        assert call[1]["model"] == "qwen2.5:7b"

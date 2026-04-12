"""Tests for khonliang_researcher.engines."""

import pytest

from khonliang_researcher.engines import (
    BaseSearchEngine,
    EngineRegistry,
    SearchResult,
    WebFetchEngine,
    WebSearchEngine,
)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


def test_search_result_defaults():
    r = SearchResult(title="Test", url="http://example.com")
    assert r.snippet == ""
    assert r.source == ""
    assert r.score == 0.0
    assert r.metadata == {}


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


class MockEngine(BaseSearchEngine):
    name = "mock"
    description = "Mock engine for tests"

    def __init__(self, results=None):
        self._results = results or []

    async def search(self, query, max_results=10, **kwargs):
        return self._results[:max_results]


def test_register_and_list():
    reg = EngineRegistry()
    reg.register(MockEngine())
    assert "mock" in reg.list_engines()
    assert reg.has("mock")
    assert not reg.has("nonexistent")


def test_register_requires_name():
    class NoName(BaseSearchEngine):
        name = ""

        async def search(self, query, **kwargs):
            return []

    reg = EngineRegistry()
    with pytest.raises(ValueError, match="no name"):
        reg.register(NoName())


def test_get_registered():
    reg = EngineRegistry()
    engine = MockEngine()
    reg.register(engine)
    assert reg.get("mock") is engine
    assert reg.get("nonexistent") is None


@pytest.mark.asyncio
async def test_search_all():
    reg = EngineRegistry()
    reg.register(MockEngine(results=[
        SearchResult(title="A", url="http://a.com"),
    ]))

    class OtherEngine(BaseSearchEngine):
        name = "other"

        async def search(self, query, max_results=10, **kwargs):
            return [SearchResult(title="B", url="http://b.com")]

    reg.register(OtherEngine())

    results = await reg.search("test query")
    assert len(results) == 2
    titles = {r.title for r in results}
    assert "A" in titles
    assert "B" in titles


@pytest.mark.asyncio
async def test_search_specific_engines():
    reg = EngineRegistry()
    reg.register(MockEngine(results=[
        SearchResult(title="A", url="http://a.com"),
    ]))

    class OtherEngine(BaseSearchEngine):
        name = "other"

        async def search(self, query, max_results=10, **kwargs):
            return [SearchResult(title="B", url="http://b.com")]

    reg.register(OtherEngine())

    results = await reg.search("test", engines=["mock"])
    assert len(results) == 1
    assert results[0].title == "A"


@pytest.mark.asyncio
async def test_search_tags_source():
    reg = EngineRegistry()
    reg.register(MockEngine(results=[
        SearchResult(title="A", url="http://a.com"),
    ]))

    results = await reg.search("test")
    assert results[0].source == "mock"


@pytest.mark.asyncio
async def test_search_handles_engine_error():
    class FailEngine(BaseSearchEngine):
        name = "fail"

        async def search(self, query, **kwargs):
            raise RuntimeError("engine broke")

    reg = EngineRegistry()
    reg.register(FailEngine())
    reg.register(MockEngine(results=[
        SearchResult(title="OK", url="http://ok.com"),
    ]))

    results = await reg.search("test")
    assert len(results) == 1
    assert results[0].title == "OK"


@pytest.mark.asyncio
async def test_search_empty_registry():
    reg = EngineRegistry()
    results = await reg.search("test")
    assert results == []


@pytest.mark.asyncio
async def test_search_unknown_engine_skipped():
    reg = EngineRegistry()
    reg.register(MockEngine(results=[
        SearchResult(title="A", url="http://a.com"),
    ]))

    results = await reg.search("test", engines=["mock", "nonexistent"])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Default engines exist
# ---------------------------------------------------------------------------


def test_web_search_engine_has_name():
    engine = WebSearchEngine()
    assert engine.name == "web_search"
    assert engine.description


def test_web_fetch_engine_has_name():
    engine = WebFetchEngine()
    assert engine.name == "web_fetch"
    assert engine.description

"""Tests for khonliang_researcher.worker."""

import asyncio
from dataclasses import dataclass
from typing import Optional, Any

import pytest

from khonliang_researcher.worker import BaseQueueWorker


# ---------------------------------------------------------------------------
# Concrete test worker
# ---------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    title: str = ""


class SimpleWorker(BaseQueueWorker):
    def __init__(self, items=None, fail_ids=None, **kwargs):
        super().__init__(**kwargs)
        self.items = list(items or [])
        self.fail_ids = set(fail_ids or [])
        self.processed_items = []

    def count_pending(self) -> int:
        return len(self.items)

    def get_next(self) -> Optional[Any]:
        for item in self.items:
            if self._failed_ids.get(item.id, 0) < self.max_retries_per_item:
                return item
        return None

    async def process_item(self, item) -> bool:
        if item.id in self.fail_ids:
            return False
        self.processed_items.append(item)
        self.items.remove(item)
        return True


class ErrorWorker(BaseQueueWorker):
    """Worker whose process_item raises exceptions."""
    def __init__(self, items=None, **kwargs):
        super().__init__(**kwargs)
        self.items = list(items or [])

    def count_pending(self) -> int:
        return len(self.items)

    def get_next(self):
        for item in self.items:
            if self._failed_ids.get(item.id, 0) < self.max_retries_per_item:
                return item
        return None

    async def process_item(self, item) -> bool:
        raise ValueError(f"Simulated error for {item.id}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_initial_stats():
    w = SimpleWorker()
    s = w.stats
    assert s["processed"] == 0
    assert s["failed"] == 0
    assert s["skipped"] == 0
    assert s["running"] is False
    assert s["pending"] == 0


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_batch_processes_all():
    items = [Item(f"item_{i}", f"Title {i}") for i in range(5)]
    w = SimpleWorker(items=items, pause_between=0)

    stats = await w.run_batch()
    assert stats["processed"] == 5
    assert stats["failed"] == 0
    assert w.count_pending() == 0


@pytest.mark.asyncio
async def test_run_batch_with_limit():
    items = [Item(f"item_{i}") for i in range(10)]
    w = SimpleWorker(items=items, pause_between=0)

    stats = await w.run_batch(limit=3)
    assert stats["processed"] == 3
    assert w.count_pending() == 7


@pytest.mark.asyncio
async def test_run_batch_empty_queue():
    w = SimpleWorker(items=[], pause_between=0)
    stats = await w.run_batch()
    assert stats["processed"] == 0


@pytest.mark.asyncio
async def test_run_batch_tracks_failures():
    items = [Item("good"), Item("bad")]
    w = SimpleWorker(items=items, fail_ids={"bad"}, pause_between=0, max_retries_per_item=1)

    stats = await w.run_batch()
    assert stats["processed"] == 1
    assert stats["skipped"] == 1  # bad exceeded retries


@pytest.mark.asyncio
async def test_run_batch_retries_then_skips():
    items = [Item("bad")]
    w = SimpleWorker(items=items, fail_ids={"bad"}, pause_between=0, max_retries_per_item=3)

    stats = await w.run_batch()
    # bad fails 3 times, then skipped
    assert stats["failed"] == 2  # first 2 failures
    assert stats["skipped"] == 1  # final skip


@pytest.mark.asyncio
async def test_run_batch_exception_handling():
    items = [Item("err")]
    w = ErrorWorker(items=items, pause_between=0, max_retries_per_item=2)

    stats = await w.run_batch()
    assert stats["failed"] == 1
    assert stats["skipped"] == 1


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_halts_run():
    items = [Item(f"item_{i}") for i in range(100)]

    class StoppingWorker(SimpleWorker):
        async def process_item(self, item):
            result = await super().process_item(item)
            if self.stats["processed"] >= 3:
                self.stop()
            return result

    w = StoppingWorker(items=items, pause_between=0)
    stats = await w.run_batch(limit=100)
    # stop() signals after processing item 3, but item 4 may already
    # be in flight — the check happens at loop top, so expect 3 or 4
    assert stats["processed"] in (3, 4)


# ---------------------------------------------------------------------------
# Retry tracking
# ---------------------------------------------------------------------------

def test_failed_ids_tracked():
    w = SimpleWorker(items=[Item("x")], fail_ids={"x"}, pause_between=0)

    async def run():
        await w.run_batch()

    asyncio.run(run())
    assert w._failed_ids["x"] >= 1


@pytest.mark.asyncio
async def test_get_next_skips_exhausted_retries():
    items = [Item("bad"), Item("good")]
    w = SimpleWorker(items=items, fail_ids={"bad"}, pause_between=0, max_retries_per_item=1)

    stats = await w.run_batch()
    # bad skipped after 1 retry, good processed
    assert stats["processed"] == 1
    assert w.processed_items[0].id == "good"

"""Tests for khonliang_researcher.worker."""

import asyncio
from dataclasses import dataclass
from typing import Optional, Any

import pytest

from khonliang_researcher.worker import SKIP, BaseQueueWorker


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
            # _stats["processed"] is incremented AFTER process_item() returns.
            # Use + 1 to count the in-flight item so stop() fires exactly after
            # item 3 completes, giving a deterministic count of 3.
            if self._stats["processed"] + 1 >= 3:
                self.stop()
            return result

    w = StoppingWorker(items=items, pause_between=0)
    stats = await w.run_batch(limit=100)
    assert stats["processed"] == 3


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


# ---------------------------------------------------------------------------
# SKIP sentinel — voluntary decline (fr_khonliang-researcher_c59b1692)
# ---------------------------------------------------------------------------


class SkipWorker(BaseQueueWorker):
    """process_item returns SKIP for ids in ``skip_ids`` (declining them without
    counting), True otherwise. Removes items so the queue drains."""

    def __init__(self, items=None, skip_ids=None, **kwargs):
        super().__init__(**kwargs)
        self.items = list(items or [])
        self.skip_ids = set(skip_ids or [])
        self.processed_items = []

    def count_pending(self) -> int:
        return len(self.items)

    def get_next(self):
        for item in self.items:
            if self._failed_ids.get(item.id, 0) < self.max_retries_per_item:
                return item
        return None

    async def process_item(self, item):
        # A declined item still leaves the queue (mirrors the lock-contention
        # case where a sibling worker claimed it) so get_next advances.
        self.items.remove(item)
        if item.id in self.skip_ids:
            return SKIP
        self.processed_items.append(item)
        return True


def test_initial_stats_include_declined():
    assert SkipWorker().stats["declined"] == 0


@pytest.mark.asyncio
async def test_run_batch_skip_counts_as_declined_not_processed_or_failed():
    items = [Item("a"), Item("skip_me"), Item("b")]
    w = SkipWorker(items=items, skip_ids={"skip_me"}, pause_between=0)

    stats = await w.run_batch()

    assert stats["declined"] == 1
    assert stats["processed"] == 2  # a, b — the skip is NOT counted as processed
    assert stats["failed"] == 0     # ...nor as a failure
    assert stats["skipped"] == 0    # ...nor as a max-retries skip
    assert "skip_me" not in w._failed_ids  # no retry bump
    assert {i.id for i in w.processed_items} == {"a", "b"}


@pytest.mark.asyncio
async def test_run_skip_counts_as_declined_not_processed_or_failed():
    # Drive the continuous `run` loop: stop once the queue is drained by
    # patching idle to raise, so the loop exits deterministically after work.
    items = [Item("a"), Item("skip_me")]
    w = SkipWorker(items=items, skip_ids={"skip_me"}, pause_between=0, idle_poll=0)

    async def stop_when_empty():
        # After the queue drains, get_next returns None → run idles; stop then.
        while w.count_pending() > 0:
            await asyncio.sleep(0)
        w.stop()

    import asyncio as _aio

    await _aio.gather(w.run(), stop_when_empty())

    assert w._stats["declined"] == 1
    assert w._stats["processed"] == 1  # only "a"
    assert w._stats["failed"] == 0
    assert w._stats["skipped"] == 0


@pytest.mark.asyncio
async def test_skip_breaks_consecutive_failure_streak_in_run(monkeypatch):
    """The consecutive-failure ``max_failures`` 60s pause lives ONLY in the
    ``run`` loop. A SKIP must reset the failure streak like a success does — a
    (fail, decline, fail) sequence with ``max_failures=2`` must NOT trip the
    pause, because the two failures are no longer consecutive. Exercised through
    ``run`` (not run_batch, which has no pause machinery), asserting the 60s
    ``asyncio.sleep`` never fires."""

    class ScriptedWorker(BaseQueueWorker):
        """Yields a fixed script of outcomes, one per item, then stops."""

        def __init__(self, script, **kwargs):
            super().__init__(**kwargs)
            self.script = list(script)  # e.g. ["fail", "skip", "fail"]
            self._i = 0

        def count_pending(self):
            return len(self.script) - self._i

        def get_next(self):
            if self._i < len(self.script):
                return Item(f"step{self._i}")
            return None

        async def process_item(self, item):
            outcome = self.script[self._i]
            self._i += 1
            if self._i >= len(self.script):
                self.stop()  # halt run() after the last scripted step
            if outcome == "fail":
                return False
            if outcome == "skip":
                return SKIP
            return True

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def record_sleep(delay, *a, **k):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    # max_failures=2, retries high so the failures don't get bucketed to skipped.
    w = ScriptedWorker(
        ["fail", "skip", "fail"], pause_between=0, idle_poll=0,
        max_failures=2, max_retries_per_item=5,
    )
    await w.run()

    assert w._stats["failed"] == 2
    assert w._stats["declined"] == 1
    # The decline reset the streak, so max_failures (2) was never reached: no
    # 60s consecutive-failure pause fired.
    assert 60 not in sleeps


@pytest.mark.asyncio
async def test_run_batch_skip_does_not_bump_retries_or_fail():
    # In run_batch (no pause machinery), a run of declines is counted purely as
    # ``declined`` with no retry bumps and no failures.
    items = [Item(f"s{i}") for i in range(5)]
    w = SkipWorker(
        items=items, skip_ids={f"s{i}" for i in range(5)},
        pause_between=0, idle_poll=0, max_failures=1,
    )

    stats = await w.run_batch()

    assert stats["declined"] == 5
    assert stats["failed"] == 0
    assert w._failed_ids == {}

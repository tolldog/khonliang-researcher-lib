"""Generic background queue worker with retry tracking.

Subclass and implement count_pending(), get_next(), and process_item()
to create a domain-specific worker.

Features:
  - Retry tracking per item with configurable max retries
  - Consecutive failure detection with automatic pause
  - Configurable pause between items and idle polling
  - Stats tracking (processed, failed, skipped, declined, duration)
  - Batch mode (process N items then stop) or continuous mode

process_item outcomes:
  - True   → success (counts as ``processed``)
  - False  → failure (retried; counts as ``failed``, then ``skipped`` once
             ``max_retries_per_item`` is exhausted)
  - SKIP   → declined cleanly, no work done (counts as ``declined`` — NOT a
             success and NOT a failure). Return the module-level ``SKIP``
             sentinel. Use for cases like a sibling
             worker having claimed the item in a race, where counting it as
             processed over-reports throughput and counting it as failed would
             wrongly trigger retries / consecutive-failure pauses.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class _Skip:
    """Sentinel returned by ``process_item`` to decline an item without counting
    it as processed or failed. A distinct object (not ``None``/``False``) so the
    skip branch can't be reached by an accidental falsy return."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "SKIP"

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        return False


#: Return this from ``process_item`` to decline an item (see module docstring).
SKIP = _Skip()

#: Return type of ``process_item``: bool (success/failure) or the SKIP sentinel.
ProcessOutcome = Union[bool, _Skip]


class BaseQueueWorker(ABC):
    """Background worker that drains a processing queue."""

    def __init__(
        self,
        pause_between: float = 2.0,
        idle_poll: float = 30.0,
        max_failures: int = 3,
        max_retries_per_item: int = 2,
    ):
        self.pause_between = pause_between
        self.idle_poll = idle_poll
        self.max_failures = max_failures
        self.max_retries_per_item = max_retries_per_item
        self._running = False
        self._failed_ids: dict[str, int] = {}  # item_id -> retry count
        self._stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "declined": 0,
            "started_at": None,
        }

    @abstractmethod
    def count_pending(self) -> int:
        """Return number of items waiting to be processed."""

    @abstractmethod
    def get_next(self) -> Optional[Any]:
        """Get next item to process, skipping items that exceeded retries.

        The returned item must have an `id` attribute for retry tracking.
        Return None when the queue is empty.
        """

    @abstractmethod
    async def process_item(self, item: Any) -> "ProcessOutcome":
        """Process a single item.

        Return ``True`` on success, ``False`` on failure (item is retried), or
        the module-level ``SKIP`` sentinel to decline the item without counting
        it as processed or failed (see module docstring)."""

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "running": self._running,
            "pending": self.count_pending(),
        }

    async def run(self):
        """Main worker loop. Runs until stopped or queue exhausted."""
        self._running = True
        self._stats["started_at"] = time.time()
        consecutive_failures = 0

        logger.info(
            "Worker started. %d items pending.", self.count_pending()
        )

        while self._running:
            item = self.get_next()

            if item is None:
                logger.info("Queue empty. Idling for %.0fs...", self.idle_poll)
                await asyncio.sleep(self.idle_poll)
                continue

            item_id = getattr(item, "id", str(id(item)))
            item_title = getattr(item, "title", item_id)

            logger.info(
                "[%d/%d] Processing: %s",
                self._stats["processed"] + 1,
                self._stats["processed"] + self.count_pending(),
                str(item_title)[:60],
            )

            try:
                outcome = await self.process_item(item)

                if outcome is SKIP:
                    # Declined cleanly — no work done. Neither processed nor
                    # failed: don't bump retries. Reset consecutive_failures like
                    # a success does — a decline breaks a failure streak, else a
                    # (fail, decline, fail) sequence would wrongly trip the
                    # max_failures pause as if the failures were consecutive.
                    self._stats["declined"] += 1
                    consecutive_failures = 0
                    logger.info("  DECLINED (skip): %s", str(item_title)[:60])
                elif outcome:
                    self._stats["processed"] += 1
                    consecutive_failures = 0
                else:
                    self._failed_ids[item_id] = self._failed_ids.get(item_id, 0) + 1
                    if self._failed_ids[item_id] >= self.max_retries_per_item:
                        self._stats["skipped"] += 1
                        logger.warning("  SKIPPED (max retries): %s", str(item_title)[:60])
                    else:
                        self._stats["failed"] += 1
                        consecutive_failures += 1
                        logger.warning("  FAILED (retry %d): %s", self._failed_ids[item_id], str(item_title)[:60])

            except Exception as e:
                self._failed_ids[item_id] = self._failed_ids.get(item_id, 0) + 1
                if self._failed_ids[item_id] >= self.max_retries_per_item:
                    self._stats["skipped"] += 1
                    logger.warning("  SKIPPED (max retries): %s — %s", str(item_title)[:60], e)
                else:
                    self._stats["failed"] += 1
                    consecutive_failures += 1
                    logger.error("  ERROR (retry %d): %s — %s", self._failed_ids[item_id], str(item_title)[:60], e)

            if consecutive_failures >= self.max_failures:
                logger.warning(
                    "Too many consecutive failures (%d). Pausing for 60s...",
                    consecutive_failures,
                )
                await asyncio.sleep(60)
                consecutive_failures = 0

            if self._running:
                await asyncio.sleep(self.pause_between)

        logger.info(
            "Worker stopped. Processed: %d, Failed: %d, Skipped: %d, Declined: %d",
            self._stats["processed"],
            self._stats["failed"],
            self._stats["skipped"],
            self._stats["declined"],
        )

    def stop(self):
        """Signal the worker to stop after current item."""
        self._running = False

    async def run_batch(self, limit: Optional[int] = None):
        """Process up to `limit` items then stop. None = all pending."""
        self._running = True
        self._stats["started_at"] = time.time()
        count = 0

        pending = self.count_pending()
        target = min(pending, limit) if limit else pending
        logger.info("Processing %d items...", target)

        while self._running and (limit is None or count < limit):
            item = self.get_next()
            if item is None:
                break

            item_id = getattr(item, "id", str(id(item)))
            item_title = getattr(item, "title", item_id)
            count += 1

            logger.info("[%d/%d] %s", count, target, str(item_title)[:60])

            try:
                outcome = await self.process_item(item)
                if outcome is SKIP:
                    # Declined cleanly — no work done: neither processed nor
                    # failed (no retry bump). ``count`` still advances (it bounds
                    # the loop by items dequeued): a decline deliberately consumes
                    # a batch slot. Rolling ``count`` back would let a re-yielding
                    # get_next spin run_batch(limit=N) forever, a worse regression
                    # than an occasional short batch; declines are rare
                    # (lock-contention race) and the declined item leaves the
                    # queue, so slot consumption is negligible.
                    self._stats["declined"] += 1
                    logger.info("  DECLINED (skip): %s", str(item_title)[:60])
                elif outcome:
                    self._stats["processed"] += 1
                else:
                    self._failed_ids[item_id] = self._failed_ids.get(item_id, 0) + 1
                    if self._failed_ids[item_id] >= self.max_retries_per_item:
                        self._stats["skipped"] += 1
                        logger.warning("  SKIPPED: %s", str(item_title)[:60])
                    else:
                        self._stats["failed"] += 1
            except Exception as e:
                self._failed_ids[item_id] = self._failed_ids.get(item_id, 0) + 1
                if self._failed_ids[item_id] >= self.max_retries_per_item:
                    self._stats["skipped"] += 1
                    logger.warning("  SKIPPED: %s — %s", str(item_title)[:60], e)
                else:
                    self._stats["failed"] += 1
                    logger.error("  ERROR: %s", e)

            if self._running and count < target:
                await asyncio.sleep(self.pause_between)

        self._running = False
        return self._stats

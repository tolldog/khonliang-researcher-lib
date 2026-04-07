"""Generic background queue worker with retry tracking.

Subclass and implement count_pending(), get_next(), and process_item()
to create a domain-specific worker.

Features:
  - Retry tracking per item with configurable max retries
  - Consecutive failure detection with automatic pause
  - Configurable pause between items and idle polling
  - Stats tracking (processed, failed, skipped, duration)
  - Batch mode (process N items then stop) or continuous mode
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    async def process_item(self, item: Any) -> bool:
        """Process a single item. Return True on success, False on failure."""

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
                success = await self.process_item(item)

                if success:
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
            "Worker stopped. Processed: %d, Failed: %d, Skipped: %d",
            self._stats["processed"],
            self._stats["failed"],
            self._stats["skipped"],
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
                success = await self.process_item(item)
                if success:
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

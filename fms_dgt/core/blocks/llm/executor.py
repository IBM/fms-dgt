# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Callable, Iterable
import asyncio

# Third Party
from tqdm import tqdm


class AsyncLLMExecutor:
    """Shared async queue-and-worker scaffolding for LLM providers.

    Each provider supplies:
    - ``work_items``: an iterable of items to enqueue (batches or individual
      instances, depending on the provider and method).
    - ``process_item``: an async callable ``(item, update_progress) -> None``
      that performs the LLM call and writes results back into the
      ``LMBlockData`` objects it receives.  ``update_progress`` is a
      zero-argument callable that advances the progress bar; the callee
      decides *how much* to advance (1 for single-instance modes, batch_size
      for batched modes).

    The executor owns queue construction, worker spawning (bounded by
    ``call_limit``), the progress bar, and ``asyncio.gather``.  Retry logic
    belongs in ``process_item`` (decorated with ``@retry`` at the call site).
    """

    @staticmethod
    async def run(
        work_items: Iterable[Any],
        process_item: Callable[[Any, Callable[[], None]], Any],
        call_limit: int,
        total_requests: int,
        method: str,
        disable_tqdm: bool = False,
    ) -> None:
        """Drive concurrent LLM calls through a shared async queue.

        Args:
            work_items: Items to enqueue (batches or single ``LMBlockData``
                instances).
            process_item: ``async callable(item, update_progress) -> None``.
                Must write results back into the ``LMBlockData`` objects it
                receives and call ``update_progress()`` once per work item
                processed.
            call_limit: Maximum number of concurrent worker coroutines.
            total_requests: Total number of *requests* used for the progress
                bar.  For batched modes this equals the number of individual
                prompts, not the number of queue items.
            method: Human-readable method name shown in the progress bar.
            disable_tqdm: Suppress the progress bar.
        """
        queue: asyncio.Queue = asyncio.Queue()
        for item in work_items:
            queue.put_nowait(item)

        pbar = tqdm(
            total=total_requests,
            disable=disable_tqdm,
            desc=f"Running {method} requests",
        )

        async def _worker() -> None:
            while not queue.empty():
                item = await queue.get()
                await process_item(item, pbar.update)
                queue.task_done()

        n_workers = min(queue.qsize(), call_limit)
        await asyncio.gather(*[_worker() for _ in range(n_workers)], return_exceptions=True)

        pbar.close()

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List
import threading
import time

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.databuilder.generation import GenerationDataBuilder
from fms_dgt.base.task import GenerationTask
from fms_dgt.base.telemetry import Span


# ===========================================================================
#                       CONCURRENT GENERATION DATA BUILDER
# ===========================================================================
class ConcurrentGenerationDataBuilder(GenerationDataBuilder):
    """A GenerationDataBuilder that runs work items concurrently via a thread pool.

    Replaces the synchronous inner loop of GenerationDataBuilder.execute_tasks()
    with a ThreadPoolExecutor that processes mini-batches of data points in
    parallel. All other behaviour — block initialization, telemetry spans,
    structured log events, postprocessing, stall detection, task completion —
    is preserved.

    Worker allocation across tasks uses a proportional-fill policy:
    - Every incomplete task is guaranteed at least 1 worker slot.
    - Remaining slots are distributed proportionally by each task's remaining
      work (num_outputs_to_generate - produced so far).
    - If a task has fewer items remaining than its allocated slots, it only
      spawns as many futures as it has work for; freed slots flow to other
      tasks on the next recompute.
    - Allocation is recomputed on every future completion.

    Stall detection mirrors GenerationDataBuilder: a task's stall counter
    increments each time one of its futures returns zero successes. The counter
    resets on any success. The task is terminated when the counter exceeds
    max_stalled_attempts.

    The __call__ contract for subclasses:
    - Take N inputs, return M outputs (M <= N).
    - Touch nothing outside those N inputs. No shared mutable state, no
      class-level caches written during a call, no task-level accumulators.
    - Safe to invoke simultaneously from multiple threads.
    All shared state (machine_data, datastore writes, stall counters) is owned
    exclusively by execute_tasks() under per-task locks.
    """

    def __init__(
        self,
        *args: Any,
        max_workers: int = 4,
        items_per_worker: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initializes the concurrent data builder.

        Args:
            max_workers: Number of parallel worker threads. Each worker processes
                one mini-batch of items_per_worker items from a single task.
                Default: 4.
            items_per_worker: Number of data points passed to __call__ per worker
                invocation. Analogous to machine_batch_size in GenerationDataBuilder.
                The effective in-flight LM batch under steady-state load is
                max_workers * items_per_worker. Default: 4.
        """
        super().__init__(*args, **kwargs)
        self._max_workers = max_workers if isinstance(max_workers, int) and max_workers > 0 else 4
        self._items_per_worker = (
            items_per_worker if isinstance(items_per_worker, int) and items_per_worker > 0 else 4
        )
        # Per-task locks for thread-safe shared state mutation.
        self._task_locks: Dict[str, threading.Lock] = {}

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_workers(self) -> int:
        """Number of parallel worker threads."""
        return self._max_workers

    @property
    def items_per_worker(self) -> int:
        """Number of data points passed to __call__ per worker invocation."""
        return self._items_per_worker

    # ===========================================================================
    #                       WORKER ALLOCATION
    # ===========================================================================
    def _compute_worker_allocation(
        self,
        active_tasks: List[GenerationTask],
        in_flight_per_task: Dict[str, int],
    ) -> Dict[str, int]:
        """Compute how many additional workers to spawn per task.

        Uses proportional fill with a per-task floor of 1:
        - Each incomplete task gets at least 1 slot (floor guarantee).
        - Remaining slots after floor allocation are distributed proportionally
          by remaining work.
        - Returns the number of *new* futures to spawn per task (not total
          slots), accounting for already in-flight futures.

        Args:
            active_tasks: Tasks that still need work.
            in_flight_per_task: Count of currently running futures per task.

        Returns:
            Dict mapping task name to number of new futures to spawn.
        """
        total_in_flight = sum(in_flight_per_task.values())
        available_slots = self._max_workers - total_in_flight

        if available_slots <= 0 or not active_tasks:
            return {}

        # Floor: 1 slot per task, adjusted for what is already in flight.
        # A task already occupying >= 1 slot consumes its floor.
        floor_slots_needed = sum(1 for t in active_tasks if in_flight_per_task.get(t.name, 0) == 0)
        slots_after_floor = max(0, available_slots - floor_slots_needed)

        total_remaining = sum(
            max(0, t.num_outputs_to_generate - len(t.machine_data)) for t in active_tasks
        )

        spawn = {}
        for task in active_tasks:
            task_remaining = max(0, task.num_outputs_to_generate - len(task.machine_data))
            in_flight = in_flight_per_task.get(task.name, 0)

            # Floor allocation: ensure at least 1 worker if none in flight.
            floor = 1 if in_flight == 0 else 0

            # Proportional share of slots beyond the floor.
            proportional = (
                round(slots_after_floor * task_remaining / total_remaining)
                if total_remaining > 0
                else 0
            )

            new_futures = max(floor, proportional)

            # Never spawn more futures than there are items remaining.
            max_batches_needed = -(-task_remaining // self._items_per_worker)  # ceiling division
            new_futures = min(new_futures, max(0, max_batches_needed - in_flight))

            if new_futures > 0:
                spawn[task.name] = new_futures

        # Clamp total spawns to available slots (rounding may exceed).
        total_to_spawn = sum(spawn.values())
        if total_to_spawn > available_slots:
            scale = available_slots / total_to_spawn
            spawn = {name: max(1, int(count * scale)) for name, count in spawn.items()}

        return spawn

    # ===========================================================================
    #                       EXECUTE TASKS
    # ===========================================================================
    def execute_tasks(self):
        """Concurrent execution loop.

        Replaces the synchronous inner loop of GenerationDataBuilder with a
        ThreadPoolExecutor. Preserves all structured log events and telemetry
        spans emitted by the synchronous implementation.
        """

        # Load existing machine data and dataloader state.
        for task in self._tasks:
            task.machine_data = task.load_intermediate_data()
            if task.machine_data:
                self.logger.debug("Loaded %s machine-generated data", len(task.machine_data))
            task.load_dataloader_state()

        # Identify active vs already-complete tasks.
        active_tasks: List[GenerationTask] = []
        for task in self._tasks:
            if task.is_complete():
                self.logger.info(
                    "Task '%s' finished.",
                    task.name,
                    extra={
                        "event": "task_finished",
                        "task_name": task.name,
                        "reason": "already_complete",
                    },
                )
                task.finish()
            else:
                self._register_task_log_handler(task)
                self._task_locks[task.name] = threading.Lock()
                self.logger.info(
                    "Task '%s' started.",
                    task.name,
                    extra={
                        "event": "task_started",
                        "task_name": task.name,
                    },
                )
                active_tasks.append(task)

        if not active_tasks:
            return

        start_time = time.time()

        # Stall counters: incremented each time a future for a task yields zero
        # successes; reset on any success. Mirrors the synchronous model.
        consecutive_empty_futures: Dict[str, int] = {t.name: 0 for t in active_tasks}

        with Span(
            "dgt.epoch",
            self._span_writer,
            parent_span_name="dgt.run",
            epoch=self._epoch,
            active_task_names=",".join(t.name for t in active_tasks),
            active_task_count=len(active_tasks),
        ):
            self.logger.info("*" * 99)
            self.logger.info("\t\t\t\tCONCURRENT GENERATION")
            self.logger.info("*" * 99)
            self.logger.info(
                "Starting concurrent generation with %s worker(s), %s item(s) per worker, %s active task(s): %s",
                self._max_workers,
                self._items_per_worker,
                len(active_tasks),
                ", ".join(t.name for t in active_tasks),
                extra={
                    "event": "epoch_started",
                    "epoch": self._epoch,
                    "active_task_names": [t.name for t in active_tasks],
                    "active_task_count": len(active_tasks),
                    "max_workers": self._max_workers,
                    "items_per_worker": self._items_per_worker,
                },
            )

            futures: Dict[Future, str] = {}
            in_flight_per_task: Dict[str, int] = {t.name: 0 for t in active_tasks}

            executor = ThreadPoolExecutor(max_workers=self._max_workers)

            try:
                # Initial fill: spawn futures up to max_workers.
                self._spawn_futures(active_tasks, futures, in_flight_per_task, executor)

                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        task_name = futures.pop(fut)
                        in_flight_per_task[task_name] -= 1

                        task = next(t for t in self._tasks if t.name == task_name)
                        results: List[DataPoint] = fut.result()
                        successes = 0

                        # Thread-safe: all shared state mutation under per-task lock.
                        with self._task_locks[task_name]:
                            for dp in results:
                                task.save_intermediate_data(dp)
                                task.save_dataloader_state()
                                task.machine_data.append(dp)
                                successes += 1

                        # Stall detection.
                        if successes > 0:
                            consecutive_empty_futures[task_name] = 0
                        else:
                            consecutive_empty_futures[task_name] += 1

                        stalled = consecutive_empty_futures[task_name] >= self._max_stalled_attempts

                        if task.is_complete() or stalled:
                            if stalled and not task.is_complete():
                                self.logger.warning(
                                    "Task %s has not generated any data in the last %s futures, terminating task",
                                    task_name,
                                    self._max_stalled_attempts,
                                )
                            reason = "complete" if task.is_complete() else "stalled_generation"
                            self.logger.info(
                                "Task '%s' finished.",
                                task_name,
                                extra={
                                    "event": "task_finished",
                                    "task_name": task_name,
                                    "reason": reason,
                                    "produced": len(task.machine_data),
                                },
                            )
                            task.finish()
                            active_tasks = [t for t in active_tasks if t.name != task_name]
                            in_flight_per_task.pop(task_name, None)
                            consecutive_empty_futures.pop(task_name, None)
                        else:
                            self._spawn_futures(active_tasks, futures, in_flight_per_task, executor)

            finally:
                executor.shutdown(wait=True)

        # Postprocessing for all tasks that produced data.
        completed_tasks = [t for t in self._tasks if t.machine_data]
        if completed_tasks:
            self.logger.info("Launch postprocessing")
            with Span(
                "dgt.postprocessing",
                self._span_writer,
                parent_span_name="dgt.epoch",
                epoch=self._epoch,
                task_count=len(completed_tasks),
            ):
                self.execute_postprocessing(completed_tasks)
            self.logger.info(
                "Postprocessing completed",
                extra={
                    "event": "postprocessing_finished",
                    "epoch": self._epoch,
                    "task_counts": {t.name: len(t.machine_data) for t in completed_tasks},
                },
            )

        self.logger.info(
            "Epoch %s finished.",
            self._epoch,
            extra={
                "event": "epoch_finished",
                "epoch": self._epoch,
                "task_counts": {t.name: len(t.machine_data) for t in self._tasks},
            },
        )
        self.logger.info("*" * 99)
        self.logger.info("Generation took %.2fs", time.time() - start_time)

    # ===========================================================================
    #                       HELPERS
    # ===========================================================================
    def _spawn_futures(
        self,
        active_tasks: List[GenerationTask],
        futures: Dict[Future, str],
        in_flight_per_task: Dict[str, int],
        executor: ThreadPoolExecutor,
    ) -> None:
        """Compute worker allocation and submit new futures for each task."""
        spawn = self._compute_worker_allocation(active_tasks, in_flight_per_task)

        for task in active_tasks:
            count = spawn.get(task.name, 0)
            for _ in range(count):
                batch = task.get_batch_examples()
                if not batch:
                    break
                fut = executor.submit(self, self._epoch, batch)
                futures[fut] = task.name
                in_flight_per_task[task.name] = in_flight_per_task.get(task.name, 0) + 1

    def call_with_task_list(
        self, tasks: List[GenerationTask], request_idx: int
    ) -> Iterable[DataPoint]:
        """Not used by the concurrent execution loop.

        The concurrent loop submits __call__ directly to the thread pool rather
        than going through call_with_task_list. This method is retained only to
        satisfy the abstract interface and should not be called directly.
        """
        self.logger.warning(
            "call_with_task_list is not used by ConcurrentGenerationDataBuilder; "
            "use execute_tasks() to run generation."
        )
        return []

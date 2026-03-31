# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping
import contextvars
import threading
import time

# Local
from fms_dgt.base.data_objects import DataBuilderConfig, DataPoint
from fms_dgt.base.databuilder.generation import GenerationDataBuilder
from fms_dgt.base.task import GenerationTask
from fms_dgt.base.telemetry import Span
from fms_dgt.utils import init_dataclass_from_dict


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass(kw_only=True)
class ConcurrentGenerationDataBuilderConfig(DataBuilderConfig):
    """Configuration class for a GenerationDataBuilder that runs work items concurrently via a thread pool."""

    max_workers: int = 4  # Holds the LLM config for finetuning


# ===========================================================================
#                       CONCURRENT GENERATION DATA BUILDER
# ===========================================================================
class ConcurrentGenerationDataBuilder(GenerationDataBuilder):
    """A GenerationDataBuilder that runs work items concurrently via a thread pool.

    Replaces the synchronous inner loop of GenerationDataBuilder.execute_tasks()
    with a ThreadPoolExecutor that processes mini-batches of data points in
    parallel. Mirrors the full epoch loop of GenerationDataBuilder including the
    outer loop that re-queues tasks after postprocessing if their data count
    dropped below num_outputs_to_generate.

    Worker allocation across tasks uses a proportional-fill policy:
    - Every incomplete task is guaranteed at least 1 worker slot.
    - Remaining slots are distributed proportionally by each task's remaining
      work (num_outputs_to_generate - produced so far).
    - If a task has fewer items remaining than its allocated slots, it only
      spawns as many futures as it has work for; freed slots flow to other
      tasks on the next recompute.
    - Allocation is recomputed on every future completion.
    - Unused slots after proportional distribution are redistributed to tasks
      with the most remaining work, ensuring full pool utilization.

    Two levels of stall detection mirror GenerationDataBuilder:
    - Per-future stall: incremented each time a future returns zero successes;
      reset on any success. Task exits generation when counter reaches
      max_stalled_attempts.
    - Per-epoch stall: incremented each time postprocessing leaves a task with
      zero data; reset when postprocessing retains any data. Task is terminated
      when this counter reaches max_stalled_attempts (stalled_postprocessing).

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
        config: Mapping | ConcurrentGenerationDataBuilderConfig = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the concurrent data builder.

        Args:
            max_workers: Number of parallel worker threads. Each worker processes
                one batch returned by task.get_batch_examples(). The actual batch
                size is task-controlled (task.batch_size = seed_batch_size +
                machine_batch_size). Default: 4.
        """
        config: ConcurrentGenerationDataBuilderConfig = init_dataclass_from_dict(
            config, ConcurrentGenerationDataBuilderConfig
        )
        super().__init__(*args, config=config, **kwargs)

        self._max_workers = (
            config.max_workers
            if isinstance(config.max_workers, int) and config.max_workers > 0
            else 4
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
            # Use task.batch_size (seed + machine) as the actual items-per-future.
            max_batches_needed = -(-task_remaining // task.batch_size)  # ceiling division
            new_futures = min(new_futures, max(0, max_batches_needed - in_flight))

            if new_futures > 0:
                spawn[task.name] = new_futures

        # Clamp total spawns to available slots (rounding may exceed).
        total_to_spawn = sum(spawn.values())
        if total_to_spawn > available_slots:
            scale = available_slots / total_to_spawn
            spawn = {name: max(1, int(count * scale)) for name, count in spawn.items()}
            total_to_spawn = sum(spawn.values())

        # Redistribute any unused slots to tasks with the most remaining work,
        # ensuring full pool utilization when there is work to be done.
        unused = available_slots - total_to_spawn
        if unused > 0:
            # Sort tasks by remaining work descending; only tasks already in
            # spawn are eligible (tasks not in spawn have no batches to give).
            eligible = sorted(
                [t for t in active_tasks if t.name in spawn],
                key=lambda t: max(0, t.num_outputs_to_generate - len(t.machine_data)),
                reverse=True,
            )
            for task in eligible:
                if unused <= 0:
                    break
                task_remaining = max(0, task.num_outputs_to_generate - len(task.machine_data))
                in_flight = in_flight_per_task.get(task.name, 0)
                max_batches_needed = -(-task_remaining // task.batch_size)
                current = spawn[task.name]
                headroom = max(0, max_batches_needed - in_flight - current)
                add = min(unused, headroom)
                if add > 0:
                    spawn[task.name] += add
                    unused -= add

        return spawn

    # ===========================================================================
    #                       EXECUTE TASKS
    # ===========================================================================
    def execute_tasks(self):
        """Concurrent execution loop.

        Mirrors GenerationDataBuilder.execute_tasks() with a two-level loop:

        Outer loop (epochs): runs while any task is still incomplete and the
        attempt budget has not been exhausted. After each epoch's generation
        phase, postprocessing runs on all tasks that produced data. Tasks that
        are still incomplete after postprocessing re-enter the next epoch's
        generation phase. A per-epoch stall counter terminates tasks where
        postprocessing consistently destroys all data.

        Inner loop (generation phase): a ThreadPoolExecutor drains all active
        tasks concurrently. Workers are allocated proportionally with a
        per-task floor of 1. Allocation is recomputed on every future
        completion. A per-future stall counter terminates tasks that
        consistently produce zero results.
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

        # Per-future stall counters: incremented when a future returns zero
        # successes; reset on any success.
        consecutive_empty_futures: Dict[str, int] = {t.name: 0 for t in active_tasks}

        # Per-epoch stall counters: incremented when postprocessing leaves a
        # task with zero data; reset when postprocessing retains any data.
        consecutive_empty_epochs: Dict[str, int] = {t.name: 0 for t in active_tasks}

        attempt = 0

        # Outer epoch loop: re-enters when tasks remain incomplete after
        # postprocessing and the attempt budget has not been exhausted.
        while active_tasks and attempt <= self._num_attempts_to_complete:

            with Span(
                "dgt.epoch",
                self._span_writer,
                parent_span_name="dgt.run",
                epoch=self._epoch,
                active_task_names=",".join(t.name for t in active_tasks),
                active_task_count=len(active_tasks),
            ):
                self.logger.info("*" * 99)
                self.logger.info("\t\t\t\tCONCURRENT GENERATION — EPOCH %s", self._epoch)
                self.logger.info("*" * 99)
                self.logger.info(
                    "Epoch %s started: %s worker(s), %s active task(s): %s",
                    self._epoch,
                    self._max_workers,
                    len(active_tasks),
                    ", ".join(t.name for t in active_tasks),
                    extra={
                        "event": "epoch_started",
                        "epoch": self._epoch,
                        "active_task_names": [t.name for t in active_tasks],
                        "active_task_count": len(active_tasks),
                        "max_workers": self._max_workers,
                    },
                )

                # ---------------------------------------------------------------
                # Generation phase: run thread pool until all active tasks are
                # either complete or stalled.
                # ---------------------------------------------------------------
                tasks_in_generation: List[GenerationTask] = list(active_tasks)
                tasks_finished_generation: List[GenerationTask] = []

                futures: Dict[Future, str] = {}
                in_flight_per_task: Dict[str, int] = {t.name: 0 for t in tasks_in_generation}

                executor = ThreadPoolExecutor(max_workers=self._max_workers)
                try:
                    self._spawn_futures(tasks_in_generation, futures, in_flight_per_task, executor)

                    while futures:
                        attempt += 1
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)

                        for fut in done:
                            task_name = futures.pop(fut)
                            if task_name in in_flight_per_task:
                                in_flight_per_task[task_name] -= 1

                            task = next(t for t in self._tasks if t.name == task_name)
                            results: List[DataPoint] = fut.result()
                            successes = 0

                            with self._task_locks[task_name]:
                                for dp in results:
                                    task.save_intermediate_data(dp)
                                    task.save_dataloader_state()
                                    task.machine_data.append(dp)
                                    successes += 1

                            # Per-future stall detection.
                            if successes > 0:
                                consecutive_empty_futures[task_name] = 0
                            else:
                                consecutive_empty_futures[task_name] += 1

                            stalled = (
                                consecutive_empty_futures[task_name] >= self._max_stalled_attempts
                            )

                            if task.is_complete() or stalled:
                                if stalled and not task.is_complete():
                                    self.logger.warning(
                                        "Task %s has not generated any data in the last %s futures, moving to postprocessing",
                                        task_name,
                                        self._max_stalled_attempts,
                                    )
                                tasks_in_generation = [
                                    t for t in tasks_in_generation if t.name != task_name
                                ]
                                tasks_finished_generation.append(task)
                                in_flight_per_task.pop(task_name, None)
                                # Do not reset consecutive_empty_futures here:
                                # the epoch decision logic reads it to determine
                                # stalled_generation reason.
                            else:
                                self._spawn_futures(
                                    tasks_in_generation, futures, in_flight_per_task, executor
                                )

                finally:
                    executor.shutdown(wait=True)

                # ---------------------------------------------------------------
                # Postprocessing phase.
                # ---------------------------------------------------------------
                tasks_for_postproc = [t for t in tasks_finished_generation if t.machine_data]
                _counts_before: Dict[str, int] = {
                    t.name: len(t.machine_data) for t in tasks_for_postproc
                }

                if tasks_for_postproc:
                    self.logger.info("Launch postprocessing")
                    with Span(
                        "dgt.postprocessing",
                        self._span_writer,
                        parent_span_name="dgt.epoch",
                        epoch=self._epoch,
                        task_count=len(tasks_for_postproc),
                    ):
                        self.execute_postprocessing(tasks_for_postproc)

                    for task in tasks_for_postproc:
                        if task.machine_data:
                            consecutive_empty_epochs[task.name] = 0
                        else:
                            consecutive_empty_epochs[task.name] += 1

                    self.logger.info(
                        "Postprocessing completed",
                        extra={
                            "event": "postprocessing_finished",
                            "epoch": self._epoch,
                            "task_counts": {
                                t.name: {
                                    "before": _counts_before[t.name],
                                    "after": len(t.machine_data),
                                }
                                for t in tasks_for_postproc
                            },
                        },
                    )

                # ---------------------------------------------------------------
                # Decide which tasks are done vs need another epoch.
                # ---------------------------------------------------------------
                _epoch_finish_reasons: Dict[str, str] = {}
                next_active: List[GenerationTask] = []

                for task in tasks_finished_generation:
                    generation_stalled = (
                        consecutive_empty_futures.get(task.name, 0) >= self._max_stalled_attempts
                    )
                    epoch_stalled = (
                        consecutive_empty_epochs.get(task.name, 0) >= self._max_stalled_attempts
                    )

                    if task.is_complete():
                        reason = "complete"
                    elif generation_stalled:
                        reason = "stalled_generation"
                        self.logger.warning(
                            "Task %s has not generated any data in the last %s attempts, terminating",
                            task.name,
                            self._max_stalled_attempts,
                        )
                    elif epoch_stalled:
                        reason = "stalled_postprocessing"
                        self.logger.warning(
                            "Task %s has not produced any data after postprocessing in the last %s epochs, terminating",
                            task.name,
                            self._max_stalled_attempts,
                        )
                    else:
                        # Still incomplete but not stalled: needs another epoch.
                        next_active.append(task)
                        continue

                    _epoch_finish_reasons[task.name] = reason
                    self.logger.info(
                        "Task '%s' finished.",
                        task.name,
                        extra={
                            "event": "task_finished",
                            "task_name": task.name,
                            "reason": reason,
                            "produced": len(task.machine_data),
                        },
                    )
                    task.finish()

                self.logger.info(
                    "Epoch %s finished.",
                    self._epoch,
                    extra={
                        "event": "epoch_finished",
                        "epoch": self._epoch,
                        "task_counts": {
                            t.name: len(t.machine_data) for t in tasks_finished_generation
                        },
                        "finish_reasons": _epoch_finish_reasons,
                    },
                )
                self.logger.info("*" * 99)

                active_tasks = next_active
                if active_tasks and attempt <= self._num_attempts_to_complete:
                    self.logger.info(
                        "Triggering new epoch since %d task%s still pending.",
                        len(active_tasks),
                        "s are" if len(active_tasks) > 1 else " is",
                    )
                    self._epoch += 1

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
                fut = executor.submit(contextvars.copy_context().run, self, self._epoch, batch)
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

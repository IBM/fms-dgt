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

    Each task owns its full lifecycle independently. Generation and postprocessing
    both run as futures in a shared ThreadPoolExecutor. The main thread only makes
    routing decisions; it never blocks on task work.

    Per-task lifecycle (state machine):

        GENERATING ──(is_complete or gen_stalled)──→ DRAINING
        DRAINING   ──(in_flight == 0)─────────────→ PENDING_POSTPROC
        PENDING_POSTPROC ──(worker available)──────→ IN_POSTPROC
        IN_POSTPROC ──(future resolves, done)──────→ task.finish()
                    ──(future resolves, re-queue)──→ GENERATING

    Worker allocation (priority-based, recomputed on every future completion):
    - Step 1: Tasks in PENDING_POSTPROC each consume one slot (highest priority).
      A task one step from finish() is served before generation gets anything.
    - Step 2: Remaining slots are distributed to GENERATING tasks using
      proportional fill with a per-task floor of 1.

    Two levels of stall detection:
    - Per-future stall: incremented each time a gen future returns zero successes;
      reset on any success. Task exits generation when counter reaches
      max_stalled_attempts.
    - Per-epoch stall: incremented each time postprocessing leaves a task with
      zero data; reset when postprocessing retains any data. Task is terminated
      when this counter reaches max_stalled_attempts.

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
        available_slots: int,
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
            available_slots: Worker slots available for generation (caller has
                already reserved slots for postprocessing tasks).

        Returns:
            Dict mapping task name to number of new futures to spawn.
        """
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
        """Flat event loop — each task progresses through its lifecycle independently.

        State sets (per-task name strings):
            generating       — tasks currently submitting gen futures
            draining         — gen exit signalled, waiting for in-flight gen futures to land
            pending_postproc — all gen futures drained, waiting for a postproc worker slot
            in_postproc      — postproc future in flight

        futures maps each Future to (task_name, "gen"|"postproc").

        Span hierarchy emitted per task:
            dgt.task            — full task lifetime
              dgt.epoch         — one generation round (first future → last future lands)
              dgt.postprocessing — one postprocessing run
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

        task_by_name: Dict[str, GenerationTask] = {t.name: t for t in active_tasks}

        # Per-task epoch counters (for logging and postproc).
        task_epoch: Dict[str, int] = {t.name: 1 for t in active_tasks}

        # Per-future stall counters: incremented when a gen future returns zero
        # successes; reset on any success. Reset to 0 on re-queue into generating.
        empty_futures: Dict[str, int] = {t.name: 0 for t in active_tasks}

        # Per-epoch stall counters: incremented when postprocessing leaves a task
        # with zero data. Accumulates across re-queues (persistent signal).
        empty_epochs: Dict[str, int] = {t.name: 0 for t in active_tasks}

        # In-flight future counts (gen + postproc combined per task).
        in_flight: Dict[str, int] = {t.name: 0 for t in active_tasks}

        # Per-task, per-epoch attempt counters. Incremented each time a gen
        # future is submitted; attached to the dgt.epoch span at close time
        # and reset when the task re-enters generation.
        epoch_attempts: Dict[str, int] = {t.name: 0 for t in active_tasks}

        # Lifecycle state sets (hold task name strings).
        generating: set = {t.name for t in active_tasks}
        draining: set = set()  # gen exit signalled, outstanding gen futures remain
        pending_postproc: set = set()
        in_postproc: set = set()

        # futures[fut] = (task_name, "gen"|"postproc")
        futures: Dict[Future, Any] = {}

        # Global attempt counter: incremented on every completed gen future across
        # all tasks and epochs. Used solely to enforce _num_attempts_to_complete,
        # the absolute cap on total generation work for the entire run.
        attempt = 0

        # Open one dgt.task span per active task. Closed just before task.finish().
        # build_id/run_id are auto-injected from the active run_context ContextVar
        # in Span.__exit__ — no need to pass them explicitly here.
        task_spans: Dict[str, Span] = {}
        for task in active_tasks:
            span = Span(
                "dgt.task",
                self._span_writer,
                parent_span_name="dgt.run",
                task_name=task.name,
            )
            span.__enter__()
            task_spans[task.name] = span

        # Open epoch and postproc spans are tracked here; populated by
        # _allocate_and_spawn and closed in the event loop.
        epoch_spans: Dict[str, Span] = {}
        postproc_spans: Dict[str, Span] = {}

        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            self._allocate_and_spawn(
                task_by_name,
                futures,
                task_epoch,
                in_flight,
                generating,
                pending_postproc,
                in_postproc,
                executor,
                epoch_spans,
                postproc_spans,
                epoch_attempts,
            )

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done:
                    task_name, future_type = futures.pop(fut)
                    in_flight[task_name] -= 1
                    task = task_by_name[task_name]

                    # ----------------------------------------------------------
                    # Generation future completed.
                    # ----------------------------------------------------------
                    if future_type == "gen":
                        attempt += 1
                        results: List[DataPoint] = fut.result()
                        successes = 0

                        with self._task_locks[task_name]:
                            for dp in results:
                                task.save_intermediate_data(dp)
                                task.save_dataloader_state()
                                task.machine_data.append(dp)
                                successes += 1

                        if successes > 0:
                            empty_futures[task_name] = 0
                        else:
                            empty_futures[task_name] += 1

                        gen_stalled = empty_futures[task_name] >= self._max_stalled_attempts
                        budget_exhausted = attempt > self._num_attempts_to_complete

                        # Decide whether this task should exit generation.
                        should_exit_gen = task_name in generating and (
                            task.is_complete() or gen_stalled or budget_exhausted
                        )

                        if should_exit_gen:
                            if gen_stalled and not task.is_complete():
                                self.logger.warning(
                                    "Task %s has not generated any data in the last %s futures, moving to postprocessing",
                                    task_name,
                                    self._max_stalled_attempts,
                                )
                            generating.discard(task_name)
                            if in_flight[task_name] == 0:
                                # Close epoch span: no more gen futures for this epoch.
                                if task_name in epoch_spans:
                                    span = epoch_spans.pop(task_name)
                                    span.num_attempts = epoch_attempts[task_name]
                                    span.__exit__(None, None, None)
                                pending_postproc.add(task_name)
                            else:
                                draining.add(task_name)

                        elif task_name in draining and in_flight[task_name] == 0:
                            # Last in-flight gen future for a draining task landed.
                            draining.discard(task_name)
                            # Close epoch span: generation round fully drained.
                            if task_name in epoch_spans:
                                span = epoch_spans.pop(task_name)
                                span.num_attempts = epoch_attempts[task_name]
                                span.__exit__(None, None, None)
                            pending_postproc.add(task_name)

                    # ----------------------------------------------------------
                    # Postprocessing future completed.
                    # ----------------------------------------------------------
                    else:
                        in_postproc.discard(task_name)
                        epoch = task_epoch[task_name]

                        # Close postproc span.
                        if task_name in postproc_spans:
                            postproc_spans.pop(task_name).__exit__(None, None, None)

                        if task.machine_data:
                            empty_epochs[task_name] = 0
                        else:
                            empty_epochs[task_name] += 1

                        epoch_stalled = empty_epochs[task_name] >= self._max_stalled_attempts
                        gen_stalled = empty_futures[task_name] >= self._max_stalled_attempts

                        if task.is_complete() or gen_stalled or epoch_stalled:
                            if epoch_stalled and not task.is_complete():
                                self.logger.warning(
                                    "Task %s has not produced any data after postprocessing in the last %s epochs, terminating",
                                    task_name,
                                    self._max_stalled_attempts,
                                )
                            reason = (
                                "complete"
                                if task.is_complete()
                                else (
                                    "stalled_generation"
                                    if gen_stalled
                                    else "stalled_postprocessing"
                                )
                            )
                            self.logger.info(
                                "Task '%s' finished.",
                                task_name,
                                extra={
                                    "event": "task_finished",
                                    "task_name": task_name,
                                    "reason": reason,
                                    "epoch": epoch,
                                    "produced": len(task.machine_data),
                                },
                            )
                            # Close task span before finish().
                            if task_name in task_spans:
                                task_spans.pop(task_name).__exit__(None, None, None)
                            task.finish()
                            # Remove from all tracking dicts so freed slots
                            # flow to remaining tasks on the next allocation.
                            del task_by_name[task_name]
                            del in_flight[task_name]
                            del empty_futures[task_name]
                            del empty_epochs[task_name]
                            del task_epoch[task_name]
                        else:
                            # Postprocessing ran but task still needs more data.
                            # Re-enter generation with a fresh empty_futures counter.
                            task_epoch[task_name] += 1
                            empty_futures[task_name] = 0
                            in_flight[task_name] = 0
                            epoch_attempts[task_name] = 0
                            generating.add(task_name)
                            self.logger.info(
                                "Task '%s' re-entering generation (epoch %s).",
                                task_name,
                                task_epoch[task_name],
                                extra={
                                    "event": "task_requeued",
                                    "task_name": task_name,
                                    "epoch": task_epoch[task_name],
                                },
                            )

                    self._allocate_and_spawn(
                        task_by_name,
                        futures,
                        task_epoch,
                        in_flight,
                        generating,
                        pending_postproc,
                        in_postproc,
                        executor,
                        epoch_spans,
                        postproc_spans,
                        epoch_attempts,
                    )

        finally:
            executor.shutdown(wait=True)
            # Close any spans left open by an unexpected exit (e.g. exception).
            for span in postproc_spans.values():
                span.__exit__(None, None, None)
            for span in epoch_spans.values():
                span.__exit__(None, None, None)
            for span in task_spans.values():
                span.__exit__(None, None, None)

        self.logger.info("Generation took %.2fs", time.time() - start_time)

    # ===========================================================================
    #                       HELPERS
    # ===========================================================================
    def _allocate_and_spawn(
        self,
        task_by_name: Dict[str, GenerationTask],
        futures: Dict[Future, Any],
        task_epoch: Dict[str, int],
        in_flight: Dict[str, int],
        generating: set,
        pending_postproc: set,
        in_postproc: set,
        executor: ThreadPoolExecutor,
        epoch_spans: Dict[str, "Span"],
        postproc_spans: Dict[str, "Span"],
        epoch_attempts: Dict[str, int],
    ) -> None:
        """Submit new futures according to the priority allocation policy.

        Step 1 (highest priority): each task in pending_postproc consumes one
        worker slot and transitions to in_postproc. Opens a dgt.postprocessing
        span for the task.

        Step 2: remaining slots are distributed to generating tasks using
        proportional fill with a per-task floor of 1. Opens a dgt.epoch span
        the first time a gen future is submitted for a task in a given epoch.
        """
        in_flight_gen = sum(in_flight[n] for n in generating if n in in_flight)
        in_flight_pp = sum(in_flight[n] for n in in_postproc if n in in_flight)
        available = self._max_workers - in_flight_gen - in_flight_pp

        # Step 1: postprocessing futures (priority).
        for task_name in list(pending_postproc):
            if available <= 0:
                break
            task = task_by_name[task_name]
            pending_postproc.discard(task_name)
            in_postproc.add(task_name)
            in_flight[task_name] = in_flight.get(task_name, 0) + 1
            epoch = task_epoch[task_name]
            # Open dgt.postprocessing span for this postproc run.
            if task_name not in postproc_spans:
                span = Span(
                    "dgt.postprocessing",
                    self._span_writer,
                    parent_span_name="dgt.task",
                    task_name=task_name,
                    epoch=epoch,
                )
                span.__enter__()
                postproc_spans[task_name] = span
            fut = executor.submit(self.execute_postprocessing, [task], epoch)
            futures[fut] = (task_name, "postproc")
            available -= 1

        # Step 2: generation futures (proportional fill).
        if available > 0 and generating:
            generating_tasks = [task_by_name[n] for n in generating if n in task_by_name]
            in_flight_generating = {n: in_flight.get(n, 0) for n in generating}
            spawn = self._compute_worker_allocation(
                generating_tasks, in_flight_generating, available
            )
            for task in generating_tasks:
                count = spawn.get(task.name, 0)
                for _ in range(count):
                    batch = task.get_batch_examples()
                    if not batch:
                        # Seed data exhausted — treat as gen_stalled.
                        generating.discard(task.name)
                        pending_postproc.add(task.name)
                        break
                    # Open dgt.epoch span on the first future for this epoch.
                    if in_flight.get(task.name, 0) == 0 and task.name not in epoch_spans:
                        span = Span(
                            "dgt.epoch",
                            self._span_writer,
                            parent_span_name="dgt.task",
                            task_name=task.name,
                            epoch=task_epoch[task.name],
                        )
                        span.__enter__()
                        epoch_spans[task.name] = span
                    fut = executor.submit(
                        contextvars.copy_context().run, self, task_epoch[task.name], batch
                    )
                    futures[fut] = (task.name, "gen")
                    in_flight[task.name] = in_flight.get(task.name, 0) + 1
                    epoch_attempts[task.name] = epoch_attempts.get(task.name, 0) + 1

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

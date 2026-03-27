# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ConcurrentGenerationDataBuilder.

Uses a fully stubbed subclass with no LM block dependency. The stub __call__
returns a configurable number of DataPoints per invocation, allowing
deterministic control over success, failure (stall), and partial success
scenarios.
"""

# Standard
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import MagicMock
import threading

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.databuilder.concurrent import ConcurrentGenerationDataBuilder


# ===========================================================================
#                       STUBS
# ===========================================================================
@dataclass(kw_only=True)
class StubDataPoint(DataPoint):
    value: int = 0


class StubTask:
    """Minimal GenerationTask stub — no file I/O, no dataloader."""

    def __init__(self, name: str, num_outputs: int):
        self.name = name
        self._num_outputs_to_generate = num_outputs
        self.machine_data: List[StubDataPoint] = []
        self.task_card = MagicMock(build_id="build-1", run_id="run-1")
        self.log_handler = None
        self._finished = False

    @property
    def num_outputs_to_generate(self) -> int:
        return self._num_outputs_to_generate

    def is_complete(self) -> bool:
        return len(self.machine_data) >= self._num_outputs_to_generate

    def load_intermediate_data(self) -> List:
        return []

    def load_dataloader_state(self) -> None:
        pass

    def save_intermediate_data(self, dp) -> None:
        pass

    def save_dataloader_state(self) -> None:
        pass

    def get_batch_examples(self) -> List[StubDataPoint]:
        return [StubDataPoint(task_name=self.name, value=i) for i in range(4)]

    @property
    def finished(self) -> bool:
        return self._finished

    def finish(self) -> None:
        self._finished = True


class StubConcurrentBuilder(ConcurrentGenerationDataBuilder):
    """ConcurrentGenerationDataBuilder with __call__ controlled by a per-task
    response map. No blocks, no LM, no file I/O."""

    def __init__(
        self,
        tasks: List[StubTask],
        responses_per_task: Dict[str, List[List[StubDataPoint]]],
        max_workers: int = 2,
        items_per_worker: int = 4,
        max_stalled_attempts: int = 3,
    ):
        # Bypass DataBuilder.__init__ entirely — set required attributes directly.
        self._tasks = tasks
        self._blocks = []
        self._block_datastores_per_task = {}
        self._epoch = 1
        self._max_stalled_attempts = max_stalled_attempts
        self._num_attempts_to_complete = 1000000
        self._config = MagicMock(name="stub_builder", postprocessors=[])
        self._span_writer = MagicMock()
        self._span_writer.write = MagicMock()
        self._fanout_handler = MagicMock()
        self._logger = MagicMock()
        self._logger.info = MagicMock()
        self._logger.debug = MagicMock()
        self._logger.warning = MagicMock()
        self._task_locks = {}

        self._max_workers = max_workers
        self._items_per_worker = items_per_worker

        # responses_per_task[task_name] is a list of return values, one per
        # __call__ invocation for that task. Cycles when exhausted.
        self._responses: Dict[str, List] = responses_per_task
        self._call_counts: Dict[str, int] = {t.name: 0 for t in tasks}
        self._call_lock = threading.Lock()

    def __call__(self, request_idx: int, instruction_data: List[DataPoint]) -> List[DataPoint]:
        task_name = instruction_data[0].task_name if instruction_data else None
        with self._call_lock:
            count = self._call_counts.get(task_name, 0)
            responses = self._responses.get(task_name, [[]])
            result = responses[count % len(responses)]
            self._call_counts[task_name] = count + 1
        return result

    @property
    def call_counts(self) -> Dict[str, int]:
        return self._call_counts

    # Disable postprocessing and telemetry helpers that require real infrastructure.
    def execute_postprocessing(self, tasks):
        pass

    def _register_task_log_handler(self, task):
        pass

    def _unregister_task_log_handler(self, task):
        pass


# ===========================================================================
#                       HELPERS
# ===========================================================================
def make_dp(task_name: str, value: int = 0) -> StubDataPoint:
    return StubDataPoint(task_name=task_name, value=value)


# ===========================================================================
#                       CONFIG TESTS
# ===========================================================================
class TestConcurrentGenerationDataBuilderDefaults:
    def test_defaults(self):
        task = StubTask("A", num_outputs=1)
        # Explicitly pass the framework defaults to verify they are stored correctly.
        builder = StubConcurrentBuilder([task], {"A": [[]]}, max_workers=4, items_per_worker=4)
        assert builder.max_workers == 4
        assert builder.items_per_worker == 4

    def test_custom_values(self):
        task = StubTask("A", num_outputs=1)
        builder = StubConcurrentBuilder([task], {"A": [[]]}, max_workers=8, items_per_worker=16)
        assert builder.max_workers == 8
        assert builder.items_per_worker == 16


# ===========================================================================
#                       WORKER ALLOCATION TESTS
# ===========================================================================
class TestWorkerAllocation:
    def _builder(self, tasks, max_workers=4):
        responses = {t.name: [[make_dp(t.name)]] for t in tasks}
        return StubConcurrentBuilder(tasks, responses, max_workers=max_workers)

    def test_single_task_gets_all_workers(self):
        task = StubTask("A", num_outputs=100)
        builder = self._builder([task], max_workers=4)
        allocation = builder._compute_worker_allocation([task], {"A": 0})
        assert allocation.get("A", 0) <= 4
        assert allocation.get("A", 0) >= 1

    def test_floor_of_one_per_task(self):
        tasks = [StubTask("A", 100), StubTask("B", 100), StubTask("C", 100)]
        builder = self._builder(tasks, max_workers=3)
        allocation = builder._compute_worker_allocation(tasks, {"A": 0, "B": 0, "C": 0})
        # Each task must get at least 1 worker when slots equal task count.
        for t in tasks:
            assert allocation.get(t.name, 0) >= 1

    def test_total_allocation_does_not_exceed_max_workers(self):
        tasks = [StubTask("A", 1000), StubTask("B", 1000), StubTask("C", 1000)]
        builder = self._builder(tasks, max_workers=4)
        allocation = builder._compute_worker_allocation(tasks, {"A": 0, "B": 0, "C": 0})
        assert sum(allocation.values()) <= 4

    def test_no_new_futures_when_pool_full(self):
        tasks = [StubTask("A", 100), StubTask("B", 100)]
        builder = self._builder(tasks, max_workers=2)
        # Both slots already occupied.
        allocation = builder._compute_worker_allocation(tasks, {"A": 1, "B": 1})
        assert sum(allocation.values()) == 0

    def test_proportional_distribution_favours_larger_task(self):
        task_small = StubTask("S", num_outputs=10)
        task_large = StubTask("L", num_outputs=1000)
        builder = self._builder([task_small, task_large], max_workers=8)
        allocation = builder._compute_worker_allocation([task_small, task_large], {"S": 0, "L": 0})
        # Large task should get more workers than small task.
        assert allocation.get("L", 0) >= allocation.get("S", 0)

    def test_completed_task_freed_slots_flow_to_remaining(self):
        task_a = StubTask("A", num_outputs=2)
        task_b = StubTask("B", num_outputs=100)
        # Simulate task_a nearly done: 2 remaining but already in_flight=1
        task_a.machine_data = [make_dp("A")] * 1
        builder = self._builder([task_a, task_b], max_workers=4)
        allocation = builder._compute_worker_allocation([task_a, task_b], {"A": 1, "B": 0})
        # task_b should absorb the free slots.
        assert allocation.get("B", 0) > 0


# ===========================================================================
#                       EXECUTION TESTS
# ===========================================================================
class TestConcurrentExecution:
    def test_single_task_completes(self):
        task = StubTask("A", num_outputs=3)
        responses = {"A": [[make_dp("A"), make_dp("A"), make_dp("A")]]}
        builder = StubConcurrentBuilder([task], responses, max_workers=2, items_per_worker=4)
        builder.execute_tasks()
        assert task.is_complete()
        assert task.finished

    def test_multiple_tasks_both_complete(self):
        task_a = StubTask("A", num_outputs=2)
        task_b = StubTask("B", num_outputs=2)
        responses = {
            "A": [[make_dp("A"), make_dp("A")]],
            "B": [[make_dp("B"), make_dp("B")]],
        }
        builder = StubConcurrentBuilder(
            [task_a, task_b], responses, max_workers=2, items_per_worker=4
        )
        builder.execute_tasks()
        assert task_a.is_complete()
        assert task_b.is_complete()

    def test_already_complete_task_skipped(self):
        task = StubTask("A", num_outputs=2)
        # load_intermediate_data must return the pre-existing data so that
        # execute_tasks sees the task as complete before entering the active set.
        pre_existing = [make_dp("A"), make_dp("A")]
        task.load_intermediate_data = lambda: pre_existing
        responses = {"A": [[make_dp("A")]]}
        builder = StubConcurrentBuilder([task], responses, max_workers=2)
        builder.execute_tasks()
        # Finished immediately without calling __call__.
        assert builder.call_counts.get("A", 0) == 0
        assert task.finished

    def test_postprocessing_wipe_triggers_new_epoch(self):
        """If postprocessing wipes machine_data the task must re-enter generation."""
        task = StubTask("A", num_outputs=2)
        # Each __call__ returns 2 data points.
        responses = {"A": [[make_dp("A"), make_dp("A")]]}
        builder = StubConcurrentBuilder([task], responses, max_workers=1, items_per_worker=4)

        wipe_count = {"n": 0}

        def postprocessing_that_wipes_once(tasks):
            if wipe_count["n"] == 0:
                for t in tasks:
                    t.machine_data.clear()
            wipe_count["n"] += 1

        builder.execute_postprocessing = postprocessing_that_wipes_once
        builder.execute_tasks()
        # After the wipe the second epoch must top up to completion.
        assert task.is_complete()
        assert task.finished
        assert wipe_count["n"] == 2  # postprocessing ran twice (two epochs)

    def test_thread_safety_no_data_corruption(self):
        """Concurrent futures writing to the same task must not corrupt machine_data."""
        task = StubTask("A", num_outputs=20)
        # Each call returns 2 data points.
        responses = {"A": [[make_dp("A"), make_dp("A")]]}
        builder = StubConcurrentBuilder([task], responses, max_workers=4, items_per_worker=4)
        builder.execute_tasks()
        # All appends must have landed; no duplicates from race conditions.
        assert len(task.machine_data) >= 20
        assert task.is_complete()


# ===========================================================================
#                       STALL DETECTION TESTS
# ===========================================================================
class TestStallDetection:
    def test_task_terminated_after_max_stalled_futures(self):
        task = StubTask("A", num_outputs=10)
        # Always returns empty — will stall.
        responses = {"A": [[]]}
        builder = StubConcurrentBuilder(
            [task], responses, max_workers=1, items_per_worker=4, max_stalled_attempts=3
        )
        builder.execute_tasks()
        # Task should be terminated (finished) even though incomplete.
        assert task.finished
        assert not task.is_complete()

    def test_stall_counter_resets_on_success(self):
        task = StubTask("A", num_outputs=3)
        # Two empty responses then a successful one — should not stall.
        responses = {
            "A": [[], [], [make_dp("A"), make_dp("A"), make_dp("A")]],
        }
        builder = StubConcurrentBuilder(
            [task], responses, max_workers=1, items_per_worker=4, max_stalled_attempts=3
        )
        builder.execute_tasks()
        assert task.is_complete()

    def test_partial_success_resets_stall_counter(self):
        task = StubTask("A", num_outputs=2)
        # Partial success (1 of 4 items) then full success.
        responses = {
            "A": [[make_dp("A")], [make_dp("A")]],
        }
        builder = StubConcurrentBuilder(
            [task], responses, max_workers=1, items_per_worker=4, max_stalled_attempts=2
        )
        builder.execute_tasks()
        assert task.is_complete()

    def test_one_stalled_task_does_not_block_other(self):
        task_stall = StubTask("stall", num_outputs=10)
        task_ok = StubTask("ok", num_outputs=2)
        responses = {
            "stall": [[]],  # always empty
            "ok": [[make_dp("ok"), make_dp("ok")]],
        }
        builder = StubConcurrentBuilder(
            [task_stall, task_ok],
            responses,
            max_workers=2,
            items_per_worker=4,
            max_stalled_attempts=3,
        )
        builder.execute_tasks()
        assert task_ok.is_complete()
        assert task_stall.finished
        assert not task_stall.is_complete()

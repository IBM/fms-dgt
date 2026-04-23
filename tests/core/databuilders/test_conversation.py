# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for:
    - ConversationGenerationTask (get_batch_examples override, stage config properties)
    - ConversationDataBuilder (stage init, _run_single_conversation, __call__ inner parallelism)

All tests are fully stubbed: no LM blocks, no file I/O, no real dataloader.
"""

# Standard
from typing import Any, List
from unittest.mock import MagicMock, patch
import copy

# Third Party
import pytest

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    FlowControllerStep,
    Step,
)
from fms_dgt.core.databuilders.conversation.generate import ConversationDataBuilder
from fms_dgt.core.databuilders.conversation.registry import (
    _STAGE_REGISTRY,
)
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.task import ConversationTask


# ===========================================================================
#                       HELPERS
# ===========================================================================
def _blank_conversation_datapoint(task_name: str = "test") -> ConversationDataPoint:
    return ConversationDataPoint(task_name=task_name)


def _make_task(
    task_name: str = "test",
    max_turns: int = 5,
    min_turns: int = 1,
    init_stages: List[dict] | None = None,
    iter_stages: List[dict] | None = None,
    seed_batch_size: int = 4,
) -> ConversationTask:
    """Create a ConversationGenerationTask stub bypassing file I/O."""
    task = object.__new__(ConversationTask)

    # Attributes from ConversationGenerationTask.__init__
    task._max_turns = max_turns
    task._min_turns = min_turns
    task._initialization_stage_configs = init_stages or []
    task._iteration_stage_configs = iter_stages or []
    task.initialization_stages = []
    task.iteration_stages = []

    # Attributes expected by ConversationDataBuilder (name + sample_examples).
    # `name` is a read-only property backed by `_name`.
    task._name = task_name
    task._seed_batch_size = seed_batch_size
    task.sample_examples = MagicMock(return_value=[])

    return task


def _make_builder(tasks: List[Any], max_concurrent: int = 4) -> ConversationDataBuilder:
    """Create a ConversationDataBuilder stub bypassing DataBuilder.__init__."""
    builder = object.__new__(ConversationDataBuilder)

    builder._tasks = tasks
    builder._blocks = []
    builder._max_concurrent_conversations = max_concurrent
    builder._stages_initialized = set()
    builder._logger = MagicMock()

    return builder


# ===========================================================================
#                       ConversationGenerationTask TESTS
# ===========================================================================
class TestConversationGenerationTask:
    def test_get_batch_examples_returns_single_blank_context(self):
        task = _make_task("my_task", seed_batch_size=1)
        ctx = task.get_batch_examples()[0]
        assert isinstance(ctx, ConversationDataPoint)
        assert ctx.task_name == "my_task"
        assert ctx.steps == []

    def test_get_batch_examples_returns_blank_contexts(self):
        task = _make_task("my_task", seed_batch_size=3)
        batch = task.get_batch_examples()
        assert len(batch) == 3
        assert all(isinstance(c, ConversationDataPoint) for c in batch)
        assert all(c.task_name == "my_task" for c in batch)
        assert all(c.steps == [] for c in batch)

    def test_get_batch_examples_does_not_return_seed_examples(self):
        """Blank contexts must not be pre-populated with seed fields."""
        task = _make_task("my_task", seed_batch_size=2)
        batch = task.get_batch_examples()
        assert all(c.steps == [] for c in batch)
        assert all(c.branch_point is None for c in batch)

    def test_stage_config_properties(self):
        init_cfg = [{"name": "stub/init"}]
        iter_cfg = [{"name": "stub/iter1"}, {"name": "stub/iter2"}]
        task = _make_task(init_stages=init_cfg, iter_stages=iter_cfg)
        assert task.initialization_stage_configs == init_cfg
        assert task.iteration_stage_configs == iter_cfg

    def test_max_turns_min_turns(self):
        task = _make_task(max_turns=10, min_turns=3)
        assert task.max_turns == 10
        assert task.min_turns == 3

    def test_each_call_produces_unique_context_ids(self):
        task = _make_task("my_task", seed_batch_size=4)
        batch = task.get_batch_examples()
        ids = [c.conversation_id for c in batch]
        assert len(set(ids)) == 4


# ===========================================================================
#                       STAGE FIXTURES — registered per-test via unique names
# ===========================================================================
def _make_append_stage(stage_name: str, role: str, content: str) -> Stage:
    """Return a Stage instance that appends one Step without using the registry."""

    class _AppendStage(Stage):
        def __call__(self, data_points, seed_data=None, **kw):
            for data_point in data_points:
                data_point.steps.append(Step(role=role, content=content, stage_name=self._name))
            return data_points

    return _AppendStage(name=stage_name)


def _make_drop_stage(stage_name: str) -> Stage:
    """Return a Stage that drops every data point."""

    class _DropStage(Stage):
        def __call__(self, data_points, seed_data=None, **kw):
            return []

    return _DropStage(name=stage_name)


def _make_terminate_stage(stage_name: str) -> Stage:
    """Return a Stage that appends a FlowControllerStep with terminate=True."""

    class _TerminateStage(Stage):
        def __call__(self, data_points, seed_data=None, **kw):
            for data_point in data_points:
                data_point.steps.append(
                    FlowControllerStep(
                        content="terminate",
                        stage_name=stage_name,
                        terminate=True,
                    )
                )
            return data_points

    return _TerminateStage(name=stage_name)


# ===========================================================================
#                       _run_single_conversation TESTS
# ===========================================================================
class TestRunSingleConversation:
    def _run(self, task, data_point=None):
        """Run _run_single_conversation and return the result list."""
        builder = _make_builder([task])
        conversation_data_point = data_point or _blank_conversation_datapoint(task.name)
        return builder._run_single_conversation(conversation_data_point, task, seed_data=[])

    def test_init_stage_appends_step(self):
        task = _make_task(max_turns=1, min_turns=1)
        task.initialization_stages = [_make_append_stage("init", "system", "init")]
        task.iteration_stages = [_make_append_stage("iter", "assistant", "hi")]

        results = self._run(task)
        assert len(results) == 1
        roles = [step.role for step in results[0].steps]
        assert "system" in roles
        assert "assistant" in roles

    def test_dropped_in_init_returns_empty(self):
        task = _make_task(max_turns=1, min_turns=1)
        task.initialization_stages = [_make_drop_stage("dropper")]
        task.iteration_stages = []

        results = self._run(task)
        assert results == []

    def test_early_terminate_below_min_turns_returns_empty(self):
        task = _make_task(max_turns=5, min_turns=2)
        task.initialization_stages = []
        task.iteration_stages = [_make_terminate_stage("terminator")]

        results = self._run(task)
        # terminated at turn 1 which is < min_turns=2
        assert results == []

    def test_early_terminate_at_min_turns_returns_context(self):
        task = _make_task(max_turns=5, min_turns=1)
        task.initialization_stages = []
        task.iteration_stages = [_make_terminate_stage("terminator")]

        results = self._run(task)
        assert len(results) == 1
        fc_steps = [step for step in results[0].steps if step.role == "flow_controller"]
        assert fc_steps and fc_steps[-1].terminate is True

    def test_max_turns_caps_iteration(self):
        task = _make_task(max_turns=3, min_turns=1)
        task.initialization_stages = []
        call_count = [0]

        class _CountingStage(Stage):
            def __call__(self, data_points, seed_data=None, **kw):
                call_count[0] += 1
                return data_points

        task.iteration_stages = [_CountingStage(name="counter")]

        builder = _make_builder([task])
        data_point = _blank_conversation_datapoint(task.name)
        results = builder._run_single_conversation(data_point, task, seed_data=[])

        assert len(results) == 1
        assert call_count[0] == 3  # exactly max_turns iterations

    def test_dropped_in_iteration_above_min_turns_returns_context(self):
        """A drop in the middle of iteration when turn_count >= min_turns yields the conversation so far."""
        task = _make_task(max_turns=5, min_turns=1)
        task.initialization_stages = [_make_append_stage("init", "system", "init")]

        call_count = [0]

        class _DropAfterOne(Stage):
            def __call__(self, data_points, seed_data=None, **kw):
                call_count[0] += 1
                if call_count[0] > 1:
                    return []  # drop on second call
                return data_points

        task.iteration_stages = [_DropAfterOne(name="drop-after-one")]

        builder = _make_builder([task])
        data_point = _blank_conversation_datapoint(task.name)
        results = builder._run_single_conversation(data_point, task, seed_data=[])

        # First iteration completes (turn_count becomes 1 >= min_turns=1).
        # Second iteration drops — should return context from first turn.
        assert len(results) == 1

    def test_snapshot_rollback_rescued_context_ends_at_clean_turn(self):
        """Mid-turn drop rescues the snapshot (end of last complete turn), not partial state."""
        task = _make_task(max_turns=5, min_turns=1)
        task.initialization_stages = []

        # Stage 1: append an "assistant" step (completes turn 1).
        # Stage 2: append a "partial" step then drop the context (simulates mid-turn failure on turn 2).
        class _AppendAndDropOnSecondTurn(Stage):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._turn = 0

            def __call__(self, data_points, seed_data=None, **kw):
                self._turn += 1
                if self._turn == 1:
                    for data_point in data_points:
                        data_point.steps.append(
                            Step(role="assistant", content="turn1", stage_name=self._name)
                        )
                    return data_points
                # Turn 2: append a partial step then drop.
                for data_point in data_points:
                    data_point.steps.append(
                        Step(role="partial", content="dropped", stage_name=self._name)
                    )
                return []

        task.iteration_stages = [_AppendAndDropOnSecondTurn(name="stage")]

        results = self._run(task)
        assert len(results) == 1
        # Rescued context should only have the step from turn 1, not the partial step from turn 2.
        roles = [step.role for step in results[0].steps]
        assert roles == ["assistant"]
        assert "partial" not in roles

    def test_min_turns_greater_than_max_turns_raises(self):
        """ConversationGenerationTask validates min_turns <= max_turns at construction."""
        # Patch the parent __init__ so we only test ConversationGenerationTask's guard.
        with patch.object(
            ConversationTask.__bases__[0],
            "__init__",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="min_turns"):
                ConversationTask(max_turns=3, min_turns=5)

    def test_multi_context_forking_all_completed(self):
        """A stage that forks one context into two should produce two completed conversations."""
        task = _make_task(max_turns=1, min_turns=1)
        task.initialization_stages = []

        class _ForkStage(Stage):
            def __call__(self, data_points, seed_data=None, **kw):
                forked = []
                for data_point in data_points:
                    data_point.steps.append(
                        Step(role="assistant", content="branch_a", stage_name=self._name)
                    )
                    fork = copy.deepcopy(data_point)
                    fork.steps[-1] = Step(
                        role="assistant", content="branch_b", stage_name=self._name
                    )
                    forked.extend([data_point, fork])
                return forked

        task.iteration_stages = [_ForkStage(name="fork")]

        results = self._run(task)
        assert len(results) == 2
        contents = {results[0].steps[-1].content, results[1].steps[-1].content}
        assert contents == {"branch_a", "branch_b"}


# ===========================================================================
#                       __call__ TESTS
# ===========================================================================
class TestConversationDataBuilderCall:
    def _builder_with_stages(
        self,
        task: Any,
        init_stages: List[Stage],
        iter_stages: List[Stage],
        max_concurrent: int = 4,
    ) -> ConversationDataBuilder:
        builder = _make_builder([task], max_concurrent=max_concurrent)
        # Pre-load stages so _init_stages() is not invoked (avoids registry lookup).
        task.initialization_stages = init_stages
        task.iteration_stages = iter_stages
        builder._stages_initialized.add(task.name)
        return builder

    def test_empty_instruction_data_returns_empty(self):
        task = _make_task()
        builder = _make_builder([task])
        result = builder(request_idx=0, instruction_data=[])
        assert result == []

    def test_successful_conversations_all_returned(self):
        task = _make_task(max_turns=1, min_turns=1)
        builder = self._builder_with_stages(
            task,
            init_stages=[],
            iter_stages=[_make_append_stage("iter", "assistant", "hello")],
            max_concurrent=4,
        )
        data_points = [_blank_conversation_datapoint(task.name) for _ in range(4)]
        results = builder(request_idx=0, instruction_data=data_points)
        assert len(results) == 4

    def test_dropped_conversations_omitted(self):
        task = _make_task(max_turns=1, min_turns=1)
        builder = self._builder_with_stages(
            task,
            init_stages=[_make_drop_stage("dropper")],
            iter_stages=[],
            max_concurrent=4,
        )
        data_points = [_blank_conversation_datapoint(task.name) for _ in range(3)]
        results = builder(request_idx=0, instruction_data=data_points)
        assert results == []

    def test_seed_data_fetched_once_per_call(self):
        task = _make_task(max_turns=1, min_turns=1, seed_batch_size=3)
        builder = self._builder_with_stages(
            task,
            init_stages=[],
            iter_stages=[_make_append_stage("iter", "assistant", "hi")],
        )
        contexts = [_blank_conversation_datapoint(task.name) for _ in range(4)]
        builder(request_idx=0, instruction_data=contexts)
        task.sample_examples.assert_called_once_with(k=3)

    def test_max_concurrent_conversations_caps_inner_pool(self):
        """Inner pool size must be min(max_concurrent, len(instruction_data))."""
        task = _make_task(max_turns=1, min_turns=1)
        builder = self._builder_with_stages(
            task,
            init_stages=[],
            iter_stages=[_make_append_stage("iter", "assistant", "hi")],
            max_concurrent=2,
        )
        data_points = [_blank_conversation_datapoint(task.name) for _ in range(10)]
        # Just verify it completes without error and returns all 10.
        results = builder(request_idx=0, instruction_data=data_points)
        assert len(results) == 10

    def test_stages_initialized_once_per_task(self):
        """After first __call__, stages must not be re-initialized for the same task."""
        task = _make_task(max_turns=1, min_turns=1)
        # Register a stage so _init_stages() can resolve it.
        stage_name = "_test_once_stage"
        _STAGE_REGISTRY[stage_name] = type(
            "_TestOnceStage",
            (Stage,),
            {"__call__": lambda self, dp, **kw: dp},
        )
        task._iteration_stage_configs = [{"name": stage_name}]
        task._initialization_stage_configs = []

        builder = _make_builder([task], max_concurrent=2)

        data_points = [_blank_conversation_datapoint(task.name) for _ in range(2)]
        builder(request_idx=0, instruction_data=data_points)
        assert task.name in builder._stages_initialized

        # Second call must not re-initialize (stages list length stays the same).
        stage_count_after_first = len(task.iteration_stages)
        builder(request_idx=1, instruction_data=data_points)
        assert len(task.iteration_stages) == stage_count_after_first

        # Clean up registry entry.
        del _STAGE_REGISTRY[stage_name]

    def test_multi_task_each_task_initialized_independently(self):
        task_a = _make_task("task_a", max_turns=1, min_turns=1)
        task_b = _make_task("task_b", max_turns=1, min_turns=1)
        builder = _make_builder([task_a, task_b], max_concurrent=2)
        builder._stages_initialized.add("task_a")
        builder._stages_initialized.add("task_b")
        task_a.initialization_stages = []
        task_b.initialization_stages = []
        task_a.iteration_stages = [_make_append_stage("iter_a", "assistant", "a")]
        task_b.iteration_stages = [_make_append_stage("iter_b", "assistant", "b")]

        data_points_a = [_blank_conversation_datapoint("task_a") for _ in range(2)]
        data_points_b = [_blank_conversation_datapoint("task_b") for _ in range(2)]

        results_a = builder(request_idx=0, instruction_data=data_points_a)
        results_b = builder(request_idx=0, instruction_data=data_points_b)

        assert len(results_a) == 2
        assert len(results_b) == 2
        # task_a conversations have the task_a stage's step; task_b has task_b's.
        assert all(results_a[i].steps[-1].stage_name == "iter_a" for i in range(2))
        assert all(results_b[i].steps[-1].stage_name == "iter_b" for i in range(2))

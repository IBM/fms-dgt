# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Set
import contextvars
import copy

# Local
from fms_dgt.base.databuilder.concurrent import ConcurrentGenerationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    FlowControllerStep,
)
from fms_dgt.core.databuilders.conversation.registry import get_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.task import ConversationTask


@register_data_builder("core/conversation")
class ConversationDataBuilder(ConcurrentGenerationDataBuilder):
    """Runs the conversation stage pipeline concurrently.

    Two levels of parallelism:

    Outer level (inherited from ConcurrentGenerationDataBuilder): a thread
    pool allocates worker slots across active tasks proportionally. Each
    outer worker receives a batch of blank ConversationDataPoint objects and
    calls __call__.

    Inner level (this class): __call__ runs each conversation in the batch
    concurrently via a second thread pool capped at max_concurrent_conversations.
    This keeps the LM executor queue saturated even when conversations have
    variable length (some finish in 2 turns, others run to max_turns).

    Seed data for ICL is fetched once per __call__ invocation via
    task.sample_examples() — thread-safe, does not advance the main dataloader.

    Inheritance chain:
        DataBuilder
        └── GenerationDataBuilder          (synchronous epoch loop)
            └── ConcurrentGenerationDataBuilder   (thread pool, task allocation)
                └── ConversationDataBuilder        (stage pipeline, inner parallelism)
    """

    TASK_TYPE = ConversationTask

    def __init__(
        self,
        *args: Any,
        max_concurrent_conversations: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize ConversationDataBuilder.

        Args:
            max_concurrent_conversations: Maximum number of conversations to
                run in parallel within a single __call__ invocation. Controls
                the inner thread pool size. Default: 8.
        """
        super().__init__(*args, **kwargs)
        self._max_concurrent_conversations = (
            max_concurrent_conversations
            if isinstance(max_concurrent_conversations, int) and max_concurrent_conversations > 0
            else 100
        )
        # Track which tasks have had their stages initialized.
        self._stages_initialized: Set[str] = set()

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_concurrent_conversations(self) -> int:
        """Maximum conversations running in parallel within one __call__."""
        return self._max_concurrent_conversations

    # ===========================================================================
    #                       STAGE INITIALIZATION
    # ===========================================================================
    def _init_stages(self, task: ConversationTask, **kwargs) -> None:
        """Resolve stage config dicts to Stage instances for a task.

        Called once per task on first __call__. Stages receive all databuilder
        blocks as keyword arguments so they can acquire LM providers,
        validators, etc. by name.

        Args:
            task: The task whose stage configs should be resolved.
        """
        block_kwargs: Dict[str, Any] = {b.name: b for b in self._blocks}

        task.initialization_stages = [
            self._build_stage(cfg, block_kwargs, kwargs)
            for cfg in task.initialization_stage_configs
        ]
        task.iteration_stages = [
            self._build_stage(cfg, block_kwargs, kwargs) for cfg in task.iteration_stage_configs
        ]
        task.termination_stages = [
            self._build_stage(cfg, block_kwargs, kwargs) for cfg in task.termination_stage_configs
        ]
        self._stages_initialized.add(task.name)

    def _build_stage(
        self, config: Dict, block_kwargs: Dict[str, Any], addtl_kwargs: Dict[str, Any]
    ) -> Stage:
        """Instantiate one stage from its config dict.

        Args:
            config: Dict with at least a "name" key. All other keys are
                passed as kwargs to the Stage subclass __init__.
            block_kwargs: Block instances keyed by block name.
            addtl_kwargs: Additional optional kwargs.

        An optional ``blocks`` key in the config dict maps constructor argument
        names to block names defined in the databuilder, allowing stages to
        receive specific blocks under specific parameter names:

            iteration_stages:
              - name: lm/assistant/naive
                blocks:
                  generator: fast_generator
                  validator: lm_judge

        Explicit ``blocks`` mappings are merged over the default injection of
        all databuilder blocks, so they take precedence. If ``blocks`` is
        absent, all databuilder blocks are passed as before (backwards
        compatible).

        Returns:
            An initialized Stage instance.
        """
        cfg = dict(config)
        name = cfg.pop("name")

        # Resolve explicit blocks: mapping from constructor arg name to block name.
        stage_blocks = {}
        for arg_name, block_name in cfg.pop("blocks", {}).items():
            if block_name not in block_kwargs:
                raise ValueError(
                    f"Stage '{name}' references block '{block_name}' "
                    f"(as '{arg_name}') which is not defined in the databuilder. "
                    f"Available blocks: {sorted(block_kwargs)}"
                )
            stage_blocks[arg_name] = block_kwargs[block_name]

        # Resolve inline block pointers: cfg keys whose value is a block name
        # (e.g. `generator: generator` in the stage YAML) are replaced with the
        # actual block object. This avoids a duplicate-keyword error when the same
        # name also appears in the all-blocks injection below.
        for key in list(cfg.keys()):
            if isinstance(cfg[key], str) and cfg[key] in block_kwargs:
                stage_blocks[key] = block_kwargs[cfg.pop(key)]

        # Inject all databuilder blocks, but skip any already resolved above so
        # we don't pass the same kwarg twice.
        remaining_blocks = {k: v for k, v in block_kwargs.items() if k not in stage_blocks}

        stage_cls = get_stage(name)
        return stage_cls(name=name, **remaining_blocks, **stage_blocks, **cfg, **addtl_kwargs)

    # ===========================================================================
    #                       SINGLE CONVERSATION RUNNER
    # ===========================================================================
    def _run_single_conversation(
        self,
        data_point: ConversationDataPoint,
        task: ConversationTask,
        seed_data: List[ConversationDataPoint],
    ) -> List[ConversationDataPoint]:
        """Run the full initialization + iteration + termination loop for one conversation.

        Called from the inner thread pool — must not touch any shared mutable
        state outside `data_point`. seed_data is read-only; fetched once per
        __call__ invocation and shared across all inner workers.

        Stages are the unit of computation. Each stage receives the current
        list of live data points, may append steps, discard data points, or fork
        data points into multiple (e.g., for DPO branching or rollout), and
        returns the resulting list. The databuilder threads each stage's
        output into the next stage as input — it does not interpret the
        contents.

        Args:
            data_point: Blank ConversationDataPoint created by task.get_batch_examples().
            task: The task providing stage lists and turn bounds.
            seed_data: ICL examples sampled once per __call__, shared read-only.

        Returns:
            List of completed ConversationDataPoint objects produced from this
            starting data point. Empty list if all data points were discarded or
            min_turns was not reached on any surviving data_point.
        """
        # Phase 1: initialization stages (run once per conversation).
        # All data points in the list are threaded through each stage together.
        data_points = [data_point]
        for stage in task.initialization_stages:
            data_points = stage(data_points, seed_data=seed_data)
            if not data_points:
                return []

        # Phase 2: iteration loop.
        # `live` holds data points still accumulating turns.
        # `completed` accumulates data points that finished with turn_count >=
        # min_turns (terminated by flow_controller or exhausted at max_turns).
        # Data points dropped by a stage or terminated before min_turns are
        # silently discarded.
        # Stages are the unit of computation: each stage receives the full live
        # list and returns whatever it decides (append steps, discard, fork).
        self.logger.info(
            "[%s] conversation %s initialized — starting iteration loop (max_turns=%d)",
            task.name,
            data_point.conversation_id[:8],
            task.max_turns,
            extra={
                "event": "conversation_initialized",
                "task_name": task.name,
                "conversation_id": data_point.conversation_id,
                "max_turns": task.max_turns,
            },
        )
        live = data_points
        completed: List[ConversationDataPoint] = []
        turn_count = 0

        while live:
            # Snapshot each live data point at turn entry (Section 13.4).
            # If a data point is dropped mid-turn and turn_count >= min_turns,
            # we rescue from the snapshot (end of turn N-1) rather than the
            # partial mid-turn state. This guarantees rescued data points always
            # end at a complete turn boundary, which downstream SFT/DPO
            # serializers require.
            snapshots = {id(data_point): copy.deepcopy(data_point) for data_point in live}

            for stage in task.iteration_stages:
                previous = live
                live = stage(live, seed_data=seed_data)

                # Data points the stage chose not to continue (omitted from
                # return). Rescue from the turn-entry snapshot so the saved
                # data point ends at the last complete turn, not mid-turn.
                live_ids = {id(data_point) for data_point in live}
                for data_point in previous:
                    if id(data_point) not in live_ids:
                        if turn_count >= task.min_turns:
                            completed.append(snapshots[id(data_point)])
                            self.logger.info(
                                "[%s] conversation %s dropped mid-turn at turn %d/%d — rescued at turn %d",
                                task.name,
                                data_point.conversation_id[:8],
                                turn_count,
                                task.max_turns,
                                turn_count - 1,
                                extra={
                                    "event": "conversation_completed",
                                    "task_name": task.name,
                                    "conversation_id": data_point.conversation_id,
                                    "turns": turn_count - 1,
                                    "reason": "stage_drop",
                                },
                            )
                        else:
                            self.logger.debug(
                                "[%s] conversation %s dropped at turn %d (below min_turns=%d)",
                                task.name,
                                data_point.conversation_id[:8],
                                turn_count,
                                task.min_turns,
                                extra={
                                    "event": "conversation_dropped",
                                    "task_name": task.name,
                                    "conversation_id": data_point.conversation_id,
                                    "turn": turn_count,
                                    "reason": "stage_drop_below_min_turns",
                                },
                            )

                if not live:
                    break

                # Drain flow_controller-terminated data points out of the live
                # pool. Only FlowControllerStep sets terminate=True; no other
                # stage uses this signal. Terminated data points completed this
                # turn cleanly, so they are appended as-is (not from snapshot).
                still_running = []
                for data_point in live:
                    fc_steps = [
                        step for step in data_point.steps if isinstance(step, FlowControllerStep)
                    ]
                    if fc_steps and fc_steps[-1].terminate:
                        if turn_count >= task.min_turns:
                            completed.append(data_point)
                            self.logger.info(
                                "[%s] conversation %s completed at turn %d/%d (flow controller)",
                                task.name,
                                data_point.conversation_id[:8],
                                turn_count,
                                task.max_turns,
                                extra={
                                    "event": "conversation_completed",
                                    "task_name": task.name,
                                    "conversation_id": data_point.conversation_id,
                                    "turns": turn_count,
                                    "reason": "flow_controller",
                                },
                            )
                        else:
                            self.logger.debug(
                                "[%s] conversation %s terminated at turn %d (below min_turns=%d)",
                                task.name,
                                data_point.conversation_id[:8],
                                turn_count,
                                task.min_turns,
                                extra={
                                    "event": "conversation_dropped",
                                    "task_name": task.name,
                                    "conversation_id": data_point.conversation_id,
                                    "turn": turn_count,
                                    "reason": "flow_controller_below_min_turns",
                                },
                            )
                    else:
                        still_running.append(data_point)
                live = still_running

            # Update turn count now that turn has completed
            turn_count += 1

            # Log turn completion for all still-live contexts.
            for ctx in live:
                self.logger.debug(
                    "[%s] conversation %s turn %d/%d complete",
                    task.name,
                    data_point.conversation_id[:8],
                    turn_count,
                    task.max_turns,
                    extra={
                        "event": "turn_completed",
                        "task_name": task.name,
                        "conversation_id": data_point.conversation_id,
                        "turn": turn_count,
                        "max_turns": task.max_turns,
                    },
                )

            if turn_count >= task.max_turns:
                for data_point in live:
                    self.logger.info(
                        "[%s] conversation %s completed at turn %d/%d (max turns)",
                        task.name,
                        data_point.conversation_id[:8],
                        turn_count,
                        task.max_turns,
                        extra={
                            "event": "conversation_completed",
                            "task_name": task.name,
                            "conversation_id": data_point.conversation_id,
                            "turns": turn_count,
                            "reason": "max_turns",
                        },
                    )
                completed.extend(live)
                live = []

        # Phase 3: termination stages (run on all successfully completed contexts).
        # Termination stages receive all contexts that made it through iteration
        # with turn_count >= min_turns. They can perform final validation, add
        # metadata, discard contexts, or fork contexts.
        if completed and task.termination_stages:
            self.logger.debug(
                "[%s] running termination stages on %d completed conversation(s)",
                task.name,
                len(completed),
                extra={
                    "event": "termination_stages_started",
                    "task_name": task.name,
                    "num_completed": len(completed),
                },
            )
            for stage in task.termination_stages:
                completed = stage(completed, seed_data=seed_data)
                if not completed:
                    self.logger.debug(
                        "[%s] all conversations dropped by termination stage",
                        task.name,
                        extra={
                            "event": "all_conversations_dropped",
                            "task_name": task.name,
                        },
                    )
                    break

        return completed

    # ===========================================================================
    #                       __call__
    # ===========================================================================
    def __call__(
        self,
        request_idx: int,
        instruction_data: List[ConversationDataPoint],
    ) -> List[ConversationDataPoint]:
        """Process a batch of conversations concurrently.

        Called by the outer thread pool in ConcurrentGenerationDataBuilder.
        Each item in instruction_data is a blank ConversationDataPoint produced
        by ConversationGenerationTask.get_batch_examples(). All items belong
        to the same task.

        Seed data for ICL is fetched once here via task.sample_examples() —
        thread-safe, does not advance the main dataloader — then shared
        read-only across all inner workers.

        Args:
            request_idx: Epoch index passed through from execute_tasks().
            instruction_data: Batch of blank ConversationDataPoint objects.

        Returns:
            Completed ConversationDataPoint objects. Dropped conversations are
            omitted; callers should expect M <= N results.
        """
        if not instruction_data:
            return []

        task_map = {t.name: t for t in self._tasks}

        # Initialize stages and pre-fetch seed data once per unique task.
        # Seed data is shared read-only across all inner workers for that task.
        seed_data_map: Dict[str, List[ConversationDataPoint]] = {}
        for task_name in {data_point.task_name for data_point in instruction_data}:
            task = task_map[task_name]
            if task.name not in self._stages_initialized:
                self._init_stages(task)
            seed_data_map[task_name] = task.sample_examples(k=task.seed_batch_size)

        # Resolve task and seed data on this thread before submitting futures.
        # Workers receive pre-computed values rather than looking them up
        # themselves: (a) seed fetching happens once per task, not once per
        # worker; (b) _run_single_conversation stays a pure stage runner with
        # no dependency on builder internals; (c) missing task_name blows up
        # here, before any futures are in flight.
        inner_workers = min(self._max_concurrent_conversations, len(instruction_data))
        with ThreadPoolExecutor(max_workers=inner_workers) as executor:
            # Each future gets its own copy_context() snapshot. A single DataPoint
            # object cannot be entered concurrently — sharing one across futures
            # raises "cannot enter context: already entered".
            futures = [
                executor.submit(
                    contextvars.copy_context().run,
                    self._run_single_conversation,
                    data_point,
                    task_map[data_point.task_name],
                    seed_data_map[data_point.task_name],
                )
                for data_point in instruction_data
            ]
            return [result for f in futures for result in f.result()]

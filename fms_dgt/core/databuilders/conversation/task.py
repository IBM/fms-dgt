# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.task import GenerationTask
from fms_dgt.core.databuilders.conversation.data_objects import (
    BranchPoint,
    ConversationDataPoint,
    Step,
)
from fms_dgt.core.databuilders.conversation.stages.base import Stage


class ConversationTask(GenerationTask):
    """A generation task that drives the conversation stage pipeline.

    Extends GenerationTask with the stage configuration and turn-bound
    parameters needed by ConversationDataBuilder. All base task
    infrastructure (datastores, dataloader, checkpointing, telemetry) is
    inherited unchanged.

    Stage lists are resolved from the stage registry at databuilder init
    time, not here. The task holds their raw YAML config dicts.
    """

    INPUT_DATA_TYPE = ConversationDataPoint
    OUTPUT_DATA_TYPE = ConversationDataPoint

    # Default batch size for conversation tasks. Flat generation tasks default
    # seed_batch_size to min(100, num_outputs_to_generate), which causes all
    # conversations to be dispatched in a single future and nothing is written
    # until the last one finishes. A small default ensures incremental writes.
    # Users can override via runner_config.seed_batch_size in the task YAML.
    DEFAULT_SEED_BATCH_SIZE = 5

    def __init__(
        self,
        *args,
        max_turns: int = 10,
        min_turns: int = 1,
        initialization_stages: List[Dict] | None = None,
        iteration_stages: List[Dict] | None = None,
        termination_stages: List[Dict] | None = None,
        **kwargs: Any,
    ):
        """Initialize a ConversationGenerationTask.

        Args:
            max_turns: Maximum number of iteration loop cycles before the
                conversation is terminated regardless of flow_signal. Acts
                as a backstop; the flow controller is the primary termination
                signal.
            min_turns: Minimum number of completed iteration cycles required
                before a conversation can be yielded. Conversations that
                terminate before this floor are discarded.
            initialization_stages: List of stage config dicts run once before
                the iteration loop. Each dict must have at least a "name" key.
            iteration_stages: List of stage config dicts run in order on each
                loop iteration until flow_signal.terminate is set or max_turns
                is reached.
            termination_stages: List of stage config dicts run on conversations
                that successfully complete the iteration loop (reaching min_turns).
                Each dict must have at least a "name" key.
        """
        # Inject a conversation-appropriate seed_batch_size default before the
        # base class resolves runner_config, but only if the user did not
        # explicitly set one. kwargs may contain a runner_config dict or nothing.
        runner_config = kwargs.get("runner_config", {}) or {}
        if isinstance(runner_config, dict) and "seed_batch_size" not in runner_config:
            runner_config = dict(runner_config)
            runner_config["seed_batch_size"] = self.DEFAULT_SEED_BATCH_SIZE
            kwargs["runner_config"] = runner_config

        super().__init__(*args, **kwargs)

        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if min_turns < 1:
            raise ValueError(f"min_turns must be >= 1, got {min_turns}")
        if min_turns > max_turns:
            raise ValueError(f"min_turns ({min_turns}) must be <= max_turns ({max_turns})")

        self._max_turns = max_turns
        self._min_turns = min_turns
        self._initialization_stage_configs: List[Dict] = initialization_stages or []
        self._iteration_stage_configs: List[Dict] = iteration_stages or []
        self._termination_stage_configs: List[Dict] = termination_stages or []

        # Resolved Stage instances — populated by ConversationDataBuilder._init_stages().
        self.initialization_stages: List[Stage] = []
        self.iteration_stages: List[Stage] = []
        self.termination_stages: List[Stage] = []

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def max_turns(self) -> int:
        """Maximum iteration loop cycles before forced termination."""
        return self._max_turns

    @property
    def min_turns(self) -> int:
        """Minimum completed iterations required before a conversation can be yielded."""
        return self._min_turns

    @property
    def initialization_stage_configs(self) -> List[Dict]:
        """Raw YAML config dicts for initialization stages."""
        return self._initialization_stage_configs

    @property
    def iteration_stage_configs(self) -> List[Dict]:
        """Raw YAML config dicts for iteration stages."""
        return self._iteration_stage_configs

    @property
    def termination_stage_configs(self) -> List[Dict]:
        """Raw YAML config dicts for termination stages."""
        return self._termination_stage_configs

    def instantiate_input_example(self, **kwargs: Any) -> ConversationDataPoint:
        """Deserialize a seed example from JSONL, reconstructing typed Step subclasses.

        The base implementation passes kwargs directly to the dataclass constructor,
        which leaves `steps` as a list of raw dicts. This override deserializes each
        step dict via Step.from_dict() so stages can query steps by role as typed objects.
        """
        task_name = kwargs.pop("task_name", self.name)
        raw_steps = kwargs.pop("steps", [])
        steps = [Step.from_dict(s) if isinstance(s, dict) else s for s in raw_steps]
        raw_branch = kwargs.pop("branch_point", None)
        if isinstance(raw_branch, dict):
            chosen = raw_branch.pop("chosen_response", None)
            if isinstance(chosen, dict):
                chosen = Step.from_dict(chosen)
            branch_point = BranchPoint(chosen_response=chosen, **raw_branch)
        else:
            branch_point = raw_branch
        return ConversationDataPoint(
            task_name=task_name, steps=steps, branch_point=branch_point, **kwargs
        )

    def instantiate_output_example(self, **kwargs: Any) -> ConversationDataPoint:
        """Deserialize a saved output example from JSONL.

        Delegates to instantiate_input_example since the deserialization logic
        (Step.from_dict, BranchPoint reconstruction) is identical for both.
        """
        return self.instantiate_input_example(**kwargs)

    def get_batch_examples(self) -> List[ConversationDataPoint]:
        """Returns blank ConversationDataPoint objects as generation inputs.

        Overrides GenerationTask.get_batch_examples(). For conversation
        generation the framework workload is a set of blank contexts to run
        through the stage pipeline, not seed examples. Seed examples are
        fetched separately via sample_examples() inside _run_single_conversation.

        Returns one blank context per seed_batch_size slot so the framework
        loop allocates the same number of conversation slots as it would
        seed example slots for a flat generation task.
        """
        return [ConversationDataPoint(task_name=self.name) for _ in range(self._seed_batch_size)]

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Local
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.databuilders.conversation.generate import ConversationDataBuilder
from fms_dgt.core.databuilders.conversation.task import ConversationTask
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.task import (
    ConversationToolCallingStatelessTask,
)


@register_data_builder("tool_calling/multi_turn/stateless")
class ConversationToolCallingStatelessDataBuilder(ConversationDataBuilder):
    """Data builder for generating multi-turn tool calling conversations.

    This builder generates synthetic multi-turn tool calling conversations using a
    stateless approach, creating realistic interactions between users and AI assistants
    that involve planning and executing tool calls.

    Key Features:
        - Persona-driven generation with diverse user personalities and roles
        - Multi-turn planning with complex tool call dependencies
        - Support for nested tool calls where outputs feed into subsequent calls
        - Tool namespace management for organizing tools from multiple sources
        - LM-based tool simulation for generating synthetic tool outputs
        - Automatic quality filtering and error hint generation
        - Flexible output formatting (multi-turn or single-turn)

    Architecture:
        The builder follows a stage-based generation pipeline:
        1. Initialization: Sample personas and generate initial scenarios
        2. Iteration: For each turn, plan tool calls, generate user messages,
           verify validity, execute tools, and summarize responses
        3. Termination: Filter results and add error hints where applicable

    Configuration:
        Task configuration supports:
        - tool_registry: Tool definitions from multiple namespaces
        - tool_engine: LM-based simulator for tool execution
        - separate_context: Keep tool contexts isolated (default: True)
        - has_nested: Allow nested/dependent tool calls (default: False)
        - min_plan_length/max_plan_length: Control plan complexity (1-8)
        - min_turns/max_turns: Control conversation length

    Example:
        See tasks/public/tool_calling/multi_turn/stateless/toolmind/conversation.yaml
        for a complete task configuration example.

    Attributes:
        TASK_TYPE: ConversationToolCallingStatelessTask class for task handling
    """

    TASK_TYPE: ConversationToolCallingStatelessTask = ConversationToolCallingStatelessTask

    def _init_stages(self, task: ConversationTask, **kwargs) -> None:
        """Initialize generation stages with task configuration.

        Args:
            task: The tool calling task containing configuration parameters.
        """
        # Type narrowing to access tool-calling-specific properties
        assert isinstance(task, ConversationToolCallingStatelessTask)

        super()._init_stages(
            task,
            **{
                **kwargs,
                "tool_registry": task.tool_registry,
                "tool_engine": task.tool_engine,
                "separate_context": task.separate_context,
                "has_nested": task.has_nested,
                "min_plan_length": task.min_plan_length,
                "max_plan_length": task.max_plan_length,
            },
        )

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

    This builder generates synthetic tool calling conversations using an
    agentic approach, creating realistic interactions between users and
    AI assistants with tool usage.
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

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
import random
import sys

# Local
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter
from fms_dgt.core.databuilders.conversation.constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    TOOL_ROLE,
    USER_ROLE,
)
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    ToolCallStep,
    ToolResultStep,
)
from fms_dgt.core.databuilders.conversation.task import ConversationTask
from fms_dgt.core.databuilders.conversation.utils import get_first_step_of_type
from fms_dgt.core.tools.constants import TOOL_NAME, TOOL_PARAMETERS, TOOL_RESULT
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
    ToolUserStep,
)


# ===========================================================================
#                       TASK
# ===========================================================================
class ConversationToolCallingStatelessTask(ConversationTask):
    """Task for generating synthetic multi-turn tool calling data.

    This task manages the configuration and execution of multi-turn tool
    calling conversation generation, including tool handler setup, nested
    tool calls, hallucination checking, and context separation.

    Attributes:
        INPUT_DATA_TYPE: Expected input data type (ToolMultiTurnData).
        OUTPUT_DATA_TYPE: Output data type (ToolMultiTurnData).
    """

    def __init__(
        self,
        *args: Any,
        has_nested: bool = False,
        separate_context: bool = True,
        min_plan_length: int = 1,
        max_plan_length: int = 8,
        **kwargs: Any,
    ):
        """Initialize the MultiTurnSdgToolTask.

        Args:
            *args: Positional arguments passed to parent.
            tool_handler: Configuration dict for initializing the ToolHandler.
                         Must be provided.
            has_nested: Whether to allow nested tool calls.
            separate_context: Whether to keep tool contexts separate.
            min_turns: Minimum number of conversation turns.
            **kwargs: Additional keyword arguments passed to parent.

        Raises:
            ValueError: If tool_handler is not specified.
        """
        super().__init__(*args, **kwargs)

        self._separate_context = separate_context
        self._has_nested = has_nested
        self._min_plan_length = min_plan_length
        self._max_plan_length = max_plan_length

        # Disable limit on string to integer conversions
        sys.set_int_max_str_digits(0)

    @property
    def separate_context(self):
        """ """
        return self._separate_context

    @property
    def has_nested(self):
        """ """
        return self._has_nested

    @property
    def min_plan_length(self):
        """ """
        return self._min_plan_length

    @property
    def max_plan_length(self):
        """ """
        return self._max_plan_length


@register_formatter("tool_calling/formatters/multi_turn")
class ToolCallingMultiTurnFormatter(Formatter):
    """Formatter for multi-turn tool calling conversations.

    Formats tool calling data into a structure suitable for model training,
    including messages, available tools, and namespace information.
    """

    def apply(self, data: ConversationDataPoint, *args, **kwargs):
        """Format multi-turn tool calling data.

        Args:
            data: The tool calling conversation data to format.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary containing:
                - messages: List of conversation messages
                - tools: List of available tool definitions (shuffled)
                - namespace: Tool namespace identifier
        """
        tool_info = get_first_step_of_type(data.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        # Shuffle tools to avoid position bias
        tools = [tool.to_dict() for tool in tool_info.tools]
        random.shuffle(tools)

        # Extract conversation messages
        messages = []
        for turn in data.steps:
            role = None
            content = turn.content
            if isinstance(turn, ToolUserStep):
                role = USER_ROLE
            elif isinstance(turn, AssistantStep | ToolCallStep):
                role = ASSISTANT_ROLE
            elif isinstance(turn, ToolResultStep):
                role = TOOL_ROLE
                if isinstance(content, dict):
                    content = content[TOOL_RESULT]
            #
            if role is not None:
                messages.append({ROLE_KEY: role, CONTENT_KEY: content})

        return {"messages": messages, "tools": tools}


@register_formatter("tool_calling/formatters/single_turn")
class ToolCallingSingleTurnFormatter(Formatter):
    """Formatter for single-turn tool calling with planning.

    Formats tool calling data into single-turn exchanges that include both
    user messages and corresponding plan/tool call responses. This format
    is useful for training models on tool planning tasks.
    """

    def apply(self, data: ConversationDataPoint, *args, **kwargs):
        """Format single-turn tool calling data with plans.

        Args:
            data: The tool calling conversation data to format.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary containing:
                - messages: Interleaved user and plan messages
                - tools: List of tool definitions (name and parameters only, shuffled)
                - namespace: Tool namespace identifier

        Raises:
            AssertionError: If user and plan message counts don't match.
        """
        tool_info = get_first_step_of_type(data.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        # Shuffle tools to avoid position bias
        tools = [tool.to_dict(keep_keys=[TOOL_NAME, TOOL_PARAMETERS]) for tool in tool_info.tools]
        random.shuffle(tools)

        # Separate user messages and plan messages
        user_messages, plan_messages = [], []
        for turn in data.steps:
            if isinstance(turn, ToolUserStep):
                user_messages.append(turn.content)
            elif isinstance(turn, ToolPlanStep):
                d_form = {
                    ROLE_KEY: ToolCallStep.role,
                    CONTENT_KEY: [tc.to_dict() for tc in turn.plan or []],
                }
                plan_messages.append(d_form)

        # Extract conversation messages
        user_messages, plan_messages = [], []
        for turn in data.steps:
            if isinstance(turn, ToolUserStep):
                d_form = {ROLE_KEY: USER_ROLE, CONTENT_KEY: turn.content}
                user_messages.append(d_form)
            elif isinstance(turn, ToolPlanStep):
                d_form = {
                    ROLE_KEY: ASSISTANT_ROLE,
                    CONTENT_KEY: [tc.to_dict() for tc in turn.plan or []],
                }
                plan_messages.append(d_form)

        assert len(user_messages) == len(
            plan_messages
        ), "Different number of planning steps to user steps"

        # Interleave user and plan messages
        messages = [
            msg for u_msg, p_msg in zip(user_messages, plan_messages) for msg in [u_msg, p_msg]
        ]

        return {"messages": messages, "tools": tools}

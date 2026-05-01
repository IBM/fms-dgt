# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import deepcopy
from typing import Any, Dict, List

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    ToolCallStep,
    ToolResultStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_first_step_of_type,
    get_last_step_of_type,
)
from fms_dgt.core.tools.constants import TOOL_CALL_ID, TOOL_RESULT
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.base import ToolEngine
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger, from_dict


# ===========================================================================
#                       MAIN CLASSES
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/execute")
class ToolCallingExecutionStage(Stage):
    """Tool execution engine for tool calling conversations.

    Executes planned tool calls in sequence, handling nested tool call
    resolution by substituting output references with actual values.
    """

    def __init__(
        self,
        *args,
        tool_engine: ToolEngine,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # set tool engine
        self._tool_engine = tool_engine

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        """ """

        assert len(data_points) == 1
        data_point = data_points[0]

        dgt_logger.info("Tool call execution")

        # Get scenario, last user step and next set of tool calls
        last_plan_step = get_last_step_of_type(data_point.steps, ToolPlanStep)
        assert isinstance(last_plan_step, ToolPlanStep)
        remaining_tool_calls = (last_plan_step.plan or []) + []

        if get_first_step_of_type(data_point.steps, ToolCallStep) is None:
            self._tool_engine.setup(data_point.conversation_id)

        while remaining_tool_calls:
            # Get first remaining tool call
            latest_tool_call = remaining_tool_calls.pop(0)

            # Substitute variables in the latest tool call with appropriate values
            substituted_tool_call = self._apply_substitution(data_point, latest_tool_call)
            if substituted_tool_call is not None:
                tool_result = next(
                    iter(
                        self._tool_engine.execute(
                            session_id=data_point.conversation_id,
                            tool_calls=[substituted_tool_call],
                        )
                    )
                )
                if tool_result.result is not None:
                    data_point.steps.extend(
                        [
                            ToolCallStep(content=substituted_tool_call.to_dict()),
                            ToolResultStep(content=tool_result.to_dict()),
                        ]
                    )
                else:
                    dgt_logger.info("Tool call execution failed")
                    return []
            else:
                dgt_logger.info("Tool call substitution failed")
                return []

        dgt_logger.info("Tool call substitution succeeded")

        return [data_point]

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def _apply_substitution(
        self, data_point: ConversationDataPoint, tool_call: ToolCall
    ) -> ToolCall | None:

        def _get_variables(d: Any):
            if isinstance(d, dict):
                return [var for k, v in d.items() for var in _get_variables(k) + _get_variables(v)]
            elif isinstance(d, (list, tuple)):
                return [var for v in d for var in _get_variables(v)]
            elif isinstance(d, str) and any([d.startswith(k) for k in tool_results.keys()]):
                return [d]
            else:
                return []

        def _substitute_variables(d: Any, m: Dict):
            if isinstance(d, dict):
                return {
                    _substitute_variables(k, m): _substitute_variables(v, m) for k, v in d.items()
                }
            elif isinstance(d, (list, tuple)):
                return type(d)([_substitute_variables(v, m) for v in d])
            elif isinstance(d, str) and any([d.startswith(k) for k in tool_results.keys()]):
                if d not in m:
                    raise ValueError(f"Could not find required variable {d} in mapping {m}")
                return m[d]
            else:
                return d

        # Initialize variables
        tool_call = deepcopy(tool_call)

        # Create map of tool call id -> tool call result from tool result steps
        tool_results = {
            step.content.get(TOOL_CALL_ID): step.content.get(TOOL_RESULT)
            for step in data_point.steps
            if isinstance(step, ToolResultStep)
            and isinstance(step.content, dict)
            and isinstance(step.content.get(TOOL_CALL_ID), str)
        }

        # Extract variables in the tool call
        variables = set(_get_variables(tool_call.arguments))
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        # Replace variables in the tool call with appropriate values from the previous tool results
        if variables:
            if all([isinstance(v, dict) for v in tool_results.values()]):
                try:
                    mapping = {var: from_dict(tool_results, var) for var in variables}
                    tool_call.arguments = _substitute_variables(tool_call.arguments, mapping)
                    is_valid = validate_tool_calls(
                        tool_calls=[tool_call.to_dict()],
                        conversation_history=None,
                        tools=tool_info.tools,
                        allow_nested=False,
                        check_arg_question_overlap=False,
                        log_failures=True,
                    )
                    if not is_valid:
                        return None
                except Exception:
                    return None

        return tool_call

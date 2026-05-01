# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List
import json

# Local
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.databuilders.conversation.constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    USER_ROLE,
)
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    Step,
    ToolCallStep,
    ToolResultStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_first_step_of_type,
    get_instruction_role,
)
from fms_dgt.core.tools.constants import TOOL_CALL_ARGS, TOOL_NAME
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
    ToolUserStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/filter")
class ToolCallingConversationFilter(Stage):
    def __init__(
        self,
        *args,
        generator: LMProvider,
        **kwargs,
    ):
        """Noise filter for tool calling conversations"""
        # initialize parent
        super().__init__(*args, **kwargs)

        self._lm = generator

        # prompts
        self._instr_prompt, self._user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(
                        Path(__file__).parent.parent,
                        "prompts",
                        "filter",
                        role + ".txt",
                    )
                ),
            )
            for role in ["instructions", "user"]
        ]

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        """Generate tool calling plans for conversations.

        Args:
            data_points: List of data points to generate plans for.
            seed_data: Seed examples for in-context learning.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List of data points with generated plans added.
        """
        assert len(data_points) == 1
        data_point = data_points[0]

        dgt_logger.info("Running LLM filter on conversation")

        #
        conversation, tool_calls = _get_ordered_conversation(data_point.steps)

        # sanity check execution results
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        is_valid = validate_tool_calls(
            tool_calls=tool_calls,
            conversation_history=None,
            tools=tool_info.tools,
            allow_nested=False,
            check_arg_question_overlap=False,
            # log_failures=True,
        )

        #
        if not is_valid:
            return []

        prompt_kwargs = {
            "tools": json.dumps([t.to_dict() for t in tool_info.tools], indent=4),
            "reasoning": _reasoning_not_built_in(self._lm.model_id_or_path),
            "conversation": json.dumps(conversation, indent=4),
        }

        #
        messages = [
            {
                ROLE_KEY: get_instruction_role(self._lm.model_id_or_path),
                CONTENT_KEY: self._instr_prompt.encode(prompt_kwargs),
            },
            {
                ROLE_KEY: USER_ROLE,
                CONTENT_KEY: self._user_prompt.encode(prompt_kwargs),
            },
        ]
        lm_inputs = [{"input": messages, "data": data_point}]

        lm_output = next(
            iter(self._lm(lm_inputs, disable_tqdm=False, method=LMProvider.CHAT_COMPLETION))
        )

        return [data_point] if self._parse_results(lm_output, prompt_kwargs) else []

    def _parse_results(self, prediction: dict, prompt_kwargs: dict):
        #
        reasoning = prompt_kwargs["reasoning"]
        results = prediction["result"]
        results = results if isinstance(results, list) else [results]

        votes = {True: [], False: []}
        for res in results:
            text = (res or dict()).get(CONTENT_KEY) or ""
            reasoning = text.split("</thought>")[0].split("<thought>")[-1].strip()
            determination = text.split("</pass>")[0].split("<pass>")[-1].strip()
            if (reasoning and (not reasoning.strip())) or (not determination.strip()):
                continue
            satisfied = determination.lower().strip().strip('"') == "yes"
            votes[satisfied].append(reasoning)

        strict = not votes[False]
        return strict


###
#
###


def _get_ordered_conversation(steps: List[Step]):
    groups, tool_calls, plans, plan_tool_calls = [], [], [], []
    for turn in steps:
        if isinstance(turn, ToolUserStep):
            groups.append([])
        #
        if isinstance(turn, ToolUserStep):
            groups[-1].append({ROLE_KEY: USER_ROLE, CONTENT_KEY: turn.content})
        elif isinstance(turn, ToolCallStep):
            groups[-1].append(
                {
                    ROLE_KEY: ASSISTANT_ROLE,
                    CONTENT_KEY: json.dumps(turn.content),
                }
            )
            tool_calls.append(
                {k: v for k, v in turn.content.items() if k in [TOOL_NAME, TOOL_CALL_ARGS]}
            )
        elif isinstance(turn, ToolResultStep):
            groups[-1].append(
                {
                    ROLE_KEY: turn.role,
                    CONTENT_KEY: (
                        turn.content if isinstance(turn.content, str) else json.dumps(turn.content)
                    ),
                }
            )
        elif isinstance(turn, ToolPlanStep):
            plan_tool_calls.extend([tc.to_dict() for tc in turn.plan or []])
            plans.append(
                {
                    ROLE_KEY: ASSISTANT_ROLE,
                    CONTENT_KEY: json.dumps(
                        [
                            tc.to_dict(keep_keys=[TOOL_NAME, TOOL_CALL_ARGS])
                            for tc in turn.plan or []
                        ],
                        indent=4,
                    ),
                }
            )

    if not tool_calls:
        assert len(groups) == len(plans)
        groups = [groups[i] + [plans[i]] for i in range(len(groups))]
    conversation = [msg for group in groups for msg in group]

    return conversation, (tool_calls or plan_tool_calls)


def _reasoning_not_built_in(model: str):
    if "gpt" in model:
        return False
    else:
        return True

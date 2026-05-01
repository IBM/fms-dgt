# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Dict, List
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
    AssistantStep,
    ConversationDataPoint,
    ToolCallStep,
    ToolResultStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_first_step_of_type,
    get_instruction_role,
    get_last_step_of_type,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolUserStep,
)
from fms_dgt.utils import dgt_logger


# ===========================================================================
#                       MAIN CLASSES
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/summarize")
class ToolCallingSummarizationGenerator(Stage):
    r""" """

    def __init__(
        self,
        *args,
        generator: LMProvider,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # set tool handler and lm
        self._lm = generator

        # initialize prompt template
        self._instr_prompt, self._user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(Path(__file__).parent.parent, "prompts", "summarize", role + ".txt")
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
        """ """
        assert len(data_points) == 1
        data_point = data_points[0]

        # Build LM inputs
        lm_inputs = []
        for data_point in data_points:
            lm_inputs.append(
                {
                    "input": self._build_prompt(data_point),
                    "data": data_point,
                    "gen_kwargs": {
                        "n": 1,
                        "temperature": 0.001,
                    },
                }
            )

        # Invoke LM
        dgt_logger.info("Summarizing tool call results")
        lm_output = next(
            iter(self._lm(lm_inputs, disable_tqdm=True, method=LMProvider.CHAT_COMPLETION))
        )

        output = self._parse_summary(lm_output)

        dgt_logger.info(
            f"Tool call summarization generation {'succeeded' if output is not None else 'failed'}",
        )

        return [output] if output is not None else []

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def _parse_summary(self, prediction: Dict) -> ConversationDataPoint | None:
        data_point: ConversationDataPoint = prediction["data"]
        #
        text = (
            ((prediction["result"] or dict()).get(CONTENT_KEY) or "")
            .strip()
            .split("</summary>")[0]
            .split("<summary>")[-1]
            .strip()
        )

        # Remove extra spaces around text
        summary = text.strip()

        # Validate summary
        # - Does not contain question
        # - Does not contain any tool names
        is_valid = summary and ("?" not in summary)  # no questions

        if not is_valid:
            dgt_logger.debug("Summarization failed to produce valid text")
            return

        data_point.steps.append(
            AssistantStep(
                content=summary,
            )
        )

        return data_point

    def _build_prompt(self, data_point: ConversationDataPoint):

        # Get information for current data point
        tool_info = get_first_step_of_type(steps=data_point.steps, tgt_class=ToolInfoStep)
        user_step = get_last_step_of_type(steps=data_point.steps, tgt_class=ToolUserStep)
        assert isinstance(user_step, ToolUserStep)

        last_steps = []
        for step in reversed(data_point.steps):
            if step == user_step:
                break
            if isinstance(step, ToolCallStep | ToolResultStep):
                last_steps.append(
                    {
                        ROLE_KEY: ASSISTANT_ROLE,
                        CONTENT_KEY: (
                            step.content.get("result")
                            if isinstance(step, ToolResultStep)
                            else step.content
                        ),
                    }
                )
        last_steps.reverse()

        # Build prompt
        prompt_kwargs = {
            "tools": json.dumps([t.to_dict() for t in tool_info.tools or []], indent=4),
            "request": user_step.content,
            "steps": json.dumps(last_steps, indent=4),
        }
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

        return messages

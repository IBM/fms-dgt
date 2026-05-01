# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Dict, List
import json
import random

# Local
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.core.blocks.llm import LMProvider
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
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_all_steps_of_type,
    get_first_step_of_type,
    get_instruction_role,
)
from fms_dgt.core.tools.constants import TOOL_CALL_ARGS, TOOL_NAME, TOOL_PARAMETERS
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolUserStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger, try_parse_json_string


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/error_hint")
class ToolCallingInteractionErrorHint(Stage):
    def __init__(
        self,
        *args,
        generator: LMProvider,
        max_errors: int = 3,
        min_errors: int = 1,
        error_p: float = 1.0,
        **kwargs,
    ):
        """Error prediction step for tool calling conversations"""
        # initialize parent
        super().__init__(*args, **kwargs)

        self._lm = generator

        # set vars
        self._max_errors = max_errors
        self._min_errors = min_errors
        self._error_p = error_p

        # prompts
        self._tc_instr_prompt, self._tc_user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(
                        Path(__file__).parent.parent,
                        "prompts",
                        "error_hints",
                        "tc_" + role + ".txt",
                    )
                ),
            )
            for role in ["instructions", "user"]
        ]
        self._err_instr_prompt, self._err_user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(
                        Path(__file__).parent.parent,
                        "prompts",
                        "error_hints",
                        "error_" + role + ".txt",
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

        dgt_logger.info("Beginning tool call error step")

        # determine steps for errors
        steps = get_all_steps_of_type(data_point.steps, ToolCallStep)
        chosen_ids = sorted(
            random.choices(
                list(range(len(steps))),
                k=random.randint(self._min_errors, self._max_errors),
            )
        )
        error_steps = (
            [steps[i] for i in chosen_ids] if self._error_p >= random.uniform(0, 1) else []
        )

        # iterate until complete
        while error_steps:
            self._get_errors(data_point, error_steps.pop(0))

        # return
        return data_points

    def _get_errors(self, data_point: ConversationDataPoint, error_step: ToolCallStep):
        #
        lm_tc_inputs = [
            {
                "input": self._get_tc_prompt(data_point, error_step),
                "gen_kwargs": {"logprobs": True},
            }
        ]

        dgt_logger.info("Making tool call predictions for error step")
        lm_output = next(
            iter(self._lm(lm_tc_inputs, disable_tqdm=False, method=LMProvider.CHAT_COMPLETION))
        )

        # extract errors
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        errored_tc = self._parse_tool_call_if_incorrect(data_point, lm_output, tool_info.tools)

        if errored_tc is None:
            dgt_logger.info("Predicted tool call was correct during error contruction")
            return

        lm_err_inputs = [
            {
                "input": self._get_error_prompt(
                    data_point,
                    errored_tc,
                    {
                        k: v
                        for k, v in error_step.content.items()
                        if k in [TOOL_NAME, TOOL_CALL_ARGS]
                    },
                ),
                "gen_kwargs": {"logprobs": True},
            }
        ]

        #
        dgt_logger.info("Making error predictions")
        lm_output = next(
            iter(self._lm(lm_err_inputs, disable_tqdm=False, method=LMProvider.CHAT_COMPLETION))
        )

        #
        error_steps = self._parse_hints(lm_output, errored_tc)

        if error_steps is not None:
            for i in range(len(data_point.steps)):
                if data_point.steps[i] == error_step:
                    data_point.steps = data_point.steps[:i] + error_steps + data_point.steps[i:]
                    break

    def _get_tc_prompt(
        self,
        data_point: ConversationDataPoint,
        error_step: ToolCallStep,
    ):
        steps = []
        for step in data_point.steps:
            if step == error_step:
                break
            steps.append(step)

        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        tools = tool_info.tools + []
        random.shuffle(tools)

        conversation = []
        for step in steps:
            if isinstance(step, ToolUserStep):
                conversation.append({ROLE_KEY: USER_ROLE, CONTENT_KEY: step.content})
            elif isinstance(step, ToolCallStep):
                conversation.append(
                    {ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: json.dumps(step.content)}
                )
            elif isinstance(step, ToolResultStep):
                conversation.append(
                    {
                        ROLE_KEY: TOOL_ROLE,
                        CONTENT_KEY: (
                            step.content
                            if isinstance(step.content, str)
                            else json.dumps(step.content)
                        ),
                    }
                )
            elif isinstance(step, AssistantStep):
                conversation.append({ROLE_KEY: step.role, CONTENT_KEY: step.content})

        # Add to LM inputs
        prompt_kwargs = {
            "tools": json.dumps([t.to_dict() for t in tools], indent=4),
            "date": tool_info.date,
        }

        messages = [
            {
                ROLE_KEY: get_instruction_role(self._lm.model_id_or_path),
                CONTENT_KEY: self._tc_instr_prompt.encode(prompt_kwargs).strip(),
            }
        ] + conversation

        return messages

    def _parse_tool_call_if_incorrect(
        self, data_point: ConversationDataPoint, prediction: Dict, tools: List[Tool]
    ):
        def _equiv_tcs(d1: Dict, d2: Dict):
            if type(d1) is not type(d2) and d1 != d2:
                return False
            elif isinstance(d1, dict):
                if len(d1) != len(d2):
                    return False
                return all(
                    [
                        k1 == k2 and _equiv_tcs(v1, v2)
                        for (k1, v1), (k2, v2) in zip(
                            sorted(d1.items(), key=lambda x: x[0]),
                            sorted(d2.items(), key=lambda x: x[0]),
                        )
                    ]
                )
            elif isinstance(d1, (list, tuple)):
                if len(d1) != len(d2):
                    return False
                return all([_equiv_tcs(v1, v2) for v1, v2 in zip(d1, d2)])
            elif isinstance(d1, str) and d1:
                wrds1, wrds2 = d1.lower().split(), d2.lower().split()
                return len(set(wrds1).intersection(set(wrds2))) / (len(set(wrds1)) or 1) > 0.66
            return d1 == d2

        results = prediction["result"]
        results = results if isinstance(results, list) else [results]
        logprobs = ((prediction or dict()).get("addtl") or dict()).get(
            "token_logprobs", [[{None: 0}]]
        )

        opts = []
        for res_i, res in enumerate(results):
            text = ((res or dict()).get(CONTENT_KEY) or "").strip()

            parsed_tc = try_parse_json_string(text)

            if not (parsed_tc and isinstance(parsed_tc, dict)):
                continue

            if isinstance(parsed_tc, dict) and parsed_tc.get(TOOL_PARAMETERS):
                # seems to be a common error, not super useful for us...
                parsed_tc[TOOL_CALL_ARGS] = parsed_tc.pop(TOOL_PARAMETERS)

            is_valid = validate_tool_calls(
                tool_calls=[parsed_tc],
                conversation_history=None,
                tools=tools,
                allow_nested=False,
                require_nested=False,
                check_arg_question_overlap=False,
            )
            if not is_valid:
                continue

            #
            if any(pl_text in text.lower() for pl_text in ["placeholder", "example"]) or any(
                any(
                    isinstance(arg, str) and arg.startswith(pair[0]) and arg.endswith(pair[1])
                    for arg in parsed_tc[TOOL_CALL_ARGS].values()
                )
                for pair in [("{{", "}}"), ("<", ">")]
            ):
                continue

            #
            parsed_tc = {k: v for k, v in parsed_tc.items() if k in [TOOL_NAME, TOOL_CALL_ARGS]}
            all_tool_calls = [
                {k: v for k, v in step.content.items() if k in [TOOL_NAME, TOOL_CALL_ARGS]}
                for step in data_point.steps
                if isinstance(step, ToolCallStep)
            ]
            if any([_equiv_tcs(parsed_tc, tc) for tc in all_tool_calls]):
                continue

            scores = [next(iter(lp.values())) for lp in logprobs[res_i]]
            score = sum([score for score in scores if isinstance(score, (float, int))])
            opts.append((score, parsed_tc))

        best_opt = max(opts, key=lambda x: x[0]) if opts else None
        if not best_opt:
            # failure case
            return None

        return best_opt[1]

    def _get_error_prompt(
        self,
        data_point: ConversationDataPoint,
        predicted_tool_call: Dict,
        gold_tool_call: Dict,
    ):
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        # Add to LM inputs
        prompt_kwargs = {
            "tools": json.dumps(
                [
                    tool.to_dict()
                    for tool in tool_info.tools
                    if tool.name
                    in [gold_tool_call.get(TOOL_NAME), predicted_tool_call.get(TOOL_NAME)]
                ],
                indent=4,
            ),
            "correct_tool_call": json.dumps(gold_tool_call, indent=4),
            "incorrect_tool_call": json.dumps(predicted_tool_call, indent=4),
        }

        messages = [
            {
                ROLE_KEY: get_instruction_role(self._lm.model_id_or_path),
                CONTENT_KEY: self._err_instr_prompt.encode(prompt_kwargs).strip(),
            },
            {
                ROLE_KEY: USER_ROLE,
                CONTENT_KEY: self._err_user_prompt.encode(prompt_kwargs).strip(),
            },
        ]

        return messages

    def _parse_hints(
        self,
        prediction: dict,
        tool_call: dict,
    ):
        results = prediction["result"]
        results = results if isinstance(results, list) else [results]
        logprobs = ((prediction or dict()).get("addtl") or dict()).get(
            "token_logprobs", [[{None: 0}]]
        )

        opts = []
        for res_i, res in enumerate(results):
            text = (
                ((res or dict()).get(CONTENT_KEY) or "")
                .strip()
                .split("</hint>")[0]
                .split("<hint>")[-1]
                .strip()
            )

            if not text.strip():
                continue

            scores = [next(iter(lp.values())) for lp in logprobs[res_i]]
            score = sum([score for score in scores if isinstance(score, (float, int))])
            opts.append((score, text))

        best_opt = max(opts, key=lambda x: x[0]) if opts else None
        if not best_opt:
            # failure case
            return None

        return [ToolCallStep(content=tool_call), ToolResultStep(content=best_opt[1])]

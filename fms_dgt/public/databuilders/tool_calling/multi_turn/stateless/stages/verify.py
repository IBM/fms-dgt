# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Any, Dict, List
import json
import random

# Local
from fms_dgt.base.prompt import JinjaPromptTemplate
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.databuilders.conversation.constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    USER_ROLE,
)
from fms_dgt.core.databuilders.conversation.data_objects import ConversationDataPoint
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_all_steps_of_type,
    get_first_step_of_type,
    get_instruction_role,
)
from fms_dgt.core.tools.constants import (
    TOOL_CALL_ARGS,
    TOOL_NAME,
    TOOL_NAMESPACE,
    TOOL_PARAMETERS,
)
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.utils import extract_first_tool_call
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
    ToolUserStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    extract_largest_connected_component,
    get_leaves,
    normalize_str,
    syntax_check,
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger, try_parse_json_string


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/verifier")
class ToolCallingInteractionVerifier(Stage):
    """Verification stage for tool calling interactions.

    Verifies that generated tool call plans match user requests and
    validates structural properties like nested tool calls and argument usage.
    """

    def __init__(
        self,
        *args,
        generator: LMProvider,
        has_nested: bool = True,
        separate_context: bool = True,
        **kwargs,
    ):
        """Initialize the ToolCallingInteractionVerifier.

        Args:
            *args: Positional arguments passed to parent.
            lm: Language model provider for verification.
            tool_handler: Handler for managing tools.
            has_nested: Whether to enforce nested tool call structure.
            min_component_size: Minimum size for connected components.
            **kwargs: Additional keyword arguments passed to parent.
        """
        # initialize parent
        super().__init__(*args, **kwargs)

        self._lm = generator

        self._instr_prompt, self._user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(Path(__file__).parent.parent, "prompts", "verify", role + ".txt")
                ),
            )
            for role in ["instructions", "user"]
        ]

        self._has_nested = has_nested
        self._separate_context = separate_context

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        r""" """
        assert len(data_points) == 1
        data_point = data_points[0]

        # Get scenario
        prompt_kwargs = _get_verifier_prompt_kwargs(data_point, self._has_nested)
        messages = [
            {
                ROLE_KEY: get_instruction_role(self._lm.model_id_or_path),
                CONTENT_KEY: self._instr_prompt.encode(prompt_kwargs).strip(),
            },
            {
                ROLE_KEY: USER_ROLE,
                CONTENT_KEY: self._user_prompt.encode(prompt_kwargs).strip(),
            },
        ]

        lm_inputs = [
            {
                "input": messages,
                "gen_kwargs": {"logprobs": True},
                "reference": data_point,
            }
        ]

        # Invoke LM
        dgt_logger.info("Verifying generated scenario plan")
        lm_output = next(
            iter(self._lm(lm_inputs, disable_tqdm=True, method=LMProvider.CHAT_COMPLETION))
        )

        # Parse
        output = self._parse_verifier_output(lm_output)
        dgt_logger.info(
            f"Tool trajectory verification {'succeeded' if output is not None else 'failed'}",
        )

        return [output] if output is not None else []

    def _parse_verifier_output(self, prediction: Dict[str, Any]) -> ConversationDataPoint | None:

        #
        data_obj: ConversationDataPoint = prediction["reference"]
        tool_info = get_first_step_of_type(data_obj.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        # all plans except for last one
        plan_steps = get_all_steps_of_type(data_obj.steps, ToolPlanStep)
        prior_tool_calls = [tc.to_dict() for pl in plan_steps[:-1] for tc in pl.plan or []]

        # all user requests
        user_steps = get_all_steps_of_type(data_obj.steps, ToolUserStep)
        user_requests = "\n".join([s.content for s in user_steps])
        required_terms = user_steps[-1].required_terms or []

        best_opt = None
        results = prediction["result"]
        results = results if isinstance(results, list) else [results]
        logprobs = ((prediction or dict()).get("addtl") or dict()).get(
            "token_logprobs", [[{None: 0}]]
        )

        options = []
        for res_i, result in enumerate(results):
            result = result or dict()

            # parse generated text
            text = (
                (result.get(CONTENT_KEY) or "")
                .strip()
                .split("</tool_calls>")[0]
                .split("<tool_calls>")[-1]
                .strip()
            )

            if text.endswith("{"):
                text = text[:-1].strip()
                if text.endswith(","):
                    text = text[:-1]
                text += "\n]"

            text = extract_first_tool_call(text)

            parsed_tool_calls = try_parse_json_string(text)
            if not (
                parsed_tool_calls
                and isinstance(parsed_tool_calls, list)
                and syntax_check(
                    [{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc} for tc in parsed_tool_calls]
                )
            ):
                dgt_logger.debug("Verification failed to pass basic syntax check")
                continue

            is_valid = validate_tool_calls(
                tool_calls=prior_tool_calls + parsed_tool_calls,
                conversation_history=user_requests,
                tools=tool_info.tools,
                allow_nested=self._has_nested,
                require_nested=self._has_nested,
                check_arg_question_overlap=False,
                # log_failures=True,
            )
            if not is_valid:
                dgt_logger.debug("Verification failed to pass specification validation")
                continue

            # validation
            connected_components = extract_largest_connected_component(
                parsed_tool_calls,
                context=prior_tool_calls,
            )
            if self._has_nested and len(connected_components) != len(parsed_tool_calls):
                dgt_logger.debug(
                    "Failed to generate a query with required connected component structure"
                )
                continue

            arg_sets = [
                {
                    str(val)
                    for val in tc.get(TOOL_CALL_ARGS).values()
                    if not (isinstance(val, str) and val.startswith("$"))
                }
                for tc in parsed_tool_calls
            ]
            empty_sets = [arg_set for arg_set in arg_sets if not arg_set]
            if len(empty_sets) > 1:
                dgt_logger.debug("Scenario verification step failed due to lack of complexity")
                continue

            arg_sets = [arg_set for arg_set in arg_sets if arg_set]
            largest_arg_set = max(arg_sets, key=len) if arg_sets else set()
            if len(arg_sets) > 1 and all(arg_set.issubset(largest_arg_set) for arg_set in arg_sets):
                dgt_logger.debug(
                    "Scenario verification step failed due to non-duplicate argument check"
                )
                continue

            orig_leaves = set(
                [normalize_str(x) for x in required_terms if not isinstance(x, list | dict)]
            )
            new_leaves = set(
                [
                    normalize_str(x)
                    for x in get_leaves(parsed_tool_calls)
                    if not isinstance(x, list | dict)
                ]
            )
            # if (len(orig_leaves) > 0 and orig_leaves.difference(new_leaves)) or [
            #     tc[TOOL_NAME] for tc in parsed_tool_calls
            # ] != [tc.name for tc in plan_steps[-1].plan or []]:
            if len(orig_leaves) > 0 and orig_leaves.difference(new_leaves):
                dgt_logger.debug(
                    "Scenario verification failed to produce tool calls satisfying leaf backtranslation"
                )
                continue

            for tool_call in parsed_tool_calls:
                matching_tools = [t for t in tool_info.tools if t.name == tool_call.get(TOOL_NAME)]
                assert len(matching_tools) == 1
                tool_call[TOOL_NAMESPACE] = matching_tools[0].namespace

            scores = [next(iter(lp.values())) for lp in logprobs[res_i]]
            score = sum([score for score in scores if isinstance(score, (float, int))])

            options.append((score, [ToolCall.from_dict(tc) for tc in parsed_tool_calls]))

        #
        if not options:
            dgt_logger.debug("Failed to produce query during validation")
            return

        # set plan with consensus
        best_opt = max(options, key=lambda x: x[0])[1]
        plan_steps[-1].plan = best_opt

        return data_obj


def _get_verifier_prompt_kwargs(data_point: ConversationDataPoint, has_nested: bool):
    tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
    assert isinstance(tool_info, ToolInfoStep)

    user_steps = get_all_steps_of_type(data_point.steps, ToolUserStep)
    plan_steps = get_all_steps_of_type(data_point.steps, ToolPlanStep)

    #
    history = []
    for user_step, plan_step in zip(user_steps[:-1], plan_steps[:-1]):
        history.append(
            {
                ROLE_KEY: USER_ROLE,
                CONTENT_KEY: user_step.content,
            }
        )
        history.extend(
            [{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc.to_dict()} for tc in plan_step.plan or []]
        )

    required_tool_names = [tc.name for tc in plan_steps[-1].plan or []]
    tools = [t.to_dict() for t in tool_info.tools or [] if t.name in required_tool_names]
    if not has_nested:
        tools = [{TOOL_NAME: t[TOOL_NAME], TOOL_PARAMETERS: t[TOOL_PARAMETERS]} for t in tools]
    random.shuffle(tools)

    # Add to LM inputs
    prompt_kwargs = {
        "tools": json.dumps(tools, indent=4),
        "history": json.dumps(history, indent=4),
        "request": user_steps[-1].content,
        "tool_calls": json.dumps([tc.to_dict() for tc in plan_steps[-1].plan or []], indent=4),
        "required": "\n".join([json.dumps(req) for req in user_steps[-1].required_terms or []]),
        "allow_nested": has_nested,
    }
    return prompt_kwargs

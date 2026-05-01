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
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    PersonaStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.stages.sample_persona import render_persona
from fms_dgt.core.databuilders.conversation.utils import (
    get_all_steps_of_type,
    get_first_step_of_type,
    get_instruction_role,
)
from fms_dgt.core.tools.constants import TOOL_CALL_ARGS, TOOL_CALL_ID
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
    ToolUserStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    correlation_check,
    get_leaves,
    shuffle_tools,
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger


# ===========================================================================
#                       MAIN CLASSES
# ===========================================================================
@register_stage("tool_calling/multi_turn/stages/user")
class ToolCallingUser(Stage):
    """LM-based user agent for generating tool calling requests.

    Generates user requests that appropriately match planned tool calls,
    with support for multi-turn conversations and context management.
    """

    def __init__(
        self,
        *args,
        generator: LMProvider,
        separate_context: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._lm = generator

        # set tool handler
        self._separate_context = separate_context

        self._instr_prompt, self._user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(Path(__file__).parent.parent, "prompts", "user", role + ".txt")
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
        r""" """
        assert len(data_points) == 1
        data_point = data_points[0]

        prompt_kwargs = self._get_prompt_kwargs(data_point)
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

        lm_inputs = [
            {
                "input": messages,
                "gen_kwargs": {"logprobs": True},
                "reference": data_point,
                "prompt_kwargs": prompt_kwargs,
            }
        ]

        dgt_logger.info("Generating user request")
        lm_output = next(
            iter(self._lm(lm_inputs, disable_tqdm=True, method=LMProvider.CHAT_COMPLETION))
        )

        output = self._parse_user_requests(lm_output)
        dgt_logger.info(
            f"User request generation {'succeeded' if output is not None else 'failed'}",
        )

        return [output] if output is not None else []

    def _get_prompt_kwargs(self, data_point: ConversationDataPoint):
        # Get scenario
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        persona = get_first_step_of_type(data_point.steps, PersonaStep)
        assert isinstance(persona, PersonaStep)
        rendered_persona = render_persona(persona, sep="\n")

        plans = get_all_steps_of_type(data_point.steps, ToolPlanStep)
        user_requests = get_all_steps_of_type(data_point.steps, ToolUserStep)
        history = []
        for request, plan in zip(user_requests, plans):
            history.append(
                {
                    ROLE_KEY: USER_ROLE,
                    CONTENT_KEY: request.content,
                }
            )
            history.extend(
                [{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc.to_dict()} for tc in plan.plan or []]
            )

        hidden_ids = _split_tool_calls([tc.to_dict() for tc in plans[-1].plan or []])

        context, tool_calls = [], []
        for tc in plans[-1].plan or []:
            if (
                # separate_context is enabled
                self._separate_context
                and tc.call_id in hidden_ids
            ):
                context.append({ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc.to_dict()})
            else:
                tool_calls.append({ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc.to_dict()})

        used_tools = [tc.name for pl in plans for tc in pl.plan or []]

        #
        tools = [tool.to_dict() for tool in tool_info.tools or [] if tool.name in used_tools]
        tools = shuffle_tools(tools)

        # get required elements
        required = get_leaves([tc.to_dict() for tc in plans[-1].plan or []])
        random.shuffle(required)

        # Add to LM inputs
        prompt_kwargs = {
            "tools": json.dumps(tools, indent=4),
            "required": "\n".join([json.dumps(req) for req in required]),
            "history": json.dumps(history, indent=4),
            "hidden": json.dumps(context, indent=4),
            "tool_calls": json.dumps(tool_calls, indent=4),
            #
            "persona": rendered_persona,
            "required_terms": required,
        }

        return prompt_kwargs

    def _parse_user_requests(self, prediction: Dict[str, Any]) -> ConversationDataPoint | None:
        #
        data_obj: ConversationDataPoint = prediction["reference"]
        prompt_kwargs: Dict = prediction["prompt_kwargs"]
        required_terms = prompt_kwargs["required_terms"]
        hidden_set = [x.get(CONTENT_KEY) for x in json.loads(prompt_kwargs["hidden"])]

        tool_info = get_first_step_of_type(data_obj.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        user_steps = get_all_steps_of_type(data_obj.steps, ToolUserStep)
        plan_steps = get_all_steps_of_type(data_obj.steps, ToolPlanStep)
        tool_calls = [tc.to_dict() for plan in plan_steps for tc in plan.plan or []]

        results = prediction["result"]
        results = results if isinstance(results, list) else [results]
        logprobs = ((prediction or dict()).get("addtl") or dict()).get(
            "token_logprobs", [[{None: 0}]]
        )

        options = []
        for res_i, res in enumerate(results):
            # goal here is to take the first result that works (since all user requests should be for the same query...)
            result: str = ((res or dict()).get(CONTENT_KEY) or "").strip()
            text = (
                result.split("</request>")[0].split("<request>")[-1].strip()
                if "<request>" in result
                else ""
            )

            #
            if not text:
                dgt_logger.debug("User utterance generation failed due to no output being produced")
                continue

            if any([tc.get(TOOL_CALL_ID, "") + "." in text for tc in tool_calls]):
                dgt_logger.debug(
                    "User utterance generation failed due to presence of tool call ID in text"
                )
                continue

            user_texts, asst_tool_calls = [], []
            for user_step, plan_step in zip(user_steps, plan_steps):
                user_texts.append(user_step.content)
                for tc in plan_step.plan or []:
                    asst_tool_calls.append(tc.to_dict())
            user_texts.append(text)
            asst_tool_calls.extend([tc.to_dict() for tc in plan_steps[-1].plan or []])

            is_valid = validate_tool_calls(
                tool_calls=asst_tool_calls,
                conversation_history="\n".join(user_texts),
                tools=tool_info.tools or [],
                allow_nested=True,
                require_nested=False,
                check_arg_question_overlap=False,
            )
            if not is_valid:
                dgt_logger.debug("User utterance generation failed during request validation")
                continue

            # tools
            if not user_steps:
                if any(
                    [
                        any([wrd.startswith(past_wrd) for wrd in text.lower().split()])
                        for past_wrd in ["was", "earlier", "previous", "prior"]
                    ]
                ) and any([past_phrase in text.lower() for past_phrase in ["you just"]]):
                    dgt_logger.debug("User request failed due to past tense validation")
                    continue

                # check rank correlation
                if (
                    self._separate_context
                    and hidden_set
                    and not correlation_check(
                        [tc.to_dict() for tc in plan_steps[-1].plan or []], text, hidden_set
                    )
                ):
                    dgt_logger.debug("User request failed to pass correlation check")
                    continue

            scores = [next(iter(lp.values())) for lp in logprobs[res_i]]
            score = sum([score for score in scores if isinstance(score, (float, int))])
            options.append([score, text])

        #
        best_opt = max(options, key=lambda x: x[0])[1] if options else None
        if not best_opt:
            dgt_logger.debug("Failed to produce user request")
            return

        data_obj.steps.append(
            ToolUserStep(
                content=best_opt,
                required_terms=required_terms,
                hidden_set=hidden_set,
            )
        )

        return data_obj


###
#
###


def _split_tool_calls(tool_calls: List[Dict]) -> List[str]:

    def _has_ref(tgt: str, d: Any):
        if not isinstance(tgt, str):
            return False
        elif isinstance(d, dict):
            return any([_has_ref(tgt, v) for v in d.values()])
        elif isinstance(d, list):
            return any([_has_ref(tgt, v) for v in d])
        elif isinstance(d, str):
            return d.startswith(str(tgt))
        else:
            return tgt == d

    # graph with keys = parents + values = children
    parent_to_args = {tc.get(TOOL_CALL_ID): set() for tc in tool_calls}
    for inp_tool_call in tool_calls:
        if isinstance(inp_tool_call.get(TOOL_CALL_ARGS), dict):
            for out_tool_call in tool_calls:
                if inp_tool_call == out_tool_call:
                    continue
                elif _has_ref(out_tool_call.get(TOOL_CALL_ID), inp_tool_call.get(TOOL_CALL_ARGS)):
                    par, arg = inp_tool_call.get(TOOL_CALL_ID), out_tool_call.get(TOOL_CALL_ID)
                    parent_to_args[par].add(arg)
                    break

    # flatten
    add_deps = {None}
    while add_deps:
        for tc in tool_calls:
            add_deps = set()
            for other_tc_id in parent_to_args[tc.get(TOOL_CALL_ID)]:
                add_deps.update(parent_to_args[other_tc_id])
            add_deps = add_deps.difference(parent_to_args[tc.get(TOOL_CALL_ID)])
            if add_deps:
                parent_to_args[tc.get(TOOL_CALL_ID)].update(add_deps)
                break

    # cannot include tool calls without dependencies
    parent_to_args = {
        tc_id: deps
        for tc_id, deps in parent_to_args.items()
        if any([tc_id in other_deps for other_deps in parent_to_args.values()])
    }

    # otherwise include at random
    hidden_ids = []
    to_add = random.randint(1, max(len(parent_to_args), 1))
    while len(hidden_ids) < to_add:
        frontier = [tc_id for tc_id in parent_to_args if parent_to_args.get(tc_id) == set()]
        if not frontier:
            break
        choice = random.choice(frontier)
        hidden_ids.append(choice)
        del parent_to_args[choice]
        for tc_id in parent_to_args:
            parent_to_args[tc_id].discard(choice)

    # return
    return hidden_ids

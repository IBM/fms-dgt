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
from fms_dgt.core.tools.constants import (
    TOOL_CALL_ARGS,
    TOOL_CALL_ID,
    TOOL_NAME,
    TOOL_NAMESPACE,
    TOOL_OUTPUT_PARAMETERS,
    TOOL_PARAMETERS,
)
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.utils import extract_first_tool_call
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
    ToolPlanStep,
)
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.utils import (
    extract_largest_connected_component,
    order_based_subst,
    shuffle_tools,
    syntax_check,
    validate_tool_calls,
)
from fms_dgt.utils import dgt_logger, try_parse_json_string


@register_stage("tool_calling/multi_turn/stages/planner")
class ToolCallingPlanGenerator(Stage):
    """Plan generator for tool calling sequences.

    Generates sequences of tool calls that form coherent plans to solve tasks,
    with support for nested tool calls and dependencies.
    """

    def __init__(
        self,
        *args,
        generator: LMProvider,
        has_nested: bool = True,
        min_plan_length: int | None = None,
        max_plan_length: int | None = None,
        **kwargs,
    ):
        """Initialize the ToolCallingPlanGenerator.

        Args:
            *args: Positional arguments passed to parent.
            lm: Language model provider for plan generation.
            tool_handler: Handler for managing available tools.
            has_nested: Whether to allow nested tool calls.
            min_plan_length: Minimum number of tool calls in a plan.
            max_plan_length: Maximum number of tool calls in a plan.
            **kwargs: Additional keyword arguments passed to parent.
        """
        # initialize parent
        super().__init__(*args, **kwargs)

        self._lm = generator

        self._instr_prompt, self._user_prompt = [
            JinjaPromptTemplate(
                template_path=str(
                    Path(Path(__file__).parent.parent, "prompts", "plan", role + ".txt")
                ),
            )
            for role in ["instructions", "user"]
        ]

        self._has_nested = has_nested
        self._min_plan_length = min_plan_length
        self._max_plan_length = max_plan_length

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

        # Get scenario
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert isinstance(tool_info, ToolInfoStep)

        persona = get_first_step_of_type(data_point.steps, PersonaStep)
        assert isinstance(persona, PersonaStep)
        rendered_persona = render_persona(persona, sep="\n")

        plans = get_all_steps_of_type(data_point.steps, ToolPlanStep)
        prior_tool_calls = [tc for pl in plans for tc in pl.plan or []]

        tools = tool_info.tools + []
        tools = [
            t.to_dict(
                keep_keys=[TOOL_NAME, TOOL_PARAMETERS]
                + ([TOOL_OUTPUT_PARAMETERS] if self._has_nested else [])
            )
            for t in tools
            if t.name not in [tc.name for tc in prior_tool_calls]
        ]
        tools = shuffle_tools(tools)

        prompt_kwargs = {
            "tools": json.dumps(tools, indent=4),
            "date": tool_info.date,
            "persona": rendered_persona,
            "plan_length": random.randint(self._min_plan_length, self._max_plan_length),
            "prior_tool_calls": json.dumps([tc.to_dict() for tc in prior_tool_calls], indent=4),
            "allow_nested": self._has_nested,
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

        # Add to LM inputs
        lm_inputs = [
            {
                "input": messages,
                "gen_kwargs": {"logprobs": True},
                "reference": data_point,
                "prompt_kwargs": prompt_kwargs,
            }
        ]

        # Invoke LM
        dgt_logger.info("Generating tool trajectories")
        lm_output = next(
            iter(self._lm(lm_inputs, disable_tqdm=True, method=LMProvider.CHAT_COMPLETION))
        )

        # Parse and prepare outputs
        output = self._parse_tool_trajectories(lm_output)

        dgt_logger.info(
            f"Tool trajectory generation {'succeeded' if output is not None else 'failed'}",
        )

        return [output] if output is not None else []

    def _parse_tool_trajectories(self, prediction: Dict) -> ConversationDataPoint | None:
        # Fetch data point
        data_point: ConversationDataPoint = prediction["reference"]
        tool_info = get_first_step_of_type(data_point.steps, ToolInfoStep)
        assert tool_info is not None

        plans = get_all_steps_of_type(data_point.steps, ToolPlanStep)
        prior_tool_calls = [tc.to_dict() for pl in plans for tc in pl.plan or []]

        prompt_kwargs = prediction["prompt_kwargs"]
        plan_length = prompt_kwargs["plan_length"]

        results = prediction["result"]
        results = results if isinstance(results, list) else [results]
        logprobs = ((prediction or dict()).get("addtl") or dict()).get(
            "token_logprobs", [[{None: 0}]]
        )

        options = []
        for res_i, res in enumerate(results):
            # Identify last conversation from the text containing both prompt text and generated text
            text = (
                ((res or dict()).get(CONTENT_KEY) or "")
                .strip()
                .split("</tool_calls>")[0]
                .split("<tool_calls>")[-1]
                .strip()
            )
            text = extract_first_tool_call(text)

            if not text:
                dgt_logger.debug(
                    "Plan generation failed when extracting tool call sequence from text"
                )
                continue

            # Extract JSON from the last conversation text given LM is instructued to generate conversation in a JSON format
            parsed_tool_calls = try_parse_json_string(text)

            if not (
                parsed_tool_calls
                and isinstance(parsed_tool_calls, list)
                and syntax_check(
                    [{ROLE_KEY: ASSISTANT_ROLE, CONTENT_KEY: tc} for tc in parsed_tool_calls]
                )
            ):
                dgt_logger.debug("Plan generation failed to pass basic syntax check")
                continue

            # extract new generation
            connected_component = (
                extract_largest_connected_component(
                    parsed_tool_calls,
                    context=prior_tool_calls,
                )
                if self._has_nested
                else parsed_tool_calls
            )

            if not connected_component:
                dgt_logger.debug("Failed to generate a query with connected components")
                continue

            if (
                len(connected_component) > plan_length
                or len(connected_component) < self._min_plan_length
            ):
                dgt_logger.debug("Failed to generate a query within size constraints")
                continue

            arg_sets = [
                {
                    str(val)
                    for val in tc.get(TOOL_CALL_ARGS).values()
                    if not (isinstance(val, str) and val.startswith("$"))
                }
                for tc in connected_component
            ]
            empty_sets = [arg_set for arg_set in arg_sets if not arg_set]
            if len(empty_sets) > 1:
                dgt_logger.debug("Plan step failed due to lack of complexity")
                continue

            arg_sets = [arg_set for arg_set in arg_sets if arg_set]
            largest_arg_set = max(arg_sets, key=len) if arg_sets else set()
            if len(arg_sets) > 1 and all(arg_set.issubset(largest_arg_set) for arg_set in arg_sets):
                dgt_logger.debug("Plan step failed to non-duplicate argument check")
                continue

            is_valid = validate_tool_calls(
                tool_calls=prior_tool_calls + connected_component,
                tools=tool_info.tools,
                conversation_history=None,
                allow_nested=self._has_nested,
                require_nested=self._has_nested,
                check_arg_question_overlap=False,
                # log_failures=True,
            )
            if not is_valid:
                dgt_logger.debug("Failed to generate a query with valid structure")
                continue

            context_ids = [tc.get(TOOL_CALL_ID) for tc in prior_tool_calls]
            tool_calls = order_based_subst(connected_component, context_ids)

            # TODO: harden this
            for tool_call in tool_calls:
                matching_tools = [t for t in tool_info.tools if t.name == tool_call.get(TOOL_NAME)]
                assert len(matching_tools) == 1
                tool_call[TOOL_NAMESPACE] = matching_tools[0].namespace

            scores = [next(iter(lp.values())) for lp in logprobs[res_i]]
            score = sum([score for score in scores if isinstance(score, (float, int))])

            options.append(
                (
                    score,
                    [ToolCall.from_dict(tc) for tc in tool_calls],
                )
            )

        #
        best_opt = max(options, key=lambda x: x[0])[1] if options else None
        if not best_opt:
            dgt_logger.debug("Failed to produce plan")
            return

        # extend data point
        data_point.steps.append(ToolPlanStep(plan=best_opt))

        # return
        return data_point

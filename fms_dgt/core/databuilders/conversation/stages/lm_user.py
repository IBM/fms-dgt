# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    FlowControllerStep,
    PersonaStep,
    ScenarioStep,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.stages.sample_persona import render_persona
from fms_dgt.core.databuilders.conversation.utils import (
    get_last_step_of_type,
    steps_to_messages,
)


def _build_user_messages(
    scenario: str,
    persona_text: str,
    hint: str | None,
    history_steps: list,
) -> list:
    system_content = "You are simulating a user in a conversation with an AI assistant."
    if scenario:
        system_content += f"\n\nConversation scenario:\n{scenario}"
    if persona_text:
        system_content += f"\n\nUser persona:\n{persona_text}"
    if hint:
        system_content += f"\n\nInstruction for this turn: {hint}"
    messages = [{"role": "system", "content": system_content}]
    # From the simulated user's perspective the roles are inverted:
    # the human user's turns are "assistant" and the AI's turns are "user".
    # Tool steps are serialized via steps_to_messages (OpenAI format) and
    # kept as-is — tool_call/tool_result are context the simulated user observes.
    for msg in steps_to_messages(history_steps):
        role = msg["role"]
        if role == "user":
            messages.append({**msg, "role": "assistant"})
        elif role == "assistant":
            # tool_call messages also have role "assistant" — both flip to "user"
            messages.append({**msg, "role": "user"})
        else:
            # "tool" results stay as-is
            messages.append(msg)
    # Prompt the LM to produce the next user turn.
    messages.append(
        {
            "role": "user",
            "content": "Write the user's next message. Output only the user message text, no labels or quotes.",
        }
    )
    return messages


@register_stage("lm/user/guided")
class GuidedUserStage(Stage):
    """Iteration stage that generates the next user turn using the flow controller hint.

    Reads:
      - The latest ScenarioStep (scenario description)
      - The latest PersonaStep with target="user" (empty string if absent)
      - The latest FlowControllerStep.hint (guidance for this turn)
      - All prior user/assistant/tool_call/tool_result steps (conversation history)

    Appends a UserStep. Drops the data point if the LM returns empty output.
    """

    def __init__(self, *, name: str, generator: LMProvider, **kwargs: Any) -> None:
        super().__init__(name=name)
        self._generator = generator

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        generator_inputs = []
        for data_point in data_points:
            scenario_step = get_last_step_of_type(data_point.steps, ScenarioStep)
            scenario = scenario_step.content if scenario_step else ""

            persona_steps = [
                step
                for step in data_point.steps
                if isinstance(step, PersonaStep) and step.target == "user"
            ]
            persona_text = render_persona(persona_steps[-1]) if persona_steps else ""

            fc_step = get_last_step_of_type(data_point.steps, FlowControllerStep)
            hint = fc_step.hint if fc_step else None

            history_steps = [
                step
                for step in data_point.steps
                if isinstance(step, (UserStep, AssistantStep, ToolCallStep, ToolResultStep))
            ]
            generator_inputs.append(
                {
                    "input": _build_user_messages(scenario, persona_text, hint, history_steps),
                    "gen_kwargs": {"max_new_tokens": 256},
                    "reference": data_point,
                    "task_name": data_point.task_name,
                }
            )

        if not generator_inputs:
            return []

        outputs = self._generator(generator_inputs, method="chat_completion", disable_tqdm=True)

        results = []
        for out in outputs:
            result = out.get("result") or ""
            if isinstance(result, dict):
                result = result.get("content") or ""
            user_text = result.strip()
            if not user_text:
                continue
            data_point: ConversationDataPoint = out["reference"]
            data_point.steps.append(UserStep(content=user_text, stage_name=self.name))
            results.append(data_point)
        return results

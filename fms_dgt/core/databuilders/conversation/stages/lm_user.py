# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.stages.sample_persona import render_persona


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
    for step in history_steps:
        # From the simulated user's perspective the roles are inverted:
        # the human user's turns are "assistant" and the AI's turns are "user".
        if step.role == "user":
            messages.append({"role": "assistant", "content": step.content})
        elif step.role == "assistant":
            messages.append({"role": "user", "content": step.content})
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
      - All prior user/assistant steps (conversation history)

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
        for ctx in data_points:
            scenario_steps = [s for s in ctx.steps if s.role == "scenario"]
            scenario = scenario_steps[-1].content if scenario_steps else ""

            persona_steps = [s for s in ctx.steps if s.role == "persona" and s.target == "user"]
            persona_text = render_persona(persona_steps[-1]) if persona_steps else ""

            fc_steps = [s for s in ctx.steps if s.role == "flow_controller"]
            hint = fc_steps[-1].hint if fc_steps else None

            history_steps = [s for s in ctx.steps if s.role in ("user", "assistant")]
            generator_inputs.append(
                {
                    "input": _build_user_messages(scenario, persona_text, hint, history_steps),
                    "gen_kwargs": {"max_new_tokens": 128},
                    "reference": ctx,
                    "task_name": ctx.task_name,
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
            ctx: ConversationDataPoint = out["reference"]
            ctx.steps.append(UserStep(content=user_text, stage_name=self.name))
            results.append(ctx)
        return results

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage


def _build_assistant_messages(scenario: str, history_steps: list) -> list:
    system_content = "You are a helpful, friendly AI assistant."
    if scenario:
        system_content += f"\n\nContext for this conversation:\n{scenario}"
    messages = [{"role": "system", "content": system_content}]
    for step in history_steps:
        if step.role == "user":
            messages.append({"role": "user", "content": step.content})
        elif step.role == "assistant":
            messages.append({"role": "assistant", "content": step.content})
    return messages


@register_stage("lm/assistant/naive")
class NaiveAssistantStage(Stage):
    """Iteration stage that generates the next assistant turn with no special guidance.

    Reads:
      - The latest ScenarioStep (scenario description)
      - All prior user/assistant steps (conversation history)

    Appends an AssistantStep. Drops the data point if the LM returns empty output.
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

            history_steps = [s for s in ctx.steps if s.role in ("user", "assistant")]
            generator_inputs.append(
                {
                    "input": _build_assistant_messages(scenario, history_steps),
                    "gen_kwargs": {"max_new_tokens": 256},
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
            assistant_text = result.strip()
            if not assistant_text:
                continue
            ctx: ConversationDataPoint = out["reference"]
            ctx.steps.append(AssistantStep(content=assistant_text, stage_name=self.name))
            results.append(ctx)
        return results

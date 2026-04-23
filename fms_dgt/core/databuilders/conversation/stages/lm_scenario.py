# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List
import random
import uuid

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    ScenarioStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import get_last_step_of_type


def _render_scenario_icl(seed_data: List[ConversationDataPoint]) -> str:
    """Render ICL examples as a newline-separated list of scenario descriptions."""
    lines = []
    for ctx in seed_data:
        scenario_step = get_last_step_of_type(ctx.steps, ScenarioStep)
        if scenario_step:
            lines.append(scenario_step.content)
    return "\n".join(f"- {line}" for line in lines) if lines else ""


def _build_scenario_messages(icl_text: str) -> list:
    content = (
        "Generate a concise scenario description (1-3 sentences) for a conversation "
        "between a user and an AI assistant. The scenario should establish who the user "
        "is, what they want to discuss, and any relevant context."
    )
    if icl_text:
        content += f"\n\nExamples of good scenarios:\n{icl_text}"
    content += "\n\nScenario:"
    return [{"role": "user", "content": content}]


@register_stage("lm/scenario")
class LMScenarioStage(Stage):
    """Initialization stage that generates a conversation scenario from ICL seed data.

    Appends a ScenarioStep whose `scenario_family_id` is set to a new UUID.
    Drops the data point if the LM call fails or returns empty output.
    """

    def __init__(
        self, *, name: str, generator: LMProvider, num_icl_examples: int = 3, **kwargs: Any
    ) -> None:
        super().__init__(name=name)
        self._generator = generator
        self._num_icl_examples = num_icl_examples

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        seed_data = seed_data or []

        # Build one prompt per data point with independently sampled ICL examples.
        generator_inputs = []
        for data_point in data_points:
            sample = random.sample(seed_data, min(self._num_icl_examples, len(seed_data)))
            icl_text = _render_scenario_icl(sample)
            generator_inputs.append(
                {
                    "input": _build_scenario_messages(icl_text),
                    "gen_kwargs": {"max_new_tokens": 1024},
                    "reference": data_point,
                    "task_name": data_point.task_name,
                }
            )

        outputs = self._generator(generator_inputs, method="chat_completion", disable_tqdm=True)

        results = []
        for out in outputs:
            result = out.get("result") or ""
            if isinstance(result, dict):
                result = result.get("content") or ""
            scenario_text = result.strip()
            if not scenario_text:
                continue
            data_point: ConversationDataPoint = out["reference"]
            data_point.steps.append(
                ScenarioStep(
                    content=scenario_text,
                    stage_name=self.name,
                    scenario_family_id=str(uuid.uuid4()),
                )
            )
            results.append(data_point)
        return results

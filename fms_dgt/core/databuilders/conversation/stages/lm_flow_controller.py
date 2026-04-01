# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    FlowControllerStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage


def _build_conversation_summary(context: ConversationDataPoint) -> str:
    """Return a condensed text representation of the conversation so far."""
    lines = []
    scenario_steps = [s for s in context.steps if s.role == "scenario"]
    if scenario_steps:
        lines.append(f"[Scenario] {scenario_steps[-1].content}")
    for step in context.steps:
        if step.role == "user":
            lines.append(f"User: {step.content}")
        elif step.role == "assistant":
            lines.append(f"Assistant: {step.content}")
    return "\n".join(lines)


def _build_fc_messages(
    conversation_summary: str,
    patterns: Dict[str, Dict[str, str]],
    termination_patterns: List[str],
    turn_count: int,
    max_turns: int,
) -> list:
    pattern_list = "\n".join(f"- {name}: {info['description']}" for name, info in patterns.items())
    near_end = turn_count >= max(1, max_turns - 1)
    if near_end and termination_patterns:
        names = " or ".join(f"'{p}'" for p in termination_patterns)
        termination_note = (
            f"\nNote: the conversation is near its maximum length; strongly prefer {names}."
        )
    else:
        termination_note = ""
    content = (
        "You are a conversation flow controller. "
        "Given the conversation so far, choose the most appropriate next interaction pattern.\n"
        f"\nConversation so far:\n{conversation_summary or '(conversation not started yet)'}"
        f"\n\nAvailable patterns:\n{pattern_list}"
        f"{termination_note}"
        "\n\nRespond with ONLY the pattern name, nothing else."
    )
    return [{"role": "user", "content": content}]


def _parse_pattern(raw: str, patterns: Dict[str, Dict[str, str]]) -> str | None:
    """Extract a valid pattern name from the LM output."""
    raw = raw.strip()
    # Exact match first.
    if raw in patterns:
        return raw
    # Substring match (LM sometimes pads with spaces or punctuation).
    for name in patterns:
        if name in raw:
            return name
    return None


@register_stage("lm/flow_controller")
class LMFlowControllerStage(Stage):
    """Iteration stage that selects the next interaction pattern.

    Appends a FlowControllerStep with:
      - content: the chosen pattern name
      - terminate: True when a termination pattern is selected
      - hint: the pattern's hint text for the user stage

    Drops the data point on LM parse failure (returns empty list for that point).

    Config kwargs:
        patterns: Required. List of dicts, each with:
            - ``name``: pattern identifier (required)
            - ``description``: shown to the LM so it can choose this pattern (required)
            - ``hint``: passed to the user stage to guide turn generation (required)
            - ``weight``: optional float, reserved for future weighted sampling
        termination_patterns: List of pattern names that signal conversation end.
            Must be a subset of the names listed in ``patterns``. Required.
        max_turns: Passed through from the task so the FC can signal early
            termination when the conversation is near its limit. Default: 10.
    """

    def __init__(
        self,
        *,
        name: str,
        generator: LMProvider,
        patterns: List[Dict],
        termination_patterns: List[str],
        max_turns: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self._generator = generator
        self._max_turns = max_turns
        self._termination_patterns: List[str] = termination_patterns

        self._patterns: Dict[str, Dict[str, str]] = {}
        for p in patterns:
            pname = p["name"]
            self._patterns[pname] = {
                "description": p.get("description", pname),
                "hint": p.get("hint", f"Generate a user turn following the '{pname}' pattern."),
            }

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        generator_inputs = []
        for ctx in data_points:
            turn_count = len([s for s in ctx.steps if s.role == "user"])
            summary = _build_conversation_summary(ctx)
            generator_inputs.append(
                {
                    "input": _build_fc_messages(
                        summary,
                        self._patterns,
                        self._termination_patterns,
                        turn_count,
                        self._max_turns,
                    ),
                    "gen_kwargs": {"max_new_tokens": 32},
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
            raw = result.strip()
            pattern_name = _parse_pattern(raw, self._patterns)
            if pattern_name is None:
                # Parse failure: drop this data point.
                continue
            ctx: ConversationDataPoint = out["reference"]
            terminate = pattern_name in self._termination_patterns
            hint = self._patterns[pattern_name].get("hint")
            ctx.steps.append(
                FlowControllerStep(
                    content=pattern_name,
                    stage_name=self.name,
                    terminate=terminate,
                    hint=hint,
                )
            )
            results.append(ctx)
        return results

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    FlowControllerStep,
    ScenarioStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_all_steps_of_type,
    get_last_step_of_type,
    steps_to_text,
)


def _build_fc_schema(pattern_names: List[str]) -> Dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning about which patterns are coherent given the documents and conversation state.",
            },
            "eligible_patterns": {
                "type": "array",
                "items": {"type": "string", "enum": pattern_names},
                "description": "Patterns that are coherent and possible given the current context.",
            },
            "pattern": {
                "type": "string",
                "enum": pattern_names,
                "description": "The single pattern selected for the next turn.",
            },
            "hint": {
                "type": "string",
                "description": "A short guide for the user stage on how to frame the next question or utterance, without revealing the answer.",
            },
        },
        "required": ["reasoning", "eligible_patterns", "pattern", "hint"],
        "additionalProperties": False,
    }


def _build_fc_messages(
    conversation_summary: str,
    patterns: Dict[str, Dict[str, str]],
    termination_patterns: List[str],
    turn_count: int,
    max_turns: int,
    pattern_history: List[str],
) -> list:
    pattern_list = "\n".join(
        f"- {name}: {info['description']}"
        + (
            f"\n  Example hint (contextualize for this conversation): {info['hint']}"
            if info.get("hint")
            else ""
        )
        for name, info in patterns.items()
    )
    near_end = turn_count >= max(1, max_turns - 1)
    if near_end and termination_patterns:
        names = " or ".join(f"'{p}'" for p in termination_patterns)
        termination_note = (
            f"\nIMPORTANT: the conversation is near its maximum length. "
            f"You MUST select {names} unless it is genuinely impossible."
        )
    else:
        termination_note = ""

    if pattern_history:
        history_line = ", ".join(pattern_history)
        pattern_history_section = (
            f"\nPatterns selected in previous turns (oldest to newest): {history_line}\n"
        )
    else:
        pattern_history_section = ""

    content = (
        "You are a conversation flow controller. "
        "Your job is to decide what interaction pattern the user should follow next.\n\n"
        "Follow these steps:\n"
        "1. Read the scenario and conversation history carefully.\n"
        "2. Reason through each available pattern: is it coherent and applicable given the conversation context and the pattern's description? "
        "Only mark a pattern as eligible if it genuinely fits the current state of the conversation. "
        "Use the pattern history below to apply any consecutive-use or frequency caps stated in the pattern descriptions.\n"
        "3. From the eligible patterns, select the one that best moves the conversation forward without repeating previous turns.\n"
        "4. Write a short hint for the user that guides how to frame the next question or utterance. "
        "Use the selected pattern's example hint as a starting point, but contextualize it for the specific scenario and conversation history. "
        "The hint must not reveal the answer and must not repeat something already asked.\n\n"
        f"Conversation so far:\n{conversation_summary or '(conversation not started yet)'}\n"
        f"{pattern_history_section}\n"
        f"Available patterns:\n{pattern_list}"
        f"{termination_note}\n\n"
        "Return a JSON object with: reasoning, eligible_patterns (list of pattern names that fit), "
        "pattern (your chosen pattern), and hint (guidance for the user stage)."
    )
    return [{"role": "user", "content": content}]


def _parse_fc_output(
    raw: str, patterns: Dict[str, Dict[str, str]]
) -> tuple[str | None, str | None, List[str], str | None]:
    """Parse LM output into (pattern_name, hint, eligible_patterns, reasoning).

    Tries JSON parse first. Falls back to substring match on pattern names
    for providers that ignore response_format, returning empty eligible list
    and None for hint and reasoning.
    """
    try:
        parsed = json.loads(raw)
        pattern = parsed.get("pattern", "").strip()
        hint = parsed.get("hint", "").strip() or None
        reasoning = parsed.get("reasoning", "").strip() or None
        eligible = [p for p in parsed.get("eligible_patterns", []) if p in patterns]
        if pattern in patterns:
            return pattern, hint, eligible, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: substring match
    raw_stripped = raw.strip()
    if raw_stripped in patterns:
        return raw_stripped, None, [], None
    for name in patterns:
        if name in raw_stripped:
            return name, None, [], None
    return None, None, [], None


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

        pattern_names = list(self._patterns.keys())
        self._fc_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "flow_controller",
                "strict": True,
                "schema": _build_fc_schema(pattern_names),
            },
        }

    def _build_conversation_summary(self, context: ConversationDataPoint) -> str:
        """Return a condensed text representation of the conversation so far.

        Override in subclasses to inject additional context (e.g. document text
        for RAG-specific eligibility reasoning).
        """
        lines = []
        scenario_step = get_last_step_of_type(context.steps, ScenarioStep)
        if scenario_step:
            lines.append(f"[Scenario]\n{scenario_step.content}")
        history = steps_to_text(context.steps)
        if history:
            lines.append(history)
        return "\n".join(lines)

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        generator_inputs = []
        for data_point in data_points:
            generator_inputs.append(
                {
                    "input": _build_fc_messages(
                        self._build_conversation_summary(data_point),
                        self._patterns,
                        self._termination_patterns,
                        len(get_all_steps_of_type(data_point.steps, UserStep)),
                        self._max_turns,
                        [
                            step.content
                            for step in get_all_steps_of_type(data_point.steps, FlowControllerStep)
                            if isinstance(step.content, str)
                        ],
                    ),
                    "gen_kwargs": {
                        "max_new_tokens": 512,
                        "response_format": self._fc_response_format,
                    },
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
            raw = result.strip()
            pattern_name, hint, eligible_patterns, reasoning = _parse_fc_output(raw, self._patterns)
            if pattern_name is None:
                continue
            data_point: ConversationDataPoint = out["reference"]
            terminate = pattern_name in self._termination_patterns
            # LM-generated hint takes precedence; fall back to static hint from config.
            if not hint:
                hint = self._patterns[pattern_name].get("hint")
            data_point.steps.append(
                FlowControllerStep(
                    content=pattern_name,
                    stage_name=self.name,
                    terminate=terminate,
                    hint=hint,
                    reasoning=reasoning,
                    eligible_patterns=eligible_patterns,
                )
            )
            results.append(data_point)
        return results

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json
import random

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


def _build_eligibility_schema(pattern_names: List[str]) -> Dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning about which patterns are coherent given the scenario and conversation state.",
            },
            "eligible_patterns": {
                "type": "array",
                "items": {"type": "string", "enum": pattern_names},
                "description": "Patterns that are coherent and possible given the current context.",
            },
        },
        "required": ["reasoning", "eligible_patterns"],
        "additionalProperties": False,
    }


def _build_hint_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "hint": {
                "type": "string",
                "description": "A short guide for the user on how to frame their next utterance.",
            },
        },
        "required": ["hint"],
        "additionalProperties": False,
    }


def _build_eligibility_messages(
    conversation_summary: str,
    patterns: Dict[str, Dict[str, str]],
    termination_patterns: List[str],
    turn_count: int,
    max_turns: int,
    pattern_history: List[str],
) -> list:
    pattern_list = "\n".join(f"- {name}: {info['description']}" for name, info in patterns.items())

    near_end = turn_count >= max(1, max_turns - 1)
    if near_end and termination_patterns:
        names = " or ".join(f"'{p}'" for p in termination_patterns)
        termination_note = (
            f"\nIMPORTANT: the conversation is near its maximum length. "
            f"You MUST include {names} in eligible_patterns unless it is genuinely impossible."
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
        "Your job is to identify which interaction patterns are valid for the next user turn.\n\n"
        "Follow these steps:\n"
        "1. Read the scenario and conversation history carefully.\n"
        "2. Reason through each available pattern: is it coherent and applicable given the "
        "conversation context and the pattern's description? "
        "Only mark a pattern as eligible if it genuinely fits the current state of the conversation. "
        "Use the pattern history below to apply any consecutive-use or frequency caps stated in "
        "the pattern descriptions.\n\n"
        f"Conversation so far:\n{conversation_summary or '(conversation not started yet)'}\n"
        f"{pattern_history_section}\n"
        f"Available patterns:\n{pattern_list}"
        f"{termination_note}\n\n"
        "Return a JSON object with: reasoning and eligible_patterns (list of pattern names that fit)."
    )

    return [{"role": "user", "content": content}]


def _build_hint_messages(
    conversation_summary: str,
    pattern_name: str,
    static_hint: str | None,
) -> list:
    if static_hint:
        hint_guidance = (
            f"Hint template: {static_hint}\n"
            "Contextualise the hint template for the specific scenario, and "
            "conversation so far."
        )
    else:
        hint_guidance = (
            f"There is no hint template for this pattern. "
            f"Write a hint that guides the user on how to frame their next utterance "
            f"following the '{pattern_name}' pattern, contextualised to the specific "
            "scenario and conversation so far."
        )

    content = (
        "You are a conversation flow controller. "
        "Your job is to write a hint that guides the user on how to frame their next utterance.\n\n"
        f"Selected pattern: {pattern_name}\n"
        f"{hint_guidance}\n\n"
        "The hint must not reveal the answer and must not repeat something already asked.\n\n"
        f"Conversation so far:\n{conversation_summary or '(conversation not started yet)'}\n\n"
        "Return a JSON object with a single 'hint' field containing the hint text."
    )

    return [{"role": "user", "content": content}]


def _parse_eligibility_output(
    raw: str, patterns: Dict[str, Dict[str, str]]
) -> tuple[List[str], str | None]:
    """Parse LM output into (eligible_patterns, reasoning).

    Tries JSON parse first. Falls back to treating all patterns as eligible
    for providers that ignore response_format.
    """
    try:
        parsed = json.loads(raw)
        reasoning = parsed.get("reasoning", "").strip() or None
        eligible = [p for p in parsed.get("eligible_patterns", []) if p in patterns]
        return eligible, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: treat all patterns as eligible
    return list(patterns.keys()), None


@register_stage("lm/flow_controller")
class LMFlowControllerStage(Stage):
    """Iteration stage that selects the next interaction pattern.

    Uses two LM calls per turn:
        - Call 1: LM reasons about which patterns are eligible given the current
        conversation state. Returns eligible_patterns and reasoning.
        - Code: weighted random sampling via random.choices selects the pattern
        from the eligible set using per-pattern weights.
        - Call 2: LM generates a contextualised hint for the selected pattern,
        using the static hint template from config as a starting point and
        adapting it to the specific scenario and conversation history.

    Appends a FlowControllerStep with:
        - content: the chosen pattern name
        - terminate: True when a termination pattern is selected
        - hint: contextualised hint text for the user stage

    Drops the data point on LM parse failure (returns empty list for that point).

    Config kwargs:
        patterns: Required. List of dicts, each with:
            - ``name``: pattern identifier (required)
            - ``description``: shown to the LM for eligibility reasoning (required)
            - ``hint``: optional static hint template. If provided, the LM uses it
              as a starting point and contextualises it for the specific conversation.
              If omitted, the LM generates a hint from scratch based on the pattern
              name and conversation context. Used as fallback if the hint LM call fails.
            - ``weight``: optional float. Controls relative sampling probability
              among eligible patterns. Treated as a ratio, not a probability —
              a pattern with weight 3.0 is three times as likely to be selected
              as one with weight 1.0. Weights do not need to sum to 1. Defaults
              to 1.0 (uniform sampling across eligible patterns).
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
                "hint": p.get("hint"),  # None if not specified; used as template for call 2
                "weight": float(p.get("weight", 1.0)),
            }

        pattern_names = list(self._patterns.keys())
        self._fc_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "flow_controller",
                "strict": True,
                "schema": _build_eligibility_schema(pattern_names),
            },
        }
        self._hint_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "hint_generator",
                "strict": True,
                "schema": _build_hint_schema(),
            },
        }

    def _build_conversation_summary(self, context: ConversationDataPoint) -> str:
        """Return a condensed text representation of the conversation so far.

        Override in subclasses to inject additional context (e.g. document text,
        retrieved passages, or other scenario-specific state needed for eligibility
        reasoning).
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

        # --- Call 1: eligibility ---
        eligibility_inputs = []
        for data_point in data_points:
            eligibility_inputs.append(
                {
                    "input": _build_eligibility_messages(
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

        if not eligibility_inputs:
            return []

        eligibility_outputs = self._generator(
            eligibility_inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True
        )

        # --- Weighted sampling ---
        # Build hint inputs only for data points that parsed successfully.
        hint_inputs = []

        for out in eligibility_outputs:
            result = out.get("result") or ""
            if isinstance(result, dict):
                result = result.get("content") or ""
            raw = result.strip()

            eligible_patterns, reasoning = _parse_eligibility_output(raw, self._patterns)

            if not eligible_patterns:
                continue

            # Select a pattern based on stated preference
            selected_pattern = random.choices(
                eligible_patterns,
                weights=[self._patterns[p]["weight"] for p in eligible_patterns],
                k=1,
            )[0]

            data_point: ConversationDataPoint = out["reference"]
            hint_inputs.append(
                {
                    "input": _build_hint_messages(
                        self._build_conversation_summary(data_point),
                        selected_pattern,
                        self._patterns[selected_pattern]["hint"],
                    ),
                    "gen_kwargs": {
                        "max_new_tokens": 256,
                        "response_format": self._hint_response_format,
                    },
                    "reference": data_point,
                    "task_name": data_point.task_name,
                    "eligible_patterns": eligible_patterns,
                    "reasoning": reasoning,
                    "selected_pattern": selected_pattern,
                }
            )

        if not hint_inputs:
            return []

        # --- Call 2: hint generation ---
        hint_outputs = self._generator(
            hint_inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True
        )

        # --- Assemble results ---
        results = []
        for hint_out in hint_outputs:
            # Extract reference fields
            data_point = hint_out.get("reference")
            eligible_patterns = hint_out.get("eligible_patterns")
            reasoning = hint_out.get("reasoning")
            selected_pattern = hint_out.get("selected_pattern")

            # Parse hint result
            hint_result = hint_out.get("result") or ""
            if isinstance(hint_result, dict):
                hint_result = hint_result.get("content") or ""
            raw_hint = hint_result.strip()
            hint = None
            try:
                hint = json.loads(raw_hint).get("hint", "").strip() or None
            except (json.JSONDecodeError, AttributeError):
                hint = raw_hint or None

            # Fall back to static hint template if LM hint generation failed.
            # If no static hint exists, fall back to a generic placeholder.
            if not hint:
                hint = self._patterns[selected_pattern]["hint"] or (
                    f"Generate a user utterance following the '{selected_pattern}' pattern."
                )

            terminate = selected_pattern in self._termination_patterns

            data_point.steps.append(
                FlowControllerStep(
                    content=selected_pattern,
                    stage_name=self.name,
                    terminate=terminate,
                    hint=hint,
                    reasoning=reasoning,
                    eligible_patterns=eligible_patterns,
                )
            )
            results.append(data_point)

        return results

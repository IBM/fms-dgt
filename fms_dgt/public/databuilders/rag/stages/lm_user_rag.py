# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List
import json

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

_USER_RAG_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "Step-by-step reasoning: what does the scenario need, what has already "
                "been covered, and how should the next utterance advance the conversation?"
            ),
        },
        "proposed_utterance": {
            "type": "string",
            "description": "Initial draft of the user's next message.",
        },
        "verification": {
            "type": "string",
            "enum": ["yes", "no"],
            "description": (
                "Does the proposed utterance advance the scenario without repeating "
                "a previous question? 'yes' or 'no'."
            ),
        },
        "verification_reasoning": {
            "type": "string",
            "description": (
                "Explanation of the verification decision. If 'no', explain what "
                "needs to change."
            ),
        },
        "utterance": {
            "type": "string",
            "description": (
                "Final user utterance. If verification is 'yes', this may be a "
                "sharpened version of proposed_utterance. If verification is 'no', "
                "this must be a corrected utterance that fixes the issue identified "
                "in verification_reasoning."
            ),
        },
    },
    "required": [
        "reasoning",
        "proposed_utterance",
        "verification",
        "verification_reasoning",
        "utterance",
    ],
    "additionalProperties": False,
}

_USER_RAG_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_user_turn",
        "strict": True,
        "schema": _USER_RAG_SCHEMA,
    },
}


def _render_documents(documents: List[dict]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        text = doc.get("text", "")
        doc_id = doc.get("id", str(i))
        parts.append(f"[Document {i} | id={doc_id}]\n{text}")
    return "\n\n".join(parts) if parts else ""


def _build_user_rag_messages(
    scenario: str,
    persona_text: str,
    pattern: str,
    hint: str | None,
    documents: List[dict],
    history_steps: list,
) -> list:
    doc_text = _render_documents(documents)
    system_content = (
        "You are simulating a user in a conversation with an AI assistant. "
        "Your goal is to resolve the scenario described below by asking questions "
        "grounded in the provided documents. Stay in character at all times.\n\n"
        "Rules:\n"
        "- Ask only one focused question per turn.\n"
        "- Do not repeat a question you have already asked.\n"
        "- Do not ask about information not present or inferable from the documents.\n"
        "- Keep the utterance concise and natural.\n"
        "- Do not reveal that you are an AI or that you are simulating a user."
    )
    if scenario:
        system_content += f"\n\nScenario:\n{scenario}"
    if persona_text:
        system_content += f"\n\nYour persona:\n{persona_text}"
    if pattern:
        system_content += f"\n\nInteraction pattern for this turn: {pattern}"
    if hint:
        system_content += f"\n\nHint: {hint}"
    if doc_text:
        system_content += f"\n\nDocuments available to ground your questions:\n{doc_text}"

    messages = [{"role": "system", "content": system_content}]

    # Invert roles: from the simulated user's perspective, assistant turns are
    # "user" (the other side) and user turns are "assistant" (its own prior turns).
    # Tool call/result pairs are included so the model sees what was retrieved.
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

    messages.append(
        {
            "role": "user",
            "content": (
                "Generate the next user utterance following the interaction pattern and hint. "
                "First reason through what the scenario needs and what has already been covered. "
                "Propose an utterance, verify it does not repeat a prior question and advances "
                "the scenario, then produce the final utterance."
            ),
        }
    )
    return messages


def _parse_user_output(raw: str) -> str:
    """Extract the final utterance from structured output, with fallback to plain text."""
    try:
        parsed = json.loads(raw)
        utterance = parsed.get("utterance", "").strip()
        if utterance:
            return utterance
        # Verification failed and model left utterance empty — fall back to proposed.
        proposed = parsed.get("proposed_utterance", "").strip()
        if proposed:
            return proposed
    except (json.JSONDecodeError, AttributeError):
        pass
    return raw.strip()


@register_stage("lm/user/rag")
class RAGUserStage(Stage):
    """RAG-specific user stage with inline verify-then-refine loop.

    Extends the generic guided user stage with:
    - Document context injected into the system prompt so the model can
      generate grounded, non-hallucinated questions.
    - Single-call reflective loop: the model proposes an utterance, verifies
      it is non-repetitive and advances the scenario, then produces a final
      utterance (corrected if verification failed, optionally sharpened if it
      passed). Structured output via JSON schema.

    Reads:
      - Latest ``RAGScenarioStep`` (scenario text + documents)
      - Latest ``PersonaStep`` with target="user"
      - Latest ``FlowControllerStep`` (pattern name + hint)
      - All prior user/assistant steps (conversation history)

    Appends a ``UserStep``. Drops the data point if the LM returns empty output.
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
            documents = getattr(scenario_step, "documents", []) or []

            persona_steps = [
                step
                for step in data_point.steps
                if isinstance(step, PersonaStep) and step.target == "user"
            ]
            persona_text = render_persona(persona_steps[-1]) if persona_steps else ""

            fc_step = get_last_step_of_type(data_point.steps, FlowControllerStep)
            pattern = fc_step.content if fc_step else ""
            hint = fc_step.hint if fc_step else None

            history_steps = [
                step
                for step in data_point.steps
                if isinstance(step, (UserStep, AssistantStep, ToolCallStep, ToolResultStep))
            ]
            generator_inputs.append(
                {
                    "input": _build_user_rag_messages(
                        scenario, persona_text, pattern, hint, documents, history_steps
                    ),
                    "gen_kwargs": {
                        "max_new_tokens": 512,
                        "response_format": _USER_RAG_RESPONSE_FORMAT,
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
            user_text = _parse_user_output(result)
            if not user_text:
                continue
            data_point: ConversationDataPoint = out["reference"]
            data_point.steps.append(UserStep(content=user_text, stage_name=self.name))
            results.append(data_point)
        return results

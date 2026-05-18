# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json
import logging

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    Step,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_last_step_of_type,
    steps_to_messages,
)
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep
from fms_dgt.public.databuilders.rag.utils import render_documents

logger = logging.getLogger(__name__)

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
_MAX_RETRIES = 3


_CRITIQUE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_critique",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "For each factual claim in the assistant response, quote the specific "
                        "sentence or sentences from the documents that ground it, or explain "
                        "the logical derivation from document content. If no document sentence "
                        "grounds a claim and it requires outside knowledge to reach, mark it "
                        "as unsupported. Refusals and partial answer acknowledgments "
                        "(e.g. 'the documents do not cover this') are always supported."
                    ),
                },
                "valid": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": (
                        "Is every factual claim in the assistant response grounded in the "
                        "provided documents, either as a direct statement or a logical "
                        "derivation from what the documents say, or explicitly acknowledged "
                        "as outside document scope? Answer 'yes' or 'no'."
                    ),
                },
                "issues": {
                    "type": "string",
                    "description": (
                        "If valid is 'no', describe the specific ungrounded claims. "
                        "If valid is 'yes', leave empty."
                    ),
                },
            },
            "required": ["reasoning", "valid", "issues"],
            "additionalProperties": False,
        },
    },
}


# ===========================================================================
#                       HELPER METHODS
# ===========================================================================


def _build_generate_input(
    documents: List[dict],
    steps: List[Step],
) -> list:
    system_content = (
        "You are a helpful AI assistant. Answer questions using only the provided documents.\n\n"
        "Follow these rules strictly:\n"
        "- If the documents fully answer the question, answer directly and completely.\n"
        "- If the documents partially answer the question, state what the documents do say, "
        "then clearly note what specific information is missing. Do not invent or infer details "
        "beyond what the documents contain.\n"
        "- Only say the documents cannot answer if there is genuinely no relevant information "
        "at all (not merely because the answer is incomplete).\n"
        "Never fabricate facts, numbers, names, or policies not present in the documents."
        f"\n\nContext documents:\n{render_documents(documents)}"
    )
    messages = [{"role": "system", "content": system_content}]
    messages.extend(steps_to_messages(steps))
    return messages


def _build_critique_input(
    conversation: List[dict],
    assistant_response: str,
    documents: List[dict],
) -> list:
    doc_text = render_documents(documents)
    system_content = (
        "You are a strict grounding evaluator for RAG assistant responses. "
        "Your job is to verify that every factual claim in the assistant response is grounded "
        "in the provided documents.\n\n"
        "A response is valid if:\n"
        "- Every factual claim is grounded in the documents, either as a direct statement or "
        "a logical derivation from what the documents say.\n"
        "- Any gap is explicitly framed as a refusal or partial answer (e.g. 'the documents "
        "do not cover this', 'based on the documents I can only confirm...').\n\n"
        "A response is NOT valid if:\n"
        "- Any factual claim requires domain knowledge or background facts not present "
        "in the documents to reach. Pure logical or definitional derivations from "
        "document content are acceptable.\n"
        "- It presents speculative or uncertain information as definitive fact without "
        "the documents supporting that certainty.\n\n"
        f"Documents:\n{doc_text}"
    )
    messages = [{"role": "system", "content": system_content}]
    messages.extend(conversation)
    messages.append({"role": "assistant", "content": assistant_response})
    messages.append(
        {
            "role": "user",
            "content": (
                "Evaluate the assistant response above. For each factual claim, quote the "
                "specific sentence or sentences from the documents that support it. If no "
                "sentence supports a claim, note it as unsupported. "
                "Return a JSON object following the schema."
            ),
        }
    )
    return messages


def _build_rewrite_input(
    documents: List[dict],
    steps: List[Step],
    failed_response: str,
    issues: str,
) -> list:
    system_content = (
        "You are a helpful AI assistant. Answer questions using only the provided documents.\n\n"
        "Follow these rules strictly:\n"
        "- If the documents fully answer the question, answer directly and completely.\n"
        "- If the documents partially answer the question, state what the documents do say, "
        "then clearly note what specific information is missing. Do not invent or infer details "
        "beyond what the documents contain.\n"
        "- Only say the documents cannot answer if there is genuinely no relevant information "
        "at all (not merely because the answer is incomplete).\n"
        "Never fabricate facts, numbers, names, or policies not present in the documents."
        f"\n\nContext documents:\n{render_documents(documents)}"
    )
    messages = [{"role": "system", "content": system_content}]
    messages.extend(steps_to_messages(steps))

    messages.append({"role": "assistant", "content": failed_response})
    messages.append(
        {
            "role": "user",
            "content": (
                "[REWRITE REQUEST] The assistant response above does not meet grounding "
                f"requirements. {issues} "
                "Please rewrite the assistant response so that every claim is grounded "
                "in the provided documents, or explicitly states what the documents do "
                "not cover."
            ),
        }
    )
    return messages


def _parse_critique(raw: str) -> Dict[str, str]:
    try:
        parsed = json.loads(raw)
        return {
            "valid": parsed.get("valid", "no").strip().lower(),
            "issues": parsed.get("issues", "").strip(),
            "reasoning": parsed.get("reasoning", "").strip(),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"valid": "no", "issues": "Could not parse critique output.", "reasoning": ""}


@register_stage("lm/assistant/rag/static")
class StaticContextAssistantStage(Stage):
    """Pattern 1 assistant stage: generates grounded responses from static document context.

    Reads the document set from the latest ``RAGScenarioStep`` (``role="rag/scenario"``)
    and injects the documents into the system prompt. The assistant generates a
    response that is grounded in those documents.

    After each generation, a critique call using the same LM validates that every
    claim is traceable to the provided documents or explicitly acknowledged as outside
    scope. On failure, a rewrite is requested with the critique issues injected back
    into the conversation. This repeats up to ``max_retries`` times (default 1,
    capped at 3). Data points that fail all retries are dropped.

    No retrieval tool calls are produced. The conversation output contains only
    user and assistant turns, training grounded synthesis and faithfulness.

    Config kwargs:
        max_retries: Number of rewrite attempts on critique failure. Default 1, max 3.
    """

    def __init__(
        self,
        *,
        name: str,
        generator: LMProvider,
        max_retries: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name)
        self._generator = generator
        self._max_retries = min(max(0, max_retries), _MAX_RETRIES)

    def _generate(self, inputs: List[Dict], max_new_tokens: int = 1024) -> List[Dict]:
        for inp in inputs:
            inp["gen_kwargs"] = {"max_new_tokens": max_new_tokens}
        return self._generator(inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True)

    def _critique(self, inputs: List[Dict]) -> List[Dict]:
        for inp in inputs:
            inp["gen_kwargs"] = {
                "max_new_tokens": 512,
                "response_format": _CRITIQUE_RESPONSE_FORMAT,
            }
        return self._generator(inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True)

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs: Any,
    ) -> List[ConversationDataPoint]:
        active: List[Dict] = []
        for data_point in data_points:
            documents = []
            scenario_step: RAGScenarioStep = get_last_step_of_type(
                data_point.steps, RAGScenarioStep
            )
            if scenario_step:
                documents: List[dict] = scenario_step.documents

            active.append(
                {
                    "input": _build_generate_input(documents, data_point.steps),
                    "documents": documents,
                    "reference": data_point,
                    "task_name": data_point.task_name,
                    "attempt": 1,
                }
            )

        if not active:
            return []

        results = []

        while active:
            generated_outputs = self._generate(active)

            critique_inputs = []
            for entry in generated_outputs:
                parsed_output = entry.get("result") or ""
                if isinstance(parsed_output, dict):
                    parsed_output = parsed_output.get("content") or ""
                response = parsed_output.strip()
                if not response:
                    continue
                critique_inputs.append({**entry, "response": response})

            if not critique_inputs:
                break

            critique_outputs = self._critique(
                [
                    {
                        "input": _build_critique_input(
                            conversation=steps_to_messages(entry["reference"].steps),
                            assistant_response=entry["response"],
                            documents=entry["documents"],
                        ),
                        **entry,
                    }
                    for entry in critique_inputs
                ]
            )

            next_active = []
            for entry in critique_outputs:
                critique_result = entry.get("result") or ""
                if isinstance(critique_result, dict):
                    critique_result = critique_result.get("content") or ""
                critique = _parse_critique(critique_result.strip())

                if critique["valid"] == "yes":
                    data_point: ConversationDataPoint = entry["reference"]
                    data_point.steps.append(
                        AssistantStep(
                            content=entry["response"],
                            stage_name=self.name,
                            metadata={
                                "critique_attempts": entry["attempt"],
                                "critique_reasoning": critique["reasoning"],
                            },
                        )
                    )
                    results.append(data_point)
                elif entry["attempt"] <= self._max_retries:
                    next_active.append(
                        {
                            **entry,
                            "input": _build_rewrite_input(
                                entry["documents"],
                                entry["reference"].steps,
                                entry["response"],
                                critique["issues"],
                            ),
                            "attempt": entry["attempt"] + 1,
                        }
                    )
                else:
                    logger.debug(
                        "Dropping conversation %s after %d attempts.\n"
                        "  Issues: %s\n"
                        "  Reasoning: %s",
                        entry["reference"].conversation_id,
                        entry["attempt"],
                        critique["issues"],
                        critique["reasoning"],
                    )

            active = next_active

        return results

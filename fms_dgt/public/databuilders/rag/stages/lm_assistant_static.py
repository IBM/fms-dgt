# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    ScenarioStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import get_last_step_of_type
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep


def _render_documents(documents: List[dict]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        text = doc.get("text", "")
        doc_id = doc.get("id", str(i))
        parts.append(f"[Document {i} | id={doc_id}]\n{text}")
    return "\n\n".join(parts) if parts else "(no documents provided)"


def _build_static_assistant_messages(
    scenario: str,
    documents: List[dict],
    history_steps: list,
) -> list:
    doc_text = _render_documents(documents)
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
        f"\n\nContext documents:\n{doc_text}"
    )
    if scenario:
        system_content = f"Scenario: {scenario}\n\n" + system_content
    messages = [{"role": "system", "content": system_content}]
    for step in history_steps:
        if isinstance(step, UserStep):
            messages.append({"role": "user", "content": step.content})
        elif isinstance(step, AssistantStep):
            messages.append({"role": "assistant", "content": step.content})
    return messages


@register_stage("lm/assistant/rag/static")
class StaticContextAssistantStage(Stage):
    """Pattern 1 assistant stage: generates grounded responses from static document context.

    Reads the document set from the latest ``RAGScenarioStep`` (``role="rag/scenario"``)
    and injects the documents into the system prompt. The assistant generates a
    response that is grounded in those documents.

    No retrieval tool calls are produced. The conversation output contains only
    user and assistant turns, training grounded synthesis and faithfulness.
    """

    def __init__(self, *, name: str, generator: LMProvider, **kwargs: Any) -> None:
        super().__init__(name=name)
        self._generator = generator

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs: Any,
    ) -> List[ConversationDataPoint]:
        generator_inputs = []
        for data_point in data_points:
            rag_step = get_last_step_of_type(data_point.steps, RAGScenarioStep)
            if rag_step:
                scenario = rag_step.content
                documents = rag_step.documents
            else:
                scenario_step = get_last_step_of_type(data_point.steps, ScenarioStep)
                scenario = scenario_step.content if scenario_step else ""
                documents = []

            history_steps = [
                step for step in data_point.steps if isinstance(step, (UserStep, AssistantStep))
            ]
            generator_inputs.append(
                {
                    "input": _build_static_assistant_messages(scenario, documents, history_steps),
                    "gen_kwargs": {"max_new_tokens": 1024},
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
            assistant_text = result.strip()
            if not assistant_text:
                continue
            data_point: ConversationDataPoint = out["reference"]
            data_point.steps.append(AssistantStep(content=assistant_text, stage_name=self.name))
            results.append(data_point)
        return results

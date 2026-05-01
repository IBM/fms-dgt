# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    ScenarioStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.lm_flow_controller import (
    LMFlowControllerStage,
)
from fms_dgt.core.databuilders.conversation.utils import (
    get_last_step_of_type,
    steps_to_text,
)
from fms_dgt.public.databuilders.rag.utils import render_documents


@register_stage("lm/flow_controller/rag")
class RAGFlowControllerStage(LMFlowControllerStage):
    """RAG-specific flow controller that includes document text in the conversation
    summary so the eligibility reasoning can assess patterns like rag/comparative,
    rag/ambiguous, and rag/unanswerable against actual document content.

    All other behavior (JSON schema output, verify-then-select loop, hint generation,
    termination handling) is inherited from LMFlowControllerStage.
    """

    def _build_conversation_summary(self, context: ConversationDataPoint) -> str:
        lines = []
        scenario_step = get_last_step_of_type(context.steps, ScenarioStep)
        if scenario_step:
            lines.append(f"[Scenario]\n{scenario_step.content}")
            documents = getattr(scenario_step, "documents", []) or []
            if documents:
                doc_text = render_documents(documents)
                lines.append(f"\n[Available Documents]\n{doc_text}")
        history = steps_to_text(context.steps)
        if history:
            lines.append(f"\n[Conversation History]\n{history}")
        else:
            lines.append("\n[Conversation History]\n(no turns yet — conversation has not started)")
        return "\n".join(lines)

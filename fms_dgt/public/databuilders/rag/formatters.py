# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.utils import get_last_step_of_type
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep


@register_formatter("formatters/conversation/rag/static")
class StaticRAGMessagesFormatter(Formatter):
    """Formats a Pattern 1 (static context) RAG conversation for SFT.

    Output shape:
        {
            "conversation_id": "...",
            "messages": [{"role": "user", "content": "..."}, ...],
            "documents": [{"id": "...", "text": "...", ...}, ...]
        }

    ``documents`` is lifted from the last ``RAGScenarioStep`` in the
    conversation, matching the convention used by the assistant and flow
    controller stages, which also read from the last scenario step. This
    ensures the exported documents are the ones that were actually active
    during generation, regardless of how many scenario steps are present.

    This formatter is intentionally scoped to Pattern 1. For Pattern 2 (live
    retrieval), use ``formatters/conversation/messages`` — tool call and tool
    result steps are already real events in that pipeline and need no special
    handling.

    Contract: this formatter assumes the document set is fixed for the entire
    conversation (set once during initialization, never changed). If you write
    a custom flow controller or scenario stage that mutates or refreshes
    documents mid-conversation, this formatter will produce incorrect output
    and you must write a new one that handles per-turn document attribution.

    Pipeline-internal steps (scenario, persona, flow_controller) are stripped
    from ``messages``. The ``documents`` field is passed through verbatim from
    the ``RAGScenarioStep`` so no information is lost. The downstream consumer
    decides how to inject documents into the training prompt (system message,
    synthetic tool result, separate context field, etc.).
    """

    def apply(self, data: ConversationDataPoint, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        scenario_step = get_last_step_of_type(data.steps, RAGScenarioStep)
        documents: List[Dict[str, Any]] = scenario_step.documents if scenario_step else []

        messages: List[Dict[str, Any]] = []
        for step in data.steps:
            if isinstance(step, UserStep):
                messages.append({"role": "user", "content": step.content})
            elif isinstance(step, AssistantStep):
                messages.append({"role": "assistant", "content": step.content})

        return {
            "conversation_id": data.conversation_id,
            "messages": messages,
            "documents": documents,
        }

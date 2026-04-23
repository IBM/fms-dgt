# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Local
from fms_dgt.core.databuilders.conversation.data_objects import ScenarioStep
from fms_dgt.core.databuilders.conversation.registry import register_step


@register_step("rag/scenario")
@dataclass
class RAGScenarioStep(ScenarioStep):
    """Conversation scenario step for RAG recipes.

    Carries the document set selected during Pattern 1 (static context)
    initialization. Downstream stages (user, assistant) read ``documents``
    from this step when generating grounded conversation turns.

    For Pattern 2 (live retrieval) the ``documents`` list is empty — document
    selection happens at conversation time via the model's retrieval tool call.
    """

    role: str = field(default="rag/scenario", init=False)

    documents: List[Dict[str, Any]] = field(default_factory=list)
    """Documents selected by the DocumentSampler during initialization.

    Each entry is a serialized Document dict with at minimum ``id`` and
    ``text`` fields, and an optional ``score``. Empty for Pattern 2 recipes.
    """

    reasoning: str | None = None
    """Optional free-text rationale recorded by the scenario stage (e.g. why
    these documents were selected or what topic domain they cover)."""

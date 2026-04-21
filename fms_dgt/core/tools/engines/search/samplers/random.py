# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Optional

# Local
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.search.base import Document
from fms_dgt.core.tools.engines.search.samplers.base import (
    DocumentSampler,
    register_document_sampler,
)


@register_document_sampler("search/random")
class RandomDocumentSampler(DocumentSampler):
    """Selects documents uniformly at random from the corpus.

    Passes an empty query to the engine.  Appropriate for baseline coverage
    recipes where topical relevance does not matter.  ``FileSearchEngine`` is
    the natural pairing since it already selects randomly regardless of query.

    This sampler is thread-safe: no instance state is modified during
    ``sample()``.
    """

    def sample(
        self,
        session_id: str,
        k: int,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        tc = ToolCall(name="sample", arguments={"size": k}, call_id=None)
        results = self._engine.simulate(session_id=session_id, tool_calls=[tc])
        if not results or results[0].result is None:
            return []
        raw = results[0].result
        if not isinstance(raw, list):
            return []
        return [
            Document(
                id=str(item.get("id", i)),
                text=str(item.get("text", "")),
                score=item.get("score"),
                metadata={k: v for k, v in item.items() if k not in ("id", "text", "score")},
            )
            for i, item in enumerate(raw)
        ]

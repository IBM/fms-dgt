# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import random

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import (
    ErrorCategory,
    ToolEngine,
)
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ===========================================================================
#                       DOCUMENT DATA MODEL
# ===========================================================================


@dataclass
class Document:
    """A single retrieved document chunk.

    Attributes:
        id: Stable identifier for this chunk within its corpus.
        text: The text content of the chunk.
        score: Relevance score (higher is better), or ``None`` if the engine
            does not rank results (e.g. random sampling).
        metadata: Engine-specific metadata (source URL, title, field values).
    """

    id: str
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "text": self.text}
        if self.score is not None:
            d["score"] = self.score
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ===========================================================================
#                       SEARCH ENGINE BASE
# ===========================================================================


class SearchToolEngine(ToolEngine):
    """Base class for all retrieval execution backends.

    Specializes ``ToolEngine`` for retrieval operations. Retriever engines are
    stateless across calls — ``execute()`` does not mutate session state.  The
    session interface is inherited from ``ToolEngine`` but ``setup``/``teardown``
    are no-ops unless a subclass overrides them (e.g. for corpus loading or
    connection pooling).

    Subclasses must implement ``_search()``, which receives parsed query
    arguments and returns a list of ``Document`` objects.  The base class
    handles argument parsing, relevance filtering, result formatting, and
    error injection.

    ``simulate()`` delegates to ``execute()`` — retrieval is stateless so
    there is no session history to roll back.

    Args:
        registry: Shared ``ToolRegistry`` for name lookup.
        relevance_threshold: If set, documents with ``score < threshold`` are
            filtered out before results are returned to the caller.
        error_categories: Optional list of error-injection descriptors (dicts
            or ``ErrorCategory`` instances).
        namespaces: Namespace restriction, forwarded to ``ToolEngine``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        relevance_threshold: Optional[float] = None,
        error_categories: Optional[List[Dict]] = None,
        namespaces: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, namespaces=namespaces)
        self._relevance_threshold = relevance_threshold
        self._error_categories: List[ErrorCategory] = [
            ec if isinstance(ec, ErrorCategory) else ErrorCategory(**ec)
            for ec in (error_categories or [])
        ]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _search(self, arguments: Dict[str, Any], limit: int, **kwargs: Any) -> List[Document]:
        """Execute a retrieval call and return raw documents.

        Args:
            arguments: Parsed tool call arguments (e.g. ``{"query": "..."}``)
            limit: Maximum number of documents to return.

        Returns:
            List of ``Document`` objects, ordered by relevance where supported.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # execute / simulate
    # ------------------------------------------------------------------

    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a batch of retrieval tool calls.

        Steps per call:
          1. Error injection check — if an error fires, return immediately.
          2. Parse arguments to extract ``query`` and optional ``size``/``limit``.
          3. Delegate to ``_search()``.
          4. Apply ``relevance_threshold`` filter if configured.
          5. Format as ``ToolResult``.

        Args:
            session_id: Active session (used for corpus-loading engines that
                need a per-session handle; ignored by stateless engines).
            tool_calls: Retrieval calls to process.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.
        """
        results = []
        for tc in tool_calls:
            results.append(self._execute_one(tc))
        return results

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Identical to ``execute()`` — retrieval is stateless, no rollback needed."""
        return self.execute(session_id, tool_calls)

    # ------------------------------------------------------------------
    # Error injection
    # ------------------------------------------------------------------

    def _inject_error(
        self,
        tc: ToolCall,
        category: Optional[ErrorCategory] = None,
    ) -> Optional[ToolResult]:
        """Probabilistically inject a simulated error.

        If ``category`` is ``None``, samples all registered error categories
        independently and picks one at random if any fired.  If ``category``
        is provided, applies it directly (used by subclasses that want to
        handle custom types and delegate base types to super).

        Args:
            tc: The originating tool call — provides ``call_id`` and ``name``
                for the returned ``ToolResult``, and ``arguments`` for
                subclasses that need to inspect them (e.g. ``index_not_found``
                reads ``tc.arguments["index"]``).
            category: If supplied, skip sampling and apply this category
                directly.

        Returns:
            A fully-populated ``ToolResult`` representing the error, or
            ``None`` if no error was injected.
        """
        if category is None:
            fired = [ec for ec in self._error_categories if ec.should_fire()]
            if not fired:
                return None
            category = random.choice(fired)

        match category.type:
            case "network_error":
                return ToolResult(
                    call_id=tc.call_id,
                    name=tc.name,
                    result=None,
                    error=category.message or "Connection timed out",
                )
            case "empty_result":
                return ToolResult(call_id=tc.call_id, name=tc.name, result=[], error=None)
            case "unparseable_result":
                return ToolResult(call_id=tc.call_id, name=tc.name, result="<garbled>", error=None)
            case _:
                return ToolResult(
                    call_id=tc.call_id,
                    name=tc.name,
                    result=None,
                    error=f"Simulated error: {category.type}",
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_one(self, tc: ToolCall) -> ToolResult:
        arguments = tc.arguments or {}

        error_result = self._inject_error(tc)
        if error_result is not None:
            return error_result

        limit = int(arguments.get("size", arguments.get("limit", self._default_limit())))

        try:
            docs = self._search(arguments, limit=limit)
        except Exception as exc:
            logger.warning("SearchToolEngine._search() raised: %s", exc)
            return ToolResult(call_id=tc.call_id, name=tc.name, error=str(exc))

        if self._relevance_threshold is not None:
            docs = [d for d in docs if d.score is None or d.score >= self._relevance_threshold]

        return ToolResult(
            call_id=tc.call_id,
            name=tc.name,
            result=[d.to_dict() for d in docs],
        )

    def _default_limit(self) -> int:
        """Default number of documents to return when ``size`` is not in arguments."""
        return 5

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import ToolEngine
from fms_dgt.core.tools.registry import ToolRegistry

# ===========================================================================
#                       COMPOSITE TOOL ENGINE
# ===========================================================================


class CompositeToolEngine(ToolEngine):
    """Composite engine that routes ``ToolCall`` objects by namespace.

    A ``ToolRegistry`` is always multi-namespace by design, so there is no
    separate ``MultiServerToolRegistry`` class.  This engine holds one
    ``ToolEngine`` per namespace and routes calls to the right sub-engine by
    splitting the qualified name.

    ``setup`` and ``teardown`` fan out to all sub-engines so the caller only
    manages lifecycle on this composite object.

    Args:
        registry: The shared ``ToolRegistry`` for this engine.
        engines: Mapping from namespace string to ``ToolEngine`` instance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        engines: Dict[str, ToolEngine],
    ) -> None:
        super().__init__(registry)
        self._engines = dict(engines)
        self._distinct_engines = {id(engine): engine for engine in self._engines.values()}.values()

    @property
    def engines(self) -> Dict[str, ToolEngine]:
        """Mapping from namespace to sub-engine."""
        return dict(self._engines)

    # ------------------------------------------------------------------
    # Lifecycle — fan out to sub-engines
    # ------------------------------------------------------------------

    def setup(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        super().setup(session_id, *args, **kwargs)
        for engine in self._distinct_engines:
            engine.setup(session_id, *args, **kwargs)

    def teardown(self, session_id: str) -> None:
        super().teardown(session_id)
        for engine in self._distinct_engines:
            engine.teardown(session_id)

    def get_session_state(self, session_id: str) -> Dict[str, Any] | None:
        """Return aggregated session state keyed by namespace.

        Sub-engines that return ``None`` are omitted.  Returns ``None`` if the
        session is unknown on this composite engine.
        """
        if session_id not in self._sessions:
            return None
        aggregated: Dict[str, Any] = {}
        for ns, engine in self._engines.items():
            state = engine.get_session_state(session_id)
            if state is not None:
                aggregated[ns] = state
        return aggregated if aggregated else None

    # ------------------------------------------------------------------
    # Execution — route by namespace
    # ------------------------------------------------------------------

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Dispatch simulate to each sub-engine by namespace.

        Pure read — no session state is modified in any sub-engine.
        Results are returned in the same order as ``tool_calls``.
        """
        return self._dispatch(session_id, tool_calls, method="simulate")

    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Dispatch each tool call to the appropriate sub-engine by namespace.

        Calls sharing the same namespace are batched together.  Results are
        returned in the same order as ``tool_calls``.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to dispatch (may span multiple namespaces).

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the original order.

        Raises:
            KeyError: If ``session_id`` is not active.
            ValueError: If a call's namespace has no registered sub-engine.
        """
        return self._dispatch(session_id, tool_calls, method="execute")

    # ------------------------------------------------------------------
    # Shared dispatch helper
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        session_id: str,
        tool_calls: List[ToolCall],
        method: str,
    ) -> List[ToolResult]:
        """Route tool calls to sub-engines by namespace and reassemble results.

        Args:
            session_id: Active session.
            tool_calls: Calls to route.
            method: Either ``"simulate"`` or ``"execute"``.
        """
        # Validate the session exists on this composite engine before
        # dispatching to sub-engines.
        with self._session_transaction(session_id):
            pass

        # Bucket calls by namespace, preserving original indices for reassembly.
        buckets: Dict[str, List[tuple]] = {}
        for idx, tc in enumerate(tool_calls):
            ns = tc.namespace
            if ns not in buckets:
                buckets[ns] = []
            buckets[ns].append((idx, tc))

        # TODO: cross-engine history is not implemented. Each LMToolEngine
        # maintains its own isolated session history, so tool calls executed
        # by engine A are not visible in the prompt context of engine B. The
        # right fix is a HistoryAwareEngine protocol that LMToolEngine
        # implements; after each dispatch, MultiToolEngine would call an
        # inject_history_entry() method on all other history-aware engines.
        # Until that is in place, conversations that span multiple namespaces
        # will produce coherent results within each namespace but not across.
        results: List[ToolResult | None] = [None] * len(tool_calls)
        for ns, indexed_calls in buckets.items():
            engine = self._engines.get(ns)
            if engine is None:
                raise ValueError(
                    f"No engine registered for namespace '{ns}'. "
                    f"Registered namespaces: {list(self._engines.keys())}"
                )
            batch = [tc for _, tc in indexed_calls]
            batch_results = getattr(engine, method)(session_id, batch)
            for (orig_idx, _), result in zip(indexed_calls, batch_results):
                results[orig_idx] = result

        return results  # type: ignore[return-value]

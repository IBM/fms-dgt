# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional
import copy
import logging
import random
import threading

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ===========================================================================
#                       ERROR CATEGORIES
# ===========================================================================


@dataclass
class ErrorCategory:
    """Probabilistic error injection descriptor.

    Attributes:
        type: One of ``"network_error"``, ``"unparseable_result"``,
            ``"schema_violation"``.
        probability: Float in [0, 1] — chance this category fires on any
            given call.
        message: Optional human-readable error string.  Used for
            ``"network_error"`` results.
    """

    type: str
    probability: float
    message: str = "Tool execution failed"

    def should_fire(self) -> bool:
        return random.random() < self.probability


# ===========================================================================
#                       ENGINE REGISTRY
# ===========================================================================

_TOOL_ENGINE_REGISTRY: Dict[str, type] = {}


def register_tool_engine(*names: str):
    """Class decorator that registers a ToolEngine subclass under one or more names.

    Usage::

        @register_tool_engine("lm")
        class LMToolEngine(ToolEngine):
            ...

    Args:
        *names: One or more string aliases for the engine class.

    Raises:
        AssertionError: If a name is already registered or the class does not
            extend ``ToolEngine``.
    """

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, ToolEngine
            ), f"Engine '{name}' ({cls.__name__}) must extend ToolEngine"
            assert name not in _TOOL_ENGINE_REGISTRY, (
                f"Tool engine named '{name}' conflicts with an existing registration. "
                f"Use a non-conflicting alias."
            )
            _TOOL_ENGINE_REGISTRY[name] = cls
        return cls

    return decorate


def get_tool_engine(name: str, *args: Any, **kwargs: Any) -> "ToolEngine":
    """Instantiate a registered ToolEngine by name.

    Args:
        name: Registry key (e.g. ``"lm"``).
        *args: Positional arguments forwarded to the engine constructor.
        **kwargs: Keyword arguments forwarded to the engine constructor.

    Returns:
        An initialized ``ToolEngine`` instance.

    Raises:
        KeyError: If no engine is registered under ``name``.
    """
    if name not in _TOOL_ENGINE_REGISTRY:
        known = ", ".join(_TOOL_ENGINE_REGISTRY.keys())
        raise KeyError(f"Tool engine '{name}' not found. Registered engines: {known}")
    return _TOOL_ENGINE_REGISTRY[name](*args, **kwargs)


# ===========================================================================
#                       BASE ENGINE
# ===========================================================================


class ToolEngine(ABC):
    """Abstract base class for tool executors.

    ``ToolEngine`` manages a per-session state dict keyed by opaque session
    IDs and provides thread-safe access to that state through
    ``_session_transaction``.  Subclasses (``LMToolEngine``, REST engines,
    MCP engines) never touch locks directly — all locking is owned here.

    Session lifecycle::

        engine.setup(session_id)
        try:
            results = engine.execute(session_id, tool_calls)
        finally:
            engine.teardown(session_id)

    Args:
        catalog: The ``ToolRegistry`` this engine dispatches against.
    """

    def __init__(
        self,
        catalog: ToolRegistry,
        namespaces: Optional[List[str]] = None,
    ) -> None:
        self._catalog = catalog
        self._namespaces = namespaces  # None means no restriction
        self._sessions: Dict[str, Dict[str, Any]] = {}
        # Guards reads/writes to the _sessions dict itself (setup, teardown).
        self._sessions_lock = threading.Lock()

    @property
    def catalog(self) -> ToolRegistry:
        return self._catalog

    @property
    def namespaces(self) -> Optional[List[str]]:
        """Namespaces this engine is scoped to, or ``None`` for no restriction."""
        return self._namespaces

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def setup(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        """Create and register a new session.

        Pass ``initial_state=<dict>`` to fork an existing session's state
        (e.g. for DPO branching).  The forked session gets its own lock.

        Thread-safe: the check-then-set is atomic under ``_sessions_lock``.

        Args:
            session_id: Opaque string key for this session.
            *args: Forwarded to ``_init_session_state``.
            **kwargs: Forwarded to ``_init_session_state``.  The key
                ``initial_state`` (a dict) is intercepted here and used as
                the starting state for the session.

        Raises:
            ValueError: If ``session_id`` is already registered.
        """
        initial_state = kwargs.pop("initial_state", None)
        state = self._init_session_state(*args, **kwargs)
        if initial_state is not None:
            state.update(copy.deepcopy(initial_state))
        state["_lock"] = threading.Lock()
        with self._sessions_lock:
            if session_id in self._sessions:
                raise ValueError(
                    f"Session '{session_id}' is already active. "
                    f"Call teardown() before re-registering."
                )
            self._sessions[session_id] = state

    def teardown(self, session_id: str) -> None:
        """Remove a session.  Safe to call on an already-removed session.

        Args:
            session_id: Session to remove.
        """
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def get_session_state(self, session_id: str) -> Dict[str, Any] | None:
        """Return a consistent snapshot of the session state, or ``None``.

        Blocks until any in-progress transaction on this session completes,
        guaranteeing the snapshot is never torn mid-mutation.  The internal
        ``_lock`` key is excluded — locks are not serializable and each
        forked session gets its own.

        Args:
            session_id: Session to inspect.

        Returns:
            Deep-copy of the state dict (excluding ``_lock``), or ``None``
            if the session is not found.
        """
        try:
            with self._session_transaction(session_id) as state:
                return copy.deepcopy({k: v for k, v in state.items() if k != "_lock"})
        except KeyError:
            return None

    # ------------------------------------------------------------------
    # Transaction primitive — all locking lives here
    # ------------------------------------------------------------------

    @contextmanager
    def _session_transaction(
        self,
        session_id: str,
        *,
        rollback: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """Context manager for atomic, consistent session state access.

        Acquires the per-session lock for the duration of the block,
        yielding the live state dict.  If ``rollback=True``, the state is
        restored to a deep-copy snapshot taken on entry — useful for
        ``simulate`` which must leave persistent state unchanged.

        Subclasses call this instead of touching locks directly::

            # read-only or persistent mutation
            with self._session_transaction(session_id) as state:
                history = state["history"]
                ...

            # transient mutation (simulate)
            with self._session_transaction(session_id, rollback=True) as state:
                history = state["history"]
                ...  # appends here are rolled back on exit

        Args:
            session_id: Active session.
            rollback: If ``True``, restore state to its pre-entry snapshot
                before releasing the lock.

        Yields:
            Live session state dict.

        Raises:
            KeyError: If the session has not been set up.
        """
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(
                f"Session '{session_id}' is not active. " f"Call setup() before execute()."
            )
        snapshot = (
            copy.deepcopy({k: v for k, v in state.items() if k != "_lock"}) if rollback else None
        )

        with state["_lock"]:
            try:
                yield state
            finally:
                if rollback and snapshot is not None:
                    state.update(snapshot)
                    # Remove any keys added during the transaction.
                    for key in list(state):
                        if key != "_lock" and key not in snapshot:
                            del state[key]

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _init_session_state(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the initial state dict for a new session.

        Subclasses override this to add engine-specific state keys.
        Do not include ``_lock`` — the base class manages that.
        """
        return {}

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Generate tool results without updating session state.

        The default implementation delegates to ``execute`` and is provided
        as a convenience for stateless engines.  Stateful engines override
        this using ``_session_transaction(session_id, rollback=True)``.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to probe.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.

        Raises:
            KeyError: If ``session_id`` has not been set up.
        """
        return self.execute(session_id, tool_calls)

    @abstractmethod
    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a batch of tool calls in the context of a session.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to execute (may be parallel).

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.

        Raises:
            KeyError: If ``session_id`` has not been set up.
        """
        raise NotImplementedError

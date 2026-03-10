# Standard
from typing import List, Tuple
import logging
import threading


class FanOutHandler(logging.Handler):
    """A logging handler that forwards every record to a dynamic set of
    child handlers.

    Intended for use on the DataBuilder logger. Task file handlers are
    registered when a task becomes active and unregistered when it
    finishes. Every record emitted on the DataBuilder logger is then
    written to all currently-active task log files, making each task log
    self-contained even for run-level events (epoch banners, profiling
    summaries) that have no single task identity.

    Thread safety: the handler registry is protected by a lock so that
    register/unregister calls from concurrent task lifecycles do not race
    with emit calls from the generation loop.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # List of (task_name, handler) pairs.
        self._handlers: List[Tuple[str, logging.Handler]] = []

    def register(self, task_name: str, handler: logging.Handler) -> None:
        """Add a task's file handler to the fan-out set."""
        with self._lock:
            self._handlers.append((task_name, handler))

    def unregister(self, task_name: str) -> None:
        """Remove the handler registered under task_name."""
        with self._lock:
            self._handlers = [(name, h) for name, h in self._handlers if name != task_name]

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for _task_name, handler in handlers:
            handler.emit(record)

    def flush(self) -> None:
        with self._lock:
            handlers = list(self._handlers)
        for _task_name, handler in handlers:
            handler.flush()

    @property
    def active_task_names(self) -> List[str]:
        with self._lock:
            return [name for name, _ in self._handlers]

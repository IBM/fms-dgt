# Standard
from datetime import datetime, timezone
from typing import List, Tuple
import logging
import threading

# Local
from fms_dgt.base.datastore import Datastore

# Derive the set of built-in LogRecord keys from an actual instance so this
# stays correct across Python versions (e.g. taskName was added in 3.12).
# Anything in record.__dict__ that is NOT in this set came from extra={...}.
_LOGRECORD_BUILTIN_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


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
        """Add a task's file handler to the fan-out set.

        No-op if task_name is already registered, so callers can safely
        call register() a second time (e.g. from execute_tasks) without
        creating duplicate entries.
        """
        with self._lock:
            if any(name == task_name for name, _ in self._handlers):
                return
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


class LogDatastoreHandler(logging.Handler):
    """A logging handler that writes structured log records to a Datastore.

    Each Python log record is serialized as a structured dict and appended to
    the log datastore via ``save_data()``. The datastore backend is determined
    entirely by the task's datastore configuration: DefaultDatastore writes
    JSONL to ``output_dir/store_name/logs.jsonl``; any other backend (S3,
    custom) writes to the same logical path in its own store.

    This replaces the hardcoded ``FileHandler`` that was previously created
    only for the default datastore type, extending durable log persistence to
    all datastore backends uniformly.
    """

    def __init__(self, log_datastore: Datastore) -> None:
        super().__init__()
        self._log_datastore = log_datastore

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }
        # Preserve any structured attributes passed via extra={...} while
        # excluding the standard LogRecord fields already captured above.
        for key, val in record.__dict__.items():
            if key not in _LOGRECORD_BUILTIN_KEYS and not key.startswith("_") and key not in entry:
                entry[key] = val
        try:
            self._log_datastore.save_data([entry])
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the underlying log datastore before releasing the handler."""
        try:
            self._log_datastore.close()
        except Exception:
            pass
        super().close()

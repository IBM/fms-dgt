# Standard
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import logging
import os

# Local
from fms_dgt.constants import DGT_DIR
from fms_dgt.log.context import _run_ctx
from fms_dgt.utils import RotatingJsonlWriter

# ---------------------------------------------------------------------------
# Rotation / retention defaults
# ---------------------------------------------------------------------------
_MAX_BYTES_DEFAULT = 100 * 1024 * 1024  # 100 MB
_MAX_AGE_DAYS_DEFAULT = 14


def _telemetry_disabled() -> bool:
    val = os.environ.get("DGT_TELEMETRY_DISABLE", "").strip().lower()
    return val in ("1", "true", "yes")


def _telemetry_dir() -> str:
    return os.environ.get("DGT_TELEMETRY_DIR", os.path.join(DGT_DIR, "telemetry"))


# ===========================================================================
#                       OTel SDK — optional import
# ===========================================================================
try:
    # Third Party
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # noqa: F401

    _otel_available = True
except ImportError:
    _otel_available = False


class _NoOpSpan:
    """Returned by _NoOpTracer when OTel SDK is not installed."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, *_):
        pass

    def set_status(self, *_):
        pass

    def record_exception(self, *_):
        pass


class _NoOpTracer:
    def start_as_current_span(self, *_, **__):
        return _NoOpSpan()


if _otel_available:
    _tracer = _otel_trace.get_tracer("fms_dgt")
else:
    _tracer = _NoOpTracer()


# ===========================================================================
#                       SpanWriter
# ===========================================================================
class SpanWriter:
    """Writes span records to ``{telemetry_dir}/traces.jsonl``.

    Each span is a single JSON object appended on close.  The file rotates at
    100 MB and rotated files older than 14 days are deleted automatically.
    ``build_id`` and ``run_id`` on every record are the query keys across runs.

    When telemetry is disabled this class is replaced by ``_NoOpSpanWriter``.
    """

    def __init__(
        self,
        telemetry_dir: str,
        max_bytes: int = _MAX_BYTES_DEFAULT,
        max_age_days: int = _MAX_AGE_DAYS_DEFAULT,
    ) -> None:
        path = os.path.join(telemetry_dir, "traces.jsonl")
        self._writer = RotatingJsonlWriter(path, max_bytes, max_age_days)

    def write(self, record: Dict[str, Any]) -> None:
        self._writer.write(json.dumps(record, default=str))


class _NoOpSpanWriter:
    """Returned by configure_telemetry() when DGT_TELEMETRY_DISABLE is set."""

    def write(self, record: Dict[str, Any]) -> None:
        pass


# ===========================================================================
#                       Span context manager
# ===========================================================================
class Span:
    """A lightweight span context manager.

    On exit it serializes itself to the ``SpanWriter`` and, when the OTel SDK
    is present, delegates to the OTel tracer for full distributed-tracing
    support.

    Usage::

        with Span("dgt.block", writer, block_name="my_block") as span:
            result = self.execute(inputs)
            span.set_attribute("output_count", len(result))
    """

    def __init__(
        self,
        name: str,
        writer: SpanWriter,
        build_id: str = "",
        run_id: str = "",
        parent_span_name: Optional[str] = None,
        **attributes,
    ) -> None:
        self._name = name
        self._writer = writer
        self._build_id = build_id
        self._run_id = run_id
        self._parent_span_name = parent_span_name
        self._attributes: Dict[str, Any] = dict(attributes)
        self._start: Optional[datetime] = None
        self._status = "ok"

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def set_error(self, exc: Exception) -> None:
        self._status = "error"
        self._attributes["error"] = str(exc)
        self._attributes["error_type"] = type(exc).__name__

    def __enter__(self) -> "Span":
        self._start = datetime.now(tz=timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, _tb) -> None:
        end = datetime.now(tz=timezone.utc)
        if exc_val is not None:
            self.set_error(exc_val)
        duration_ms = round((end - self._start).total_seconds() * 1000, 2)
        # Auto-inject build_id/run_id from the active run context when they
        # were not passed explicitly (or were passed as empty string).  This
        # means callers never need to forward them manually; the ContextVar is
        # the single source of truth.
        build_id = self._build_id
        run_id = self._run_id
        if not build_id or not run_id:
            ctx = _run_ctx.get()
            if ctx:
                build_id = build_id or ctx["build_id"]
                run_id = run_id or ctx["run_id"]
        record = {
            "span_name": self._name,
            "build_id": build_id,
            "run_id": run_id,
            "start_time": self._start.isoformat(),
            "end_time": end.isoformat(),
            "duration_ms": duration_ms,
            "parent_span_name": self._parent_span_name,
            "status": self._status,
            **self._attributes,
        }
        self._writer.write(record)
        return False  # do not suppress exceptions


# ===========================================================================
#                       TelemetryEventHandler
# ===========================================================================
class TelemetryEventHandler(logging.Handler):
    """A logging handler that filters structured lifecycle events into
    ``{telemetry_dir}/events.jsonl``.

    Only records that carry an ``event`` key in their ``extra`` payload are
    written.  Plain log records without an ``event`` key pass through silently
    (they belong in the task log file, not in the telemetry event stream).

    The file rotates at 100 MB; rotated files older than 14 days are deleted.

    Attach to the builder logger (not per-task loggers) so each event is
    written once, with no fan-out duplication.
    """

    # Keys present on every LogRecord that we do not want to repeat in the
    # telemetry event payload.
    _SKIP_KEYS = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys() | {"message", "event"}
    )

    def __init__(
        self,
        telemetry_dir: str,
        max_bytes: int = _MAX_BYTES_DEFAULT,
        max_age_days: int = _MAX_AGE_DAYS_DEFAULT,
    ) -> None:
        super().__init__()
        path = os.path.join(telemetry_dir, "events.jsonl")
        self._writer = RotatingJsonlWriter(path, max_bytes, max_age_days)

    def emit(self, record: logging.LogRecord) -> None:
        event_name = getattr(record, "event", None)
        if event_name is None:
            return

        entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "event": event_name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in self._SKIP_KEYS and not key.startswith("_") and key not in entry:
                entry[key] = val
        # Auto-inject build_id/run_id from the active run context if the
        # event payload does not already carry them.
        if not entry.get("build_id") or not entry.get("run_id"):
            ctx = _run_ctx.get()
            if ctx:
                entry.setdefault("build_id", ctx["build_id"])
                entry.setdefault("run_id", ctx["run_id"])

        try:
            self._writer.write(json.dumps(entry, default=str))
        except Exception:
            self.handleError(record)


# ===========================================================================
#                       configure_telemetry
# ===========================================================================
def configure_telemetry(
    builder_logger: logging.Logger,
    build_id: str,
    run_id: str,
    max_bytes: int = _MAX_BYTES_DEFAULT,
    max_age_days: int = _MAX_AGE_DAYS_DEFAULT,
) -> SpanWriter:
    """Initialize the telemetry layer for a run.

    Called once from ``generate_data.py`` before ``execute_tasks()``.  Attaches
    a ``TelemetryEventHandler`` to the builder logger and returns a
    ``SpanWriter`` for use at instrumentation sites (block, epoch, run spans).

    Both sinks rotate at ``max_bytes`` (default 100 MB) and delete rotated
    files older than ``max_age_days`` (default 14 days).

    When ``DGT_TELEMETRY_DISABLE`` is set, returns a no-op ``SpanWriter`` and
    attaches nothing.  The ``telemetry/`` directory is never created.

    Args:
        builder_logger: The builder-scoped logger (``data_builder.logger``).
            ``TelemetryEventHandler`` is attached here so each lifecycle event
            is written once, with no fan-out duplication.
        build_id: The current build identifier, embedded in every record.
        run_id: The current run identifier, embedded in every record.
        max_bytes: Rotate files at this size. Default 100 MB.
        max_age_days: Delete rotated files older than this many days. Default 14.

    Returns:
        A ``SpanWriter`` instance (or no-op) for the caller to pass to
        instrumentation sites.
    """
    if _telemetry_disabled():
        return _NoOpSpanWriter()

    tdir = _telemetry_dir()
    event_handler = TelemetryEventHandler(tdir, max_bytes, max_age_days)
    builder_logger.addHandler(event_handler)

    return SpanWriter(tdir, max_bytes, max_age_days)

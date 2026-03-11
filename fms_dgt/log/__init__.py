"""fms_dgt.log — central home for all DGT logging infrastructure.

Named 'log' rather than 'logging' to avoid shadowing the Python stdlib
logging module, which would cause import errors inside this package.

Modules
-------
handlers
    FanOutHandler       — fan-out to dynamic set of task log handlers
    LogDatastoreHandler — writes structured JSONL records to a Datastore
filters
    RunContextFilter    — injects build_id / run_id onto every log record

Deferred (TODO P1.2)
--------------------
root
    dgt_logger and DGT_LOG_FORMATTER will be moved here from fms_dgt/utils.py
    once the contextvars-based RunContextFilter and run_context() context
    manager are implemented. Moving them now would touch every import in the
    codebase for no functional gain. Track in otel-observability.md TODO list.
"""

# Local
from fms_dgt.log.filters import RunContextFilter
from fms_dgt.log.handlers import FanOutHandler, LogDatastoreHandler

__all__ = [
    "FanOutHandler",
    "LogDatastoreHandler",
    "RunContextFilter",
]

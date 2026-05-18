# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""fms_dgt.log — central home for all DGT logging infrastructure.

Named 'log' rather than 'logging' to avoid shadowing the Python stdlib
logging module, which would cause import errors inside this package.

Modules
-------
context
    RunContextFilter    — contextvars-based filter; injects build_id / run_id
    run_context()       — context manager that sets the active run context
handlers
    FanOutHandler       — fan-out to dynamic set of task log handlers
    LogDatastoreHandler — writes structured JSONL records to a Datastore
filters
    RunContextFilter    — re-export from context (backwards-compat alias)
"""

# Local
from fms_dgt.log.context import RunContextFilter, run_context
from fms_dgt.log.handlers import FanOutHandler, LogDatastoreHandler

__all__ = [
    "FanOutHandler",
    "LogDatastoreHandler",
    "RunContextFilter",
    "run_context",
]

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import contextmanager
from typing import Generator, Optional
import contextvars
import logging
import uuid

# ---------------------------------------------------------------------------
# Run context ContextVar
# ---------------------------------------------------------------------------
_run_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "fms_dgt_run_context", default=None
)


# ---------------------------------------------------------------------------
# contextvars-based RunContextFilter
# ---------------------------------------------------------------------------
#
# Log record attribution — three-channel design
# =============================================
#
# Every fms_dgt log record carries three identity fields: build_id, run_id,
# and task_name.  Each is set by a different mechanism depending on the call
# site.  Understanding all three channels prevents confusion when reading or
# extending the logging infrastructure.
#
# Channel 1 — ContextVar (build_id, run_id)
#   RunContextFilter reads _run_ctx and stamps build_id / run_id onto every
#   record unconditionally.  generate_data() sets this context once per
#   databuilder run; Task._post_init() sets a nested scope during enrichment
#   so enrichment-phase logs carry the correct task-level build_id / run_id.
#   The filter is the single authoritative source for these two fields —
#   it always overwrites, no exceptions, so formatters can rely on them
#   always being present.
#
# Channel 2 — LoggerAdapter (task_name, enrichment scope only)
#   Task._post_init() wraps each enrichment's module logger in a
#   LoggerAdapter(logger, {"task_name": self._name}) before calling
#   enrichment.enrich().  This stamps task_name onto enrichment-phase records
#   without touching the ToolEnrichment / ToolRegistry API — those stay
#   task-agnostic.  Outside enrichment scope task_name is not set by any
#   adapter; the filter backfills "" so the field always exists.
#
# Channel 3 — explicit extra= at call site (task_name, execution scope)
#   Databuilder lifecycle events (task_started, task_finished, epoch_started,
#   etc.) pass task_name explicitly via extra={"task_name": task.name}.
#   Block __call__ structured logs derive task_name from get_row_name(datapoint).
#   These call-site values are set before the filter runs and are never
#   overwritten by the filter (only build_id / run_id are overwritten).
#
# Summary table:
#   Record source              build_id / run_id       task_name
#   -------------------------  ----------------------  --------------------------
#   Enrichment (_post_init)    ContextVar (Channel 1)  LoggerAdapter (Channel 2)
#   Lifecycle events           ContextVar (Channel 1)  explicit extra= (Channel 3)
#   Block __call__ logs        ContextVar (Channel 1)  get_row_name() (Channel 3)
#   All other records          ContextVar (Channel 1)  "" (filter fallback)
#
class RunContextFilter(logging.Filter):
    """Stamps ``build_id``, ``run_id``, and ``task_name`` onto every log record.

    ``build_id`` and ``run_id`` are read from the active ``_run_ctx``
    ContextVar and written unconditionally — the filter is the single
    authoritative source for these two fields.  When no context is active
    they are set to ``""``.

    ``task_name`` is not managed by the ContextVar.  It is set by a
    ``logging.LoggerAdapter`` during enrichment scope, or by explicit
    ``extra=`` at lifecycle event call sites.  The filter only ensures the
    field exists on every record (backfills ``""`` if absent) so formatters
    never encounter a missing attribute.

    See the module-level comment above for the full three-channel attribution
    design.

    Thread safety: each thread inherits a copy of the context at spawn time,
    so mutations in one thread are invisible to others.  Works correctly for
    the MT thread pool and ``concurrent.futures`` workers.

    Attach once to ``dgt_logger`` (the root fms_dgt logger) so *all* child
    loggers — including module-level loggers in ``openai.py``, ``anthropic.py``,
    ``utils.py``, and ``prompt.py`` — automatically carry run provenance
    without per-call-site changes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _run_ctx.get()
        if ctx:
            record.build_id = ctx["build_id"]
            record.run_id = ctx["run_id"]
        else:
            record.build_id = ""
            record.run_id = ""
        # task_name is set by LoggerAdapter (enrichment scope) or explicit
        # extra= (lifecycle events / block logs).  Only backfill if absent.
        if not hasattr(record, "task_name"):
            record.task_name = ""
        return True


# ---------------------------------------------------------------------------
# run_context() public context manager
# ---------------------------------------------------------------------------
@contextmanager
def run_context(
    build_id: str,
    run_id: Optional[str] = None,
) -> Generator[None, None, None]:
    """Set the active run context for the duration of the ``with`` block.

    All log records emitted on any ``fms_dgt`` logger while this context is
    active will carry ``build_id`` and ``run_id`` automatically, including
    records from module-level loggers (LLM connectors, prompt utilities).

    ``generate_data()`` calls this internally, so CLI and programmatic
    ``generate_data()`` usage gets full provenance with zero user effort.
    Library users who drive DGT components directly opt in explicitly::

        with fms_dgt.run_context(build_id="my_experiment"):
            outputs = my_block(inputs)

    Args:
        build_id: Identifies the experiment or pipeline run.  Should be stable
            across resumed runs of the same experiment.
        run_id: Unique identifier for this specific process invocation.
            Defaults to a fresh UUID4 if not provided.

    Yields:
        Nothing — the context is set for the duration of the block.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    token = _run_ctx.set({"build_id": build_id, "run_id": run_id})
    try:
        yield
    finally:
        _run_ctx.reset(token)

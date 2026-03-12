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
class RunContextFilter(logging.Filter):
    """Injects ``build_id`` and ``run_id`` onto every log record.

    Reads the current run context from a ``contextvars.ContextVar`` set by
    ``run_context()``.  When no context is active (e.g. module-level code
    running outside a run), the attributes are set to empty strings so
    downstream formatters and handlers never see a missing field.

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

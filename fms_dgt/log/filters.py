# Standard
import logging


class RunContextFilter(logging.Filter):
    """A logging filter that injects build_id and run_id onto every record.

    Attach to a task-scoped logger so all records emitted through it carry
    run provenance automatically — no per-call-site repetition required.

    Usage::

        filter = RunContextFilter(build_id="exp_001", run_id="abc-123")
        logger.addFilter(filter)
        logger.info("msg")  # record.build_id and record.run_id are set

    TODO (P1.2): Replace this fixed-value implementation with a
    contextvars-based version that reads from a ContextVar set by the runner.
    That version will be attached to the root dgt_logger (not per-task) so
    that module-level loggers (openai.py, anthropic.py, utils.py) also carry
    run provenance in the OTel collector. The fixed-value version here is
    sufficient for Tier 1 because all four structured lifecycle events go
    through task-scoped or builder-scoped loggers. See otel-observability.md
    "Run context propagation" section for full design.
    """

    def __init__(self, build_id: str, run_id: str) -> None:
        super().__init__()
        self._build_id = build_id
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.build_id = self._build_id
        record.run_id = self._run_id
        return True

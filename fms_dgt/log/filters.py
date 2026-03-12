# Re-export the contextvars-based RunContextFilter from context.py.
#
# The fixed-value implementation (build_id/run_id set at construction time)
# has been replaced by a ContextVar-based version that reads from the active
# run context at emit time.  All existing import sites use the same name and
# see no change.
# Local
from fms_dgt.log.context import RunContextFilter

__all__ = ["RunContextFilter"]

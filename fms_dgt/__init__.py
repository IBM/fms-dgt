# Standard
import os

# Local
from fms_dgt.log.context import run_context  # noqa: E402

INTERNAL_DGT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
SRC_DGT_DIR = os.path.join(INTERNAL_DGT_DIR, "fms_dgt")

# Public API: run_context() lets library users opt into run provenance when
# driving DGT components directly (outside of generate_data()).
__all__ = ["run_context"]

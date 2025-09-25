# Standard
from dataclasses import dataclass
from typing import Any, Dict, Tuple

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block


@dataclass(kw_only=True)
class NOOPBlockData(ValidatorBlockData):
    input: Any


@register_block("noop")
class NOOPValidator(ValidatorBlock):
    """Class for placeholder validator that always returns true"""

    DATA_TYPE: NOOPBlockData = NOOPBlockData

    def _validate(self, instance: NOOPBlockData) -> bool | Tuple[bool, Dict | None]:
        return True

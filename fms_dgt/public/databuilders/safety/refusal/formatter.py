# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict

# Local
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter
from fms_dgt.public.databuilders.safety.refusal.data_objects import SafetyRefusalData

# Metadata keys that are internal to the generation pipeline and should not
# appear in the final training dataset.
_INTERNAL_METADATA_KEYS = frozenset({"mode", "guardian_reasoning"})


@register_formatter("formatters/safety/refusal")
class SafetyRefusalFormatter(Formatter):
    """Formats a ``SafetyRefusalData`` point into a clean SFT training example.

    Output shape::

        {
            "instruction": "...",
            "response": "...",
            "metadata": {
                "risk": "...",
                "risk_group": "...",
                "risk_probability": 0.97,
                "label": "no",
                "policy_version": "v0.1",
                ...
            }
        }

    Internal pipeline fields (``mode``, ``guardian_reasoning``) are stripped.
    The ``task_name`` and ``is_seed`` bookkeeping fields are also excluded.
    """

    def apply(self, data: SafetyRefusalData, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        metadata = {
            k: v for k, v in (data.metadata or {}).items() if k not in _INTERNAL_METADATA_KEYS
        }
        return {
            "instruction": data.instruction,
            "response": data.response,
            "metadata": metadata,
        }

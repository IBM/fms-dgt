# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Local
from fms_dgt.base.data_objects import DataPoint


@dataclass(kw_only=True)
class SafetyRefusalData(DataPoint):
    """A single seed or machine-generated safety alignment example.

    Attributes:
        task_description: Free-text description of the generation task (used in
            instruction-generation prompts as ICL context).
        instruction: The harmful or adversarial prompt being modeled.
        response: The target model response (a calibrated refusal). ``None``
            for intermediate data points that have instructions but no response
            yet.
        source: ``"seed"`` for human-curated examples, ``"machine"`` for
            LM-generated ones.
        metadata: Rich metadata attached after Guardian validation, including
            risk category, Guardian confidence score, policy version, and
            filtering mode.
    """

    task_description: str
    instruction: str
    response: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

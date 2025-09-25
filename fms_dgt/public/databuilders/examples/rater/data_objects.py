"""
In DiGiT, operating over dataclasses in databuilders is beneficial since it facilitate stricter type checks and data consistenty.
We strongly recommend databuilder developers to define a `*Data` class extending from `DataPoint` which captures necessary fields in generated synthethic data and/or data to be transformed.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Dict

# Local
from fms_dgt.base.data_objects import DataPoint


@dataclass(kw_only=True)
class InputData(DataPoint):

    question: str
    answer: str


@dataclass(kw_only=True)
class OutputData(DataPoint):

    question: str
    answer: str
    ratings: Dict[str, Any]

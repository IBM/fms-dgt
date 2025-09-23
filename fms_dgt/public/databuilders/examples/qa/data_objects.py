"""
In DiGiT, operating over dataclasses in databuilders is beneficial since it facilitate stricter type checks and data consistenty.
We strongly recommend databuilder developers to define a `*Data` class extending from `DataPoint` which captures necessary fields in generated synthethic data and/or seed data.
"""

# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.data_objects import DataPoint


@dataclass(kw_only=True)
class GeographyQAData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    question: str
    answer: str

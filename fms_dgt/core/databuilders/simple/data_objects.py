# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.task import DataPoint


@dataclass(kw_only=True)
class SimpleData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    taxonomy_path: str
    task_description: str
    instruction: str
    input: str
    output: str
    document: str

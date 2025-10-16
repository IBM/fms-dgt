# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.task import DataPoint


@dataclass(kw_only=True)
class TimeSeriesInputData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_description: str = ""
    observations: dict = ""


@dataclass(kw_only=True)
class TimeSeriesOutputData(DataPoint):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_description: str = ""
    generated_time_series: dict = ""

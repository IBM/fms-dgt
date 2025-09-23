# Standard
from dataclasses import dataclass

# Local
from fms_dgt.base.task import DataPoint


@dataclass(kw_only=True)
class SkillsData(DataPoint):

    task_description: str
    instruction: str
    response: str
    context: str | None = None
    tags: str | dict = None

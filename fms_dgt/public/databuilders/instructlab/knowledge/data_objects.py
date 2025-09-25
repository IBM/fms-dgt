# Standard
from dataclasses import dataclass
from typing import Any, Dict

# Local
from fms_dgt.base.task import DataPoint


@dataclass(kw_only=True)
class KnowledgeData(DataPoint):
    task_description: str
    domain: str
    question: str
    answer: str
    context: Dict[str, Any] | None = None
    tags: dict | None = None

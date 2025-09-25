# Standard
from abc import ABC, abstractmethod

# Local
from fms_dgt.base.data_objects import DataPoint


class Formatter(ABC):
    """Base Class for all formatters"""

    @abstractmethod
    def apply(self, data: DataPoint, *args, **kwargs):
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

# Standard
from abc import ABC, abstractmethod
from typing import Any


# ===========================================================================
#                       BASE
# ===========================================================================
class Dataloader(ABC):
    """Base Class for all dataloaders"""

    @abstractmethod
    def get_state(self) -> Any:
        """Gets the state of the dataloader which influences the __next__ function"""
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def set_state(self, state: Any) -> None:
        """Sets the state of the dataloader which influences the __next__ function

        Args:
            state (Any): object representing state of dataloader
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def __next__(self) -> Any:
        """Gets next element from dataloader

        Returns:
            Any: Element of dataloader
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    def __iter__(self):
        return self

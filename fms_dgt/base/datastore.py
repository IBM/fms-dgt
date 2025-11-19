# Standard
from abc import abstractmethod
from typing import Any, Iterator, List, Optional

# Local
from fms_dgt.constants import DATASET_TYPE


# ===========================================================================
#                       BASE
# ===========================================================================
class Datastore:
    """Base Class for all data stores"""

    def __init__(
        self,
        store_name: str,
        restart: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        self._store_name = store_name
        self._restart = restart

        # Additional kwargs
        self._addtl_kwargs = kwargs | {}

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def store_name(self):
        return self._store_name

    # ===========================================================================
    #                       FUNCTIONS
    # ===========================================================================
    @abstractmethod
    def save_data(self, data_to_save: DATASET_TYPE) -> None:
        """
        Saves generated data to specified location

        Args:
            data_to_save (DATASET_TYPE): A list of data items to be saved
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def load_iterators(
        self,
    ) -> List[Iterator]:
        """
        Returns a list of iterators over the data elements

        Returns:
            A list of iterators
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def load_data(
        self,
    ) -> DATASET_TYPE:
        """Loads generated data from save location.

        Returns:
            A list of generated data of type DATASET_TYPE.
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def close(self) -> None:
        """Method for closing a datastore when generation has completed"""
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

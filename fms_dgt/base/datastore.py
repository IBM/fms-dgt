# Standard
from abc import abstractmethod
from typing import Any, Iterator, List, Optional
import logging

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
        fanout_handler: logging.Handler | None = None,
        **kwargs: Any,
    ) -> None:
        self._store_name = store_name
        self._restart = restart

        # Initialize datastore-scoped logger. Attach the shared FanOutHandler so
        # records are routed to all currently-active task log files. Falls back
        # to stdout-only via propagation to dgt_logger if no handler is provided.
        self._logger = logging.getLogger(f"fms_dgt.datastore.{store_name}")
        if fanout_handler is not None:
            self._logger.addHandler(fanout_handler)

        # Additional kwargs
        self._addtl_kwargs = kwargs | {}

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def store_name(self):
        return self._store_name

    @property
    def logger(self) -> logging.Logger:
        """Returns the datastore-scoped logger.

        Records propagate to the root dgt_logger for terminal output and, when
        a FanOutHandler is attached, are also routed to all active task log files.

        Returns:
            logging.Logger: Datastore-scoped logger
        """
        return self._logger

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

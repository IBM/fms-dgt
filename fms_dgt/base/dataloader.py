# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any
import logging


# ===========================================================================
#                       BASE
# ===========================================================================
class Dataloader(ABC):
    """Base Class for all dataloaders"""

    def __init__(
        self,
        name: str | None = None,
        fanout_handler: logging.Handler | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize dataloader-scoped logger. Attach the shared FanOutHandler so
        # records are routed to all currently-active task log files. Falls back
        # to stdout-only via propagation to dgt_logger if no handler is provided.
        logger_name = f"fms_dgt.dataloader.{name}" if name else "fms_dgt.dataloader"
        self._logger = logging.getLogger(logger_name)
        if fanout_handler is not None:
            self._logger.addHandler(fanout_handler)

    @property
    def logger(self) -> logging.Logger:
        """Returns the dataloader-scoped logger.

        Records propagate to the root dgt_logger for terminal output and, when
        a FanOutHandler is attached, are also routed to all active task log files.

        Returns:
            logging.Logger: Dataloader-scoped logger
        """
        return self._logger

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

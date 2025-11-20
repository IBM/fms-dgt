# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
PROJECTION_FIELD_TEXT = "text"
PROJECTION_FIELD_ID = "id"


# ===========================================================================
#                       DATA OBJECTS
# ===========================================================================
@dataclass(kw_only=True)
class UnstructuredTextDocument:
    """
    Document
    """

    id: str
    text: str
    score: Optional[float] = None
    metadata: Optional[dict] = None


class UnstructuredTextRetriever(ABC):
    """Base class for unstructured text retrievers"""

    def __init__(
        self,
        projection: Dict[str, str] = {"text": "text", "id": "id"},
        limit: int = 10,
        _id: Optional[str] = str(uuid4()),
        **kwargs: Any,
    ) -> None:

        # Assert all necessary information is available
        if projection is None:
            raise ValueError("Must specify 'projection' field")

        if PROJECTION_FIELD_TEXT not in projection.values():
            raise ValueError(
                f"Must specify {PROJECTION_FIELD_TEXT} as of the values for 'projection' field"
            )

        if PROJECTION_FIELD_ID not in projection.values():
            raise ValueError(
                f"Must specify {PROJECTION_FIELD_ID} as of the values for 'projection' field"
            )

        if not isinstance(limit, int) and limit <= 0:
            raise ValueError("Must specify 'limit' field as an integer and greater than 0.")

        # Step 1: Initialize variables
        self._mappings = {v: k for k, v in projection.items()}
        self._limit = limit
        self._id = _id

    def form_query(self, query_text: str) -> str:
        """
        Method to specify custom query formation logic. By default, query text is returned as it is.

        Args:
            query_text (str): text to use in query formation

        Returns:
            str: formed query
        """
        return query_text

    @abstractmethod
    def __call__(
        self,
        requests: List[Union[str, dict, None]],
        *args,
        **kwargs,
    ) -> List[List[UnstructuredTextDocument]]:
        """
        Top-level process method to retrieving unstructured text records

        Args:
            query: query to be run

        Returns:
            List[dict]: unstructured text records
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @property
    def limit(self):
        return self._limit

    @property
    def id(self):
        return self._id

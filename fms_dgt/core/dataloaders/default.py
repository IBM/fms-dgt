# Standard
from itertools import islice
from typing import Any, Dict, Iterator, List

# Local
from fms_dgt.base.dataloader import Dataloader
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.registry import register_dataloader
from fms_dgt.utils import dgt_logger, from_dict, to_dict

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
_ITER_INDEX = "iter_index"
_ROW_INDEX = "row_index"


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_dataloader("default")
class DatastoreDataloader(Dataloader):
    """
    A dataloader that lazily yields data items from one or more iterators,
    typically sourced from a datastore.

    This dataloader supports loading data either directly from a `Datastore`
    (via its `load_iterators()` method) or from a provided list of iterators.
    It yields items one by one and supports resuming iteration from a previously
    saved state using internal indices. Items can optionally be transformed based
    on a provided set of fields.

    Note:
        - This loader is **non-repeatable** if using `iterators` — once the iterators are exhausted,
          they cannot be reused.
        - However, it can be reset if using a `datastore`.
        - Designed for use cases where datasets are large or streaming in nature.

    Args:
        datastore (Datastore, optional): A datastore object that provides data iterators.
        iterators (List[Iterator], optional): A list of iterators supplying data items.
        state_datastore (Datastore, optional): Optional secondary datastore for state-tracking.
        fields (Dict, optional): Optional dictionary of field transformations to apply to each item.

    Raises:
        ValueError: If an item returned from an iterator is not a dictionary and field transformation is required.
    """

    def __init__(
        self,
        datastore: Datastore = None,
        iterators: List[Iterator] = None,
        state_datastore: Datastore = None,
        fields: Dict | None = None,
        loop_over: bool = False,
        **kwargs: Any,
    ) -> None:

        if datastore and iterators:
            raise ValueError(
                "Must specify one of 'datastore' or 'iterators' but not both"
            )

        if datastore:
            self._datastore = datastore
            self._iterators = datastore.load_iterators()
        elif iterators is not None:
            self._datastore = None
            self._iterators = iterators
        else:
            raise ValueError("Must provide either `datastore` or `iterators`.")

        self._state_datastore = state_datastore
        self._fields = fields
        self._loop_over = loop_over

        self._iterator_index = 0
        self._row_index = 0
        self._has_skipped = True

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def state_datastore(self) -> Datastore:
        """Returns the datastore used to track saved state, if any."""
        return self._state_datastore

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def get_state(self) -> Dict[str, int]:
        """Returns the current iterator and row position."""
        return {_ITER_INDEX: self._iterator_index, _ROW_INDEX: self._row_index}

    def set_state(self, state: Dict[str, int]) -> None:
        """
        Resumes dataloader from the specified state.

        Args:
            state: Dictionary containing 'iter_index' and 'row_index'.

        Sets `_has_skipped` to False so skip logic is applied on next call.
        """
        self._iterator_index = state.get(_ITER_INDEX, 0)
        self._row_index = state.get(_ROW_INDEX, 0)
        self._has_skipped = False  # ensures we skip on the next __next__ call

    def reset_state(self) -> None:
        """
        Resets the iterator and row index to start from the beginning.

        Only supported if initialized via a `datastore`. Raises an error otherwise.
        """
        if self._datastore:
            self._iterators = self._datastore.load_iterators()
        elif self._iterators:
            # Attempt to reinitialize only if original iterators were callable (like generator functions)
            dgt_logger.warning(
                "reset_state() is not supported when initialized with iterators.\n"
                "Please reinitialize with a datastore instead."
            )
            raise StopIteration
        self._iterator_index = 0
        self._row_index = 0
        self._has_skipped = True

    def _transform(self, item: Any) -> Dict[str, Any]:
        """
        Transforms the given item based on field mapping.

        Handles:
            1. Retain-all: {"*": "*"}
            2. Retain-all with rename
            3. Selective extraction
            4. Full rename map

        Args:
            item: A dictionary item.

        Returns:
            Transformed dictionary.

        Raises:
            ValueError: If input is not a dictionary.
            KeyError: If specified field not found in input.
        """
        if not isinstance(item, dict):
            raise ValueError("Expected a dictionary item for transformation.")

        # No fields specified or wildcard only -> retain everything
        if not self._fields or self._fields == {"*": "*"}:
            return item

        # Wildcard present with some renames
        updated_item = {}
        if self._fields.get("*") == "*":
            updated_item.update(item)

        for src_field, dest_field in self._fields.items():
            if src_field == "*":
                continue
            try:
                to_dict(updated_item, key=dest_field, value=from_dict(item, src_field))
            except (AttributeError, ValueError, TypeError, KeyError) as e:
                dgt_logger.error("Error transforming field '%s': %s", src_field, e)
                raise KeyError(
                    f"Missing or invalid key in input item: {src_field}"
                ) from e

        return updated_item

    def __next__(self) -> Any:
        while self._iterator_index < len(self._iterators):
            iterator = self._iterators[self._iterator_index]

            # Apply skip when resuming from a saved state
            if not self._has_skipped and self._row_index > 0:
                iterator = islice(iterator, self._row_index, None)
                self._iterators[self._iterator_index] = iterator
                self._has_skipped = True

            try:
                item = next(iterator)
                self._row_index += 1
                return self._transform(item) if self._fields else item
            except StopIteration:
                # move on to the next iterator
                self._iterator_index += 1
                self._row_index = 0

        if self._loop_over:
            dgt_logger.info("No more rows left in dataloader. Resetting index to 0.")
            self.reset_state()

        raise StopIteration

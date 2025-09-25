# Standard
from typing import Any

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.dataloader import Dataloader
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.registry import register_dataloader


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_dataloader("simple")
class SimpleDataloader(Dataloader):
    """Base Class for all dataloaders"""

    def __init__(
        self,
        data: DATASET_TYPE,
        state_datastore: Datastore = None,
        loop_over: bool = True,
        **kwargs: Any,
    ) -> None:
        self._data = data
        self._state_datastore = state_datastore
        self._i = 0
        self._loop_over = loop_over

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
    def get_state(self) -> Any:
        return self._i

    def set_state(self, state: Any) -> None:
        self._i = state

    def __next__(self) -> Any:
        try:
            value = self._data[self._i]
            self._i += 1
            return value
        except IndexError as err:
            # reset cycle
            if self._loop_over:
                self._i = 0
            raise StopIteration from err

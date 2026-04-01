# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List

# Local
from fms_dgt.core.databuilders.conversation.data_objects import ConversationDataPoint


class Stage:
    """Base class for all conversation pipeline stages.

    A stage is a stateless batch processor: it takes a list of
    ConversationDataPoint objects, performs its work (LM call, tool execution,
    validation, etc.), and returns a (possibly shorter) list. Dropping a
    data point is done by omitting it from the returned list — no exceptions,
    no side channels.

    All state mutation goes through the ConversationDataPoint object itself
    (appending Steps, setting flow_signal, writing scores onto steps, etc.).
    Stages must not mutate shared mutable state outside the data points they
    receive.

    Subclasses register themselves with @register_stage("my/stage/name") and
    receive their dependencies (LM blocks, config kwargs) through __init__.
    """

    def __init__(self, *, name: str, **kwargs) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        """Process a batch of conversation contexts.

        Args:
            data_points: Contexts to process. The stage may mutate each
                context in place (e.g., appending a Step) and return it,
                or drop it by omitting it from the returned list.
            seed_data: Optional ICL seed conversations. Stages that use
                demonstrations sample from this list. Stages that do not
                need ICL ignore it.

        Returns:
            A (possibly shorter) list of processed ConversationDataPoint objects.
            Drop a data point silently by not including it in the return value.
        """
        raise NotImplementedError

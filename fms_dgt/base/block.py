# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Tuple
import asyncio
import dataclasses
import inspect
import logging
import os
import time
import tracemalloc

# Third Party
from datasets import Dataset
import pandas as pd

# Local
from fms_dgt.base.data_objects import BlockData, ValidatorBlockData
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.registry import get_datastore
from fms_dgt.base.telemetry import Span, _NoOpSpanWriter
from fms_dgt.constants import (
    DATASET_ROW_TYPE,
    DATASET_TYPE,
    DGT_DIR,
    STORE_NAME_KEY,
    TYPE_KEY,
)

# ===========================================================================
#                       CONSTATNTS
# ===========================================================================
_SRC_DATA = "SRC_DATA"


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
_warned_missing_task_name: set = set()


def get_row_name(gen_inst: DATASET_ROW_TYPE) -> str | None:
    """Gets the task name associated with the particular input instance.

    Lookup order:
    1. ``task_name`` directly on the object (covers ``DataPoint`` and dicts).
    2. ``SRC_DATA`` recursion (covers ``BlockData`` / ``ValidatorBlockData``
       whose original row is a ``DataPoint`` carrying ``task_name``).
    3. Returns ``None`` and emits a one-time warning per type so developers
       know that structured logs and lifecycle events will be degraded.

    Args:
        gen_inst (DATASET_ROW_TYPE): The input to get the task name from.

    Returns:
        str | None: Name of task, or None if not found.
    """
    if isinstance(gen_inst, dict):
        if "task_name" in gen_inst:
            return gen_inst["task_name"]
        src = gen_inst.get("SRC_DATA")
        if src is not None:
            return get_row_name(src)
    else:
        task_name = getattr(gen_inst, "task_name", None)
        if task_name is not None:
            return task_name
        src = getattr(gen_inst, "SRC_DATA", None)
        if src is not None:
            return get_row_name(src)

    type_name = type(gen_inst).__qualname__
    if type_name not in _warned_missing_task_name:
        _warned_missing_task_name.add(type_name)
        logging.getLogger(__name__).warning(
            "get_row_name: could not find 'task_name' on '%s' or its SRC_DATA; "
            "task_name will be missing from structured logs and lifecycle events.",
            type_name,
        )
    return None


# ===========================================================================
#                       MAIN CLASSES
# ===========================================================================
class Block:
    """Base Class for all Blocks."""

    DATA_TYPE: BlockData = None
    profiler_data: Dict[str, Any] = None

    def __init__(
        self,
        name: str | None = None,
        type: str | None = None,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        builder_name: str | None = None,
        datastores: List[Dict] | Dict = None,
        fanout_handler: logging.Handler | None = None,
        **kwargs: Any,
    ) -> None:
        """A block is a unit of computation that takes in some inputs and produces an
        output. It is intended to be specialized algorithms or processes that teams can
        contribute for others to use to build their pipelines.

        Args:
            name (str, optional): The name of the block.
            type (str, optional): The type of the block.

        Kwargs:
            input_map (List | Dict, optional): A mapping of field names from input objects to internal objects.
            output_map (List | Dict, optional): A mapping of field names from internal objects to output objects.
            builder_name (str, optional): Name of the calling databuilder
            datastores (List[Dict] | Dict, optional): Dictionaries containing the configuration for the datastores.
            fanout_handler (logging.Handler, optional): Shared FanOutHandler from the DataBuilder. When provided,
                block log records are routed to all currently-active task log files.

        Raises:
            TypeError: If any of the arguments are not of the correct type.
        """
        if not isinstance(input_map, (dict, list, None.__class__)):
            raise TypeError("[input_map] must be of type 'dict' or 'list'")
        if not isinstance(output_map, (dict, list, None.__class__)):
            raise TypeError("[output_map] must be of type 'dict' or 'list'")

        self._name = name
        self._block_type = type

        # input / output maps
        self._input_map = input_map
        self._output_map = output_map
        self._req_args, self._opt_args = [], []
        if not (self.DATA_TYPE is None or issubclass(self.DATA_TYPE, dict)):
            self._req_args = [
                f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.default == dataclasses.MISSING and f.name != _SRC_DATA
            ]
            self._opt_args = [
                f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.default != dataclasses.MISSING
            ]

        # datastore params
        self._datastores = None
        if datastores is not None:
            self._datastores = {
                datastore_cfg[STORE_NAME_KEY]: get_datastore(
                    datastore_cfg.get(TYPE_KEY),
                    **datastore_cfg,
                )
                for datastore_cfg in datastores
            }

        self._blocks: List[Block] = []

        # Initialize profiler data
        self.profiler_data = {"executions": []}

        # Initialize block-scoped logger. Attach the shared FanOutHandler so
        # records are routed to all currently-active task log files alongside
        # the builder's own records. Falls back to stdout-only via propagation
        # to dgt_logger if no FanOutHandler is provided (e.g. standalone tests).
        self._logger = logging.getLogger(f"fms_dgt.block.{self._name}")
        if fanout_handler is not None:
            self._logger.addHandler(fanout_handler)

        # SpanWriter set by DataBuilder.span_writer setter when telemetry is
        # configured. Defaults to no-op so __call__ is safe before wiring.
        self._span_writer = _NoOpSpanWriter()
        self._builder_name = builder_name or ""

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def name(self) -> str:
        """Returns the name of the block.

        Returns:
            str: The name of the block.
        """
        return self._name

    @property
    def block_type(self) -> str:
        """Returns a string representing type of the block.

        Returns:
            str: The type of the block
        """
        return self._block_type

    @property
    def span_writer(self):
        """Returns the SpanWriter for telemetry (set by DataBuilder)."""
        return self._span_writer

    @span_writer.setter
    def span_writer(self, writer) -> None:
        self._span_writer = writer

    @property
    def input_map(self) -> List | Dict:
        """Returns a dictionary or list that will be used to map field names from input
        objects to internal objects.

        Returns:
            List[str] | Dict: A dictionary or list of fields to extract
        """
        return self._input_map

    @property
    def output_map(self) -> List | Dict:
        """Returns a dictionary or list that will be used to map field names from
        internal objects to output objects.

        Returns:
            List[str] | Dict: A dictionary or list of fields to extract
        """
        return self._output_map

    @property
    def datastores(self) -> List[Datastore]:
        """Returns the datastore of the block.

        Returns:
            List[Datastore]: Datastore of the block
        """
        return self._datastores.values()

    @property
    def blocks(self) -> List["Block"]:
        return self._blocks

    @property
    def logger(self) -> logging.Logger:
        """Returns the block-scoped logger.

        Records are routed to all currently-active task log files via the
        shared FanOutHandler, and propagate to the root dgt_logger for
        terminal output.

        Returns:
            logging.Logger: Block-scoped logger
        """
        return self._logger

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def close(self):
        """Method for safely deallocating all resources used by a block."""
        for block in self._blocks:
            block.close()

    def save_data(self, data: DATASET_TYPE) -> None:
        def to_serializable(x):
            def _to_serializable_inner(x):
                if isinstance(x, pd.Series):
                    return _to_serializable_inner(x.to_dict())
                elif isinstance(x, BlockData):
                    return x.to_dict()
                elif dataclasses.is_dataclass(x):
                    return _to_serializable_inner(dataclasses.asdict(x))
                elif isinstance(x, dict):
                    return {k: _to_serializable_inner(v) for k, v in x.items()}
                elif isinstance(x, (tuple, list)):
                    return [_to_serializable_inner(y) for y in x]
                return x

            x = _to_serializable_inner(x)
            if not isinstance(x, dict):
                raise ValueError(
                    f"Attempting to serialize {x} to datastore, but data cannot be converted into dictionary"
                )
            return x

        if data and self._datastores is not None:
            # Collate data points per store
            data_per_store = {}
            for datapoint in data:
                if isinstance(datapoint, BlockData) and datapoint.store_names:
                    for store_name in datapoint.store_names:
                        try:
                            data_per_store[store_name].append(datapoint)
                        except KeyError:
                            data_per_store[store_name] = [datapoint]

            # Bulk write datapoints per store
            for store_name, datapoints in data_per_store.items():
                try:
                    self._datastores[store_name].save_data([to_serializable(x) for x in datapoints])
                except KeyError:
                    self.logger.warning(
                        'Unable to save instances due to missing datastore with "%s" name.',
                        store_name,
                    )

    def transform_input(
        self,
        inp: DATASET_ROW_TYPE | DATA_TYPE,  # type: ignore
        input_map: Dict,
    ) -> DATA_TYPE:  # type: ignore
        """Extracts the elements of the input as specified by map.

        Args:
            inp (Union[DATASET_ROW_TYPE, DATA_TYPE]): The input data to be mapped
            input_map (Union[List, Dict]): A mapping of field names from input objects to internal objects.

        Returns:
            Dict: A dictionary containing the result of the mapping.
        """

        inp_obj = asdict(inp) if is_dataclass(inp) else dict(inp)

        # if none is provided, assume it maps directly
        if input_map is None:
            input_map = dict()

        if isinstance(inp_obj, (dict, pd.DataFrame, Dataset)):
            # NOTE: we flip this here because from a DGT pipeline, the input map goes from UserData -> BlockData
            data_type_map = {
                **{v: k for k, v in self._get_default_map(inp).items()},
                **{v: k for k, v in input_map.items()},
            }

            args = (self._req_args + self._opt_args) or data_type_map.keys()

            mapped_data = {
                arg: inp_obj.get(data_type_map.get(arg))
                for arg in args
                if data_type_map.get(arg) in inp_obj
            }

            missing = [r_a for r_a in self._req_args if r_a not in mapped_data]
            if missing:
                raise ValueError(f"Required inputs {missing} are not provided in 'input_map'")

            return (
                {**mapped_data, _SRC_DATA: inp}
                if self.DATA_TYPE is None
                else self.DATA_TYPE(**mapped_data, SRC_DATA=inp)
            )

        raise TypeError(f"Unexpected input type: {type(inp)}")

    def transform_output(
        self,
        inp: BlockData,  # type: ignore
        output_map: Dict,
    ) -> Dict:
        """Extracts the elements of the internal data type as specified by output_map.

        Args:
            inp (Union[DATASET_ROW_TYPE, DATA_TYPE]): The input data to be mapped
            output_map (Union[List, Dict]): A mapping of field names from input objects to internal objects.

        Returns:
            Dict: A dictionary containing the result of the mapping.
        """
        src_data = inp[_SRC_DATA] if isinstance(inp, dict) else inp.SRC_DATA

        # if none is provided, assume it maps directly
        if output_map is None:
            output_map = dict()

        # start with assuming elements of input will be used
        output_map = {**self._get_default_map(src_data), **output_map}

        if is_dataclass(src_data):
            for k, v in output_map.items():
                # since a dataclass will throw an error, only try to add attributes if original data type has them
                if hasattr(src_data, v):
                    attr_val = inp[k] if isinstance(inp, dict) else getattr(inp, k)
                    setattr(src_data, v, attr_val)
        elif isinstance(src_data, (dict, pd.DataFrame, Dataset)):
            # TODO: handle things other than dictionaries
            for k, v in output_map.items():
                attr_val = inp[k] if isinstance(inp, dict) else getattr(inp, k)
                src_data[v] = attr_val
        else:
            raise TypeError(f"Unexpected input type: {type(inp)}")

        return src_data

    def _get_default_map(self, data: Dict | BlockData):
        # if DATA_TYPE is not provided, assume it maps to the input
        if is_dataclass(self.DATA_TYPE):
            fields = dataclasses.fields(self.DATA_TYPE)
        else:
            fields = data.keys() if isinstance(data, dict) else dataclasses.fields(data)
        fields = [f if isinstance(f, str) else f.name for f in fields]
        return {f: f for f in fields if f != _SRC_DATA}

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def __call__(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The __call__ function is the primary interface to a Block. Internally, it
        calls the `execute` method which contains the logic of the block. This function
        exists to have meta-processes (e.g., logging) that wrap around the core logic of
        a block.

        Args:
            inputs (DATASET_TYPE): Dataset to be processed by 'execute' method of block.
            input_map (Optional[Union[List, Dict]], optional): Mapping applied to each row of dataset that will convert row to instance of self.DATA_TYPE.
            output_map (Optional[Union[List, Dict]], optional): Mapping applied to each instance of self.DATA_TYPE that will convert instance back into row of dataset.

        Returns:
            DATASET_TYPE: Dataset resulting from processing contained in execute function.
        """
        input_map = input_map or self._input_map
        output_map = output_map or self._output_map

        transformed_inputs = map(lambda x: self.transform_input(x, input_map), inputs)
        if isinstance(inputs, (list, tuple)):
            transformed_inputs = type(inputs)(transformed_inputs)

        stack = []
        for elem in inspect.stack()[1:]:
            if elem.filename.startswith(DGT_DIR) and not elem.filename.endswith("block.py"):
                if elem.filename.endswith(os.path.join("base", "databuilder.py")):
                    break

                stack.append(
                    {
                        "filename": elem.filename[len(DGT_DIR) :].lstrip("/"),
                        "lineno": elem.lineno,
                        "function": elem.function,
                    }
                )

        input_count = len(inputs) if isinstance(inputs, (list, tuple)) else None

        start_time = time.monotonic()
        tracemalloc.start()
        with Span(
            "dgt.block",
            self._span_writer,
            parent_span_name="dgt.epoch",
            block_name=self._name,
            block_type=self._block_type,
            builder_name=self._builder_name,
            input_count=input_count,
        ) as _span:
            outputs = self.execute(transformed_inputs, *args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                _span.set_attribute("output_count", len(outputs))
        memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.profiler_data["executions"].append(
            {
                "time": round(time.monotonic() - start_time, 2),
                "stack": stack,
                "peak_memory": memory[-1],
            }
        )

        transformed_outputs = map(lambda x: self.transform_output(x, output_map), outputs)
        if isinstance(inputs, (list, tuple)):
            transformed_outputs = type(inputs)(transformed_outputs)

        return transformed_outputs

    async def acall(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """Async variant of ``__call__``.

        Applies the same input/output mapping as ``__call__`` but runs
        ``execute`` in a thread pool via ``asyncio.to_thread`` so the event
        loop is not blocked.  Subclasses with native async implementations
        (e.g. ``LMProvider``) may override ``aexecute`` to avoid the thread
        hop entirely.

        Args:
            inputs (DATASET_TYPE): Dataset to be processed.
            input_map (Optional[Union[List, Dict]]): Override for instance-level input map.
            output_map (Optional[Union[List, Dict]]): Override for instance-level output map.

        Returns:
            DATASET_TYPE: Dataset resulting from processing in ``execute``.
        """
        input_map = input_map or self._input_map
        output_map = output_map or self._output_map

        transformed_inputs = list(map(lambda x: self.transform_input(x, input_map), inputs))

        outputs = await self.aexecute(transformed_inputs, *args, **kwargs)

        transformed_outputs = map(lambda x: self.transform_output(x, output_map), outputs)
        if isinstance(inputs, (list, tuple)):
            transformed_outputs = type(inputs)(transformed_outputs)

        return transformed_outputs

    async def aexecute(
        self,
        inputs: DATASET_TYPE,
        *args,
        **kwargs,
    ) -> DATASET_TYPE:
        """Async variant of ``execute``.

        The default implementation offloads the synchronous ``execute`` to a
        thread pool via ``asyncio.to_thread``, which is safe for any subclass
        whose ``execute`` does not itself schedule work on the running event
        loop.  Subclasses with a native async implementation may override this
        method directly.

        Args:
            inputs (DATASET_TYPE): Pre-transformed inputs (output of ``transform_input``).

        Returns:
            DATASET_TYPE: Results as returned by ``execute``.

        TODO: Override ``aexecute`` in ``LMProvider`` to drive
        ``_execute_requests`` directly on the caller's event loop, eliminating
        the ``asyncio.to_thread`` hop and the internal ``run_async`` /
        ``run_coroutine_threadsafe`` round-trip.  Requires refactoring
        ``completion`` / ``chat_completion`` in each provider so the async
        path is callable without going through the sync wrapper.
        """
        return await asyncio.to_thread(self.execute, inputs, *args, **kwargs)

    @abstractmethod
    def execute(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: List | Dict | None = None,
        output_map: List | Dict | None = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The `execute` function is the primary logic of a Block.

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            input_map (Optional[Union[List, Dict]], optional): A mapping of field names from input objects to internal objects.
            output_map (Optional[Union[List, Dict]], optional): A mapping of field names from internal objects to output objects.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function

        Returns:
            DATASET_TYPE: Input dataset with results added
        """


class ValidatorBlock(Block):
    def __init__(
        self,
        filter: bool | None = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a block that validates (and possibly filters) its input.

        Parameters:
            filter (Optional[bool]): Whether to filter out invalid values from the list.
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._filter_invalids = filter

    def execute(
        self,
        inputs: Iterable[ValidatorBlockData],
        *args,
        filter: bool = True,
        **kwargs,
    ) -> DATASET_TYPE:
        """The execute function is the primary interface to a Block. For validator
        blocks, the implementation differs from Block in that the result is always a
        boolean value indicating whether the validation succeeded or failed. In
        addition, the validator block can optionally filter out invalid inputs that
        would return False instead of writing the result to the input.

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            input_map (Optional[Union[List, Dict]], optional): A mapping of field names from input objects to internal objects.
            output_map (Optional[Union[List, Dict]], optional): A mapping of field names from internal objects to output objects.
            filter (Optional[bool]): Whether to filter out invalid values from the list.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function

        Returns:
            DATASET_TYPE: Input dataset with results added (possibly filtered to remove any invalid inputs)
        """

        # Turn filtering ON/OFF, if requested
        filter = filter and self._filter_invalids

        # Validate instances
        retained_instances, filtered_instances = [], []
        for x in inputs:
            validation_output = self._validate(x)
            if isinstance(validation_output, bool):
                x.is_valid = validation_output
            elif isinstance(validation_output, tuple) and len(validation_output) == 2:
                x.is_valid, x.metadata = validation_output
            else:
                raise RuntimeError(
                    '"_validate" function must return a bool or tuple of [bool, Dict]',
                )

            if x.is_valid or not filter:
                retained_instances.append(x)
            elif not x.is_valid:
                filtered_instances.append(x)
                self.logger.info(
                    "Data point rejected by block '%s'",
                    self.name,
                    extra={
                        "event": "data_point_rejected",
                        "block_name": self.name,
                        "block_type": self.block_type,
                        "task_name": get_row_name(x),
                        "reason": getattr(x, "metadata", None),
                    },
                )

        # Save filtered instances for record keeping
        self.save_data(filtered_instances)

        # Return retained instances
        return retained_instances

    @abstractmethod
    def _validate(self, *args: Any, **kwargs: Any) -> bool | Tuple[bool, Dict | None]:
        """Derived validators must implement _validate with their core logic.

        Returns:
            Tuple[bool, Dict | None]: a boolean (True or False) to reflect whether an input was valid or not and optional second entry to reflect any additional metadata
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

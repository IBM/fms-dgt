# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union
import logging
import os
import random

# Local
from fms_dgt.base.data_objects import (
    DataPoint,
    GenerationTaskRunnerConfig,
    TaskRunnerConfig,
    TransformationTaskRunnerConfig,
)
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import get_dataloader, get_datastore, get_formatter
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.constants import TASK_NAME_KEY, TYPE_KEY
from fms_dgt.log import LogDatastoreHandler
from fms_dgt.utils import (
    group_data_by_attribute,
    init_dataclass_from_dict,
)

# Logger name prefix for all task-scoped loggers; child of dgt_logger so records
# propagate to its stdout handler automatically.
_TASK_LOGGER_PREFIX = "fms_dgt"

# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    """Utility function that groups input data by task name.

    Args:
        data_list (List[T]): List of DataPoint to group into tasks

    Returns:
        List[List[T]]: DataPoint that has been grouped into tasks
    """
    return group_data_by_attribute(data_list, TASK_NAME_KEY)


# ===========================================================================
#                       BASE
# ===========================================================================
class Task:
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = DataPoint
    OUTPUT_DATA_TYPE: DataPoint = None

    def __init__(
        self,
        task_name: str,
        task_description: str,
        created_by: str,
        data_builder: str,
        task_card: TaskRunCard,
        runner_config: Mapping | TaskRunnerConfig,
        formatter: Dict | None = None,
        datastore: Dict | None = None,
        final_datastore: Dict | None = None,
        formatted_datastore: Dict | None = None,
        store_name: str | None = None,
        **kwargs: Any,
    ):
        """Initializes task object.

        Args:
            task_name (str): The name of the Task object.
            task_description (str): A description of the SDG task is designed to solve.
            created_by (str): The name of the individual / group who created the code assistant.
            data_builder (str): The name of the data builder that should be used to process this task.
            task_card (TaskCard): The task card containing all experiment information.
            runner_config (Union[Mapping, TaskRunnerConfig]): Config specifying the run settings of the task.
            formatter (Optional[Dict]): A dictionary containing the configuration for the formatter.
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.
            final_datastore (Optional[Dict]): A dictionary containing the configuration for the datastore used for storing final data.
            formatted_datastore (Optional[Dict]): A dictionary containing the configuration for the datastore used for storing formatted data.
            store_name (Optional[str]): A base name to use for the datastores. Will be set to [task_name] if None

        """
        # Set output data type to input data type, if unspecified
        if self.OUTPUT_DATA_TYPE is None:
            self.OUTPUT_DATA_TYPE = self.INPUT_DATA_TYPE

        # Save task specific mandatory fields
        self._name = task_name
        self._task_description = task_description
        self._created_by = created_by

        self._data_builder = data_builder
        self._task_card = task_card

        # Save additional arguments
        self._kwargs = kwargs

        # Extract required variables from the runner configuration
        self._runner_config = init_dataclass_from_dict(runner_config, TaskRunnerConfig)
        self._output_dir = self._runner_config.output_dir
        self._save_formatted_output = self._runner_config.save_formatted_output
        # Subclasses that support restart (GenerationTask) set this in their own
        # __init__ after calling super(). Transformation tasks always do a full
        # pass so they never restart.
        self._restart_generation: bool = False

        # Initialize necessary state variables
        self._post_proc_id = 0  # Tracks Post processor IDs
        self.machine_data = []  # Tracks machine generated/transformed data

        # Determine store name from __init__ OR datastore's property, if defined OR task's name
        self._store_name = store_name or (datastore or dict()).pop("store_name", None) or self._name

        # Store raw datastore kwargs for each store type. _post_init() merges
        # these with _minimum_datastore_config (which includes the correct
        # restart flag) once subclasses have finished their own setup.
        self._minimum_datastore_config: dict = {}
        _ds_extra = datastore if datastore is not None else {TYPE_KEY: "default"}
        self._datastore_cfg = dict(_ds_extra)
        self._final_datastore_config = dict(
            final_datastore if final_datastore is not None else _ds_extra
        )
        self._formatted_datastore_config = dict(
            formatted_datastore if formatted_datastore is not None else _ds_extra
        )
        self._task_card_datastore_cfg = dict(_ds_extra)

        # Datastores
        self._intermediate_data_datastore: Datastore = None
        self._final_datastore: Datastore = None
        self._formatted_datastore: Datastore = None

        # Configure formatter
        self._formatter: Formatter | None = (
            get_formatter(
                formatter.get(TYPE_KEY),
                **{k: v for k, v in formatter.items() if k != TYPE_KEY},
            )
            if formatter
            else None
        )

    def _post_init(self):
        """Finalise datastore configs and initialise stores + logger.

        Called by each concrete subclass at the end of its own ``__init__``,
        after all subclass-specific state (notably ``_restart_generation``) has
        been set.  Separating this from ``Task.__init__`` ensures that
        ``_minimum_datastore_config`` is built with the correct restart flag.
        """
        self._minimum_datastore_config = {
            "restart": self._restart_generation,
            "output_dir": self._output_dir,
        }
        self._datastore_cfg = {
            **self._minimum_datastore_config,
            **{
                k: v
                for k, v in self._datastore_cfg.items()
                if k not in self._minimum_datastore_config
            },
        }
        self._final_datastore_config = {
            **self._minimum_datastore_config,
            **{
                k: v
                for k, v in self._final_datastore_config.items()
                if k not in self._minimum_datastore_config
            },
        }
        self._formatted_datastore_config = {
            **self._minimum_datastore_config,
            **{
                k: v
                for k, v in self._formatted_datastore_config.items()
                if k not in self._minimum_datastore_config
            },
        }
        self._task_card_datastore_cfg = {
            **self._minimum_datastore_config,
            **{
                k: v
                for k, v in self._task_card_datastore_cfg.items()
                if k not in self._minimum_datastore_config
            },
        }
        self._save_task_card()
        self._init_datastores()
        self._init_logger()

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def runner_config(self) -> TaskRunnerConfig:
        """Returns the run config of the task.

        Returns:
            TaskRunnerConfig: Run config for the task
        """
        return self._runner_config

    @property
    def name(self) -> str:
        """Returns name of task.

        Returns:
            str: Name of task
        """
        return self._name

    @property
    def task_description(self) -> str:
        """Returns the task description.

        Returns:
            str: Task description
        """
        return self._task_description

    @property
    def restart_generation(self) -> bool:
        """Flag used to determine if datastores should be reset.

        Returns:
            bool: Whether or not to reset datastores.
        """
        return self._restart_generation

    @property
    def task_card(self) -> TaskRunCard:
        """Returns the task card.

        Returns:
            TaskRunCard: Task card
        """
        return self._task_card

    @property
    def store_name(self) -> str:
        return self._store_name

    @property
    def datastore_configuration(self) -> Dict:
        return self._datastore_cfg

    @property
    def datastore(self) -> Datastore:
        """Returns the datastore of the class.

        Returns:
            Datastore: Datastore
        """
        return self._intermediate_data_datastore

    @property
    def final_datastore(self) -> Datastore:
        """Returns the final datastore of the class.

        Returns:
            Datastore: Final datastore
        """
        return self._final_datastore

    @property
    def formatted_datastore(self) -> Datastore:
        """Returns the formatted datastore of the class.

        Returns:
            Datastore: Formatted datastore
        """
        return self._formatted_datastore

    @property
    def task_results_datastore(self) -> Datastore:
        """Returns the task results datastore.

        Returns:
            Datastore: Task results datastore
        """
        return self._task_results_datastore

    @property
    def formatter(self) -> Formatter | None:
        return self._formatter

    @property
    def logger(self) -> logging.Logger:
        """Returns the task-scoped logger.

        Returns:
            logging.Logger: Logger scoped to this task
        """
        return self._logger

    @property
    def log_handler(self) -> LogDatastoreHandler | None:
        """Returns the LogDatastoreHandler attached to this task's logger.

        The DataBuilder registers this handler with its FanOutHandler so that
        run-level log records are duplicated into this task's log store.

        Returns:
            LogDatastoreHandler | None: The task's log handler, or None
        """
        return self._log_handler

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def _save_task_card(self):
        """Saves task card to datastore."""
        task_card_ds_kwargs = {
            "store_name": os.path.join(self._store_name, "task_card"),
            **self._task_card_datastore_cfg,
        }
        task_card_datastore = get_datastore(
            task_card_ds_kwargs.get(TYPE_KEY), **task_card_ds_kwargs
        )

        prev_card = None
        if not self._restart_generation:
            prev_task_cards: List[Dict] = [
                card
                for card in task_card_datastore.load_data()
                if card["build_id"] == self.task_card.build_id
            ]
            if prev_task_cards:
                prev_card = TaskRunCard(**prev_task_cards[-1])
                self.task_card.run_id = prev_card.run_id

        if self.task_card.run_id is None:
            raise ValueError("TaskCard.run_id cannot be set to None")

        task_card_datastore.save_data([self.task_card.to_dict()])
        task_card_datastore.close()

    def _clear_stale_postproc_datastores(self):
        """Clear postproc_data_N stores left over from a previous run.

        On restart we don't know how many post-processing epochs the prior run
        produced, so we probe N=1,2,3,... calling clear() on each until we
        reach one that had nothing to clear (i.e. no prior run wrote it).
        This uses the datastore abstraction so it works for any backend.
        """
        n = 1
        while True:
            ds = get_datastore(
                self._datastore_cfg.get(TYPE_KEY),
                **{
                    "store_name": os.path.join(self._store_name, f"postproc_data_{n}"),
                    **self._datastore_cfg,
                    "restart": False,  # we call clear() ourselves below
                },
            )
            if not ds.exists():
                break
            ds.clear()
            n += 1

    def _init_datastores(self):
        # When restarting, wipe any postproc_data_N stores from the prior run
        # before initialising the current run's datastores.  Without this a
        # user inspecting the output directory would see stale postproc_data_5
        # files alongside the current run's postproc_data_1 / postproc_data_2,
        # with no way to tell which epoch count is authoritative.
        if self._restart_generation:
            self._clear_stale_postproc_datastores()

        # Initialize datastore to save intermediate generated/transformed data
        self._intermediate_data_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "data"),
                **self._datastore_cfg,
            },
        )

        # Initialize datastore to save final generated/transformed data
        self._final_datastore = get_datastore(
            self._final_datastore_config.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "final_data"),
                **self._final_datastore_config,
                "restart": True,  # always restart final datastore
            },
        )

        # Initialize datastore to save formatted generated/transformed data
        self._formatted_datastore = get_datastore(
            self._formatted_datastore_config.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "formatted_data"),
                **self._formatted_datastore_config,
                "restart": True,  # always restart formatted datastore
            },
        )

        # Initialize datastore to save task results
        self._task_results_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "task_results"),
                **self._datastore_cfg,
            },
        )

    def _init_logger(self):
        # Create a task-scoped child logger. It inherits the log level from the
        # root dgt_logger and propagates records up to it (reaching the stdout
        # handler) without adding handlers to the global logger.
        self._logger = logging.getLogger(f"{_TASK_LOGGER_PREFIX}.{self._name}")
        self._log_handler: LogDatastoreHandler | None = None

        # Initialize the log datastore using the same type and config as all
        # other task artifact stores. The store_name path follows the convention
        # of every other store: <task_store_name>/logs. On restart, the datastore
        # is recreated from scratch (restart=True wipes the existing store file).
        log_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "logs"),
                **self._datastore_cfg,
                "restart": self._restart_generation,
            },
        )

        log_handler = LogDatastoreHandler(log_datastore)
        self._logger.addHandler(log_handler)
        self._log_handler = log_handler

        # On resume, emit a structured marker so the log file records exactly
        # where this process invocation picked up. run_id is the same as the
        # previous run (set by _save_task_card when resuming) so no linking is
        # needed — the marker is purely a process boundary indicator.
        if not self._restart_generation:
            self._logger.info(
                "run_resumed",
                extra={
                    "event": "run_resumed",
                    "task_name": self._name,
                    "pid": os.getpid(),
                },
            )

    def set_new_postprocessing_datastore(self):
        """Sets default datastore (which is used to gather data for final_datastore)

        Args:
            datastore (Datastore): Datastore to set
        """
        self._post_proc_id += 1
        pp_ds_kwargs = {
            "store_name": os.path.join(self._store_name, f"postproc_data_{self._post_proc_id}"),
            **self._datastore_cfg,
            "restart": True,
        }

        # close existing datastore before updating
        self._intermediate_data_datastore.close()

        # update pointer to new datastore
        self._intermediate_data_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY), **pp_ds_kwargs
        )

    def instantiate_input_example(self, **kwargs: Any) -> INPUT_DATA_TYPE:
        """Instantiate an input example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an input example object.

        Returns:
            INPUT_DATA_TYPE: An instance of INPUT_DATA_TYPE.
        """
        return self.INPUT_DATA_TYPE(task_name=kwargs.pop("task_name", self.name), **kwargs)

    def instantiate_output_example(self, **kwargs: Any) -> OUTPUT_DATA_TYPE:  # type: ignore
        """Instantiate an output example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an output example object.

        Returns:
            OUTPUT_DATA_TYPE: An instance of OUTPUT_DATA_TYPE.
        """
        return self.OUTPUT_DATA_TYPE(**kwargs)

    def load_intermediate_data(self) -> List[DataPoint]:
        """Loads intermediate data produced during SDG (will be used to resume SDG). This function loads the data from datastore, which is either
            the latest datastore defined during post processing or the original input/output datastore.

        Returns:
            List[DataPoint]: List of DataPoint that has been loaded
        """
        loaded_data = self._intermediate_data_datastore.load_data() or []
        return [self.instantiate_output_example(**d) for d in loaded_data]

    def save_intermediate_data(
        self,
        new_data: Union[DataPoint, List[DataPoint]],
    ) -> None:
        """Saves intermediate data produced during SDG (useful for checkpointing).

        Args:
            new_data (Union[DataPoint, List[DataPoint]]): List of DataPoint to save.
        """
        if not isinstance(new_data, list):
            new_data: List[DataPoint] = [new_data]

        to_save = [d if isinstance(d, dict) else d.to_dict() for d in new_data]
        self._intermediate_data_datastore.save_data(to_save)

    def save_final_data(self) -> None:
        """Saves final data that can be used directly for training."""
        iterators = self._intermediate_data_datastore.load_iterators() or []
        if iterators:
            iterator = iterators[0]  # since there is only one data.jsonl
            self._logger.info("Saving final data to %s", self.final_datastore.output_path)
            self.final_datastore.save_data(iterator)

    def apply_formatting(self, data: OUTPUT_DATA_TYPE) -> Dict:  # type: ignore
        """Apply formatting to output data instance.

        Args:
            data (OUTPUT_DATA_TYPE): Data to be formatted.

        Returns:
            Dict: Formatted data.
        """

        if not self.formatter:
            raise ValueError('"formatter" must be specified in the task to apply formatting.')

        return self.formatter.apply(data=data)

    def save_formatted_data(self) -> None:
        """Saves formatted instruction-tuning data that can be used directly for training."""
        if self._save_formatted_output:
            iterators = self.final_datastore.load_iterators() or []
            if iterators:
                iterator = iterators[0]  # since we only have one final_data.jsonl
                formatted_iterator = (
                    self.apply_formatting(self.instantiate_output_example(**d)) for d in iterator
                )
                self._logger.info(
                    "Saving formatted data to %s", self.formatted_datastore.output_path
                )
                self.formatted_datastore.save_data(formatted_iterator)

    def finish(self) -> None:
        """Method for wrapping up task execution. Called after `is_complete` signals task has completed"""
        # close datastores, which may involve writing any buffered data
        self._intermediate_data_datastore.close()

        # save final data
        self.save_final_data()
        self.save_formatted_data()

        # close
        self.final_datastore.close()
        self.formatted_datastore.close()

        # Remove all handlers except _log_handler from the task-scoped logger.
        # _log_handler (LogDatastoreHandler) is intentionally left open so that
        # run-level events emitted after execute_tasks() returns (run_finished,
        # run_errored) can still reach this task's log file via the FanOutHandler.
        # DataBuilder.close() calls close_log_handler() after those events fire.
        for handler in self._logger.handlers[:]:
            if handler is self._log_handler:
                continue
            handler.close()
            self._logger.removeHandler(handler)

    def close_log_handler(self) -> None:
        """Close and remove the log handler after all run-level events have been written.

        Called by DataBuilder.close() after run_finished / run_errored has been
        emitted, so the final events are guaranteed to land in the log file.
        """
        if self._log_handler is not None:
            self._log_handler.close()
            self._logger.removeHandler(self._log_handler)
            self._log_handler = None

    def record_task_results(self, intermediate_data: List[DataPoint]) -> Dict[str, Any]:
        """Creates a json object that captures all relevant information describing the results of the SDG task. The json
        object should be stored in the self._task_results_datastore once filled.

        Args:
            intermediate_data (List[DataPoint]): Data that has been generated that should be summarized and reported

        Returns:
            dict: task results
        """
        return {}

    @abstractmethod
    def is_complete(self) -> bool:
        """Indicates whether task has completed.

        Returns:
            bool: Whether task is complete or not.
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def get_batch_examples(self) -> List[DataPoint]:
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )


# ===========================================================================
#                       GENERATION
# ===========================================================================
class GenerationTask(Task):
    """This class is intended to hold general task information"""

    def __init__(
        self,
        *args,
        runner_config: Mapping | GenerationTaskRunnerConfig,
        seed_examples: Optional[List[Any]] = None,
        seed_datastore: Optional[Dict] = None,
        dataloader: Optional[Dict] = None,
        **kwargs: Any,
    ):
        """Initializes generation task object.

        Args:
            runner_config (Union[Mapping, GenerationTaskRunnerConfig]): Config specifying the run settings of the generation task.
            seed_examples (Optional[List[Any]]): A list of seed examples.
            seed_datastore (Optional[Dict]): A dictionary containing the configuration for the seed datastore.
            dataloader (Optional[Dict]): A dictionary containing the configuration for the seed dataloader.

        """
        # Initialize parent
        super().__init__(
            *args,
            runner_config=init_dataclass_from_dict(runner_config, GenerationTaskRunnerConfig),
            **kwargs,
        )

        # restart_generation lives on GenerationTaskRunnerConfig (not the base
        # TaskRunnerConfig) because it is meaningless for transformation tasks.
        # Set it before _post_init() so datastores are initialised with the
        # correct restart flag.
        self._restart_generation = self.runner_config.restart_generation
        self._post_init()

        self._seed_batch_size = self.runner_config.seed_batch_size
        self._machine_batch_size = self.runner_config.machine_batch_size
        self._num_outputs_to_generate = self.runner_config.num_outputs_to_generate

        for attr in [
            "seed_batch_size",
            "machine_batch_size",
            "num_outputs_to_generate",
        ]:
            if getattr(self, f"_{attr}") < 0:
                raise ValueError(
                    f"Cannot have negative value of {getattr(self, f'_{attr}')} for {attr} parameter"
                )

        # Initialize seed datastore
        self._seed_examples = seed_examples
        self._seed_datastore_config = {
            **self._minimum_datastore_config,
            **(seed_datastore if seed_datastore is not None else {TYPE_KEY: "default"}),
        }
        self._seed_datastore = get_datastore(
            self._seed_datastore_config.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "seed_data"),
                "data": self._seed_examples,
                **self._seed_datastore_config,
                "restart": False,
            },
        )

        # In-memory cache for seed examples. Populated on first call to
        # get_seed_examples(). Avoids repeated dataloader construction and
        # disk reads across calls to sample_examples() within a run.
        # Pass reload=True to get_seed_examples() to bust the cache (e.g.
        # when the seed datastore is updated mid-run).
        self._seed_examples_cache: List[DataPoint] | None = None

        # Initialize seed dataloader
        self._dataloader = None
        self._seed_dataloader_config = (
            dataloader if dataloader is not None else {TYPE_KEY: "default", "loop_over": True}
        )
        self._dataloader_state_datastore: Datastore = None
        self._dataloader_state: Any = None
        self._init_dataloader()

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def seed_batch_size(self) -> int:
        """Number of seed examples to pass as input to round of generation.

        Returns:
            int: Number of seed examples
        """
        return self._seed_batch_size

    @property
    def machine_batch_size(self) -> int:
        """Number of machine examples to pass as input to round of generation.

        Returns:
            int: Number of machine examples
        """
        return self._machine_batch_size

    @property
    def batch_size(self) -> int:
        """Total number of items returned by get_batch_examples() per call.

        Combines seed examples (human-authored) and machine-generated examples
        (synthetic data accumulated so far). This is the actual batch size each
        worker future receives, and is used by the concurrent executor to
        estimate how many futures are needed to cover remaining work.

        Returns:
            int: seed_batch_size + machine_batch_size
        """
        return self._seed_batch_size + self._machine_batch_size

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================

    def _init_dataloader(self) -> None:
        """Initialize dataloader to iterate over seed data"""
        # Initialize dataloader state datastore (should be same as base datastore)
        self._dataloader_state_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "dataloader_state"),
                **self._datastore_cfg,
            },
        )

        # Initialize dataloader
        self._dataloader = get_dataloader(
            self._seed_dataloader_config.get(TYPE_KEY),
            datastore=self._seed_datastore,
            **self._seed_dataloader_config,
        )

    def save_dataloader_state(self) -> None:
        """Saves the state of the dataloader"""
        curr_state = self._dataloader.get_state()
        if self._dataloader_state != curr_state:
            self._dataloader_state = curr_state
            self._dataloader_state_datastore.save_data([curr_state])

    def load_dataloader_state(self) -> None:
        """Loads the state of the dataloader"""
        prev_state = self._dataloader_state_datastore.load_data()
        if prev_state:
            self._dataloader.set_state(prev_state[-1])
            self._dataloader_state = prev_state

    def finish(self) -> None:
        """Method for wrapping up task execution. Called after `is_complete` signals task has completed"""
        # close dataloader state datastores, which may involve writing any buffered data
        self._dataloader_state_datastore.close()

        super().finish()

    @property
    def num_outputs_to_generate(self) -> int:
        """Returns the number of outputs to generate for this task."""
        return self._num_outputs_to_generate

    def is_complete(self):
        """Indicates whether task has completed.

        Returns:
            bool: Whether task is complete or not.
        """
        return len(self.machine_data) >= self._num_outputs_to_generate

    def get_example(self) -> DataPoint:
        """Returns single seed example from dataloader.

        Returns:
            DataPoint: Seed example to be used for SDG.
        """
        try:
            seed_example = self.instantiate_input_example(**next(self._dataloader))
            seed_example.is_seed = True
            return seed_example
        except StopIteration:
            return None

    def get_seed_examples(self, reload: bool = False) -> List[DataPoint]:
        """Gets all seed examples and returns them in a list.

        Results are cached after the first load. Subsequent calls return the
        cached list unless reload=True is passed, which forces a fresh read
        from the datastore and updates the cache.

        Args:
            reload: If True, bypass the cache and reload from disk.

        Returns:
            List[DataPoint]: List of all seed examples
        """
        if self._seed_examples_cache is not None and not reload:
            return self._seed_examples_cache
        dataloader = get_dataloader(
            self._seed_dataloader_config.get(TYPE_KEY),
            datastore=self._seed_datastore,
            **self._seed_dataloader_config,
        )
        seed_data = []
        try:
            while ex := self.instantiate_input_example(**next(dataloader)):
                ex.is_seed = True
                seed_data.append(ex)
        except StopIteration:
            pass
        self._seed_examples_cache = seed_data
        return self._seed_examples_cache

    def sample_examples(
        self,
        k: int,
        seed_fraction: float | None = None,
        reload: bool = False,
    ) -> List[Any]:
        """Randomly sample up to k examples from seed data and/or machine-generated data.

        Safe to call from anywhere — stages, __call__ implementations, inner
        thread pools. Does NOT advance the main dataloader. Thread-safe.

        k is always the total cap. seed_fraction controls the seed/synthetic
        split within that cap. When seed_fraction=None the split is derived
        from the task config ratio (seed_batch_size / (seed_batch_size +
        machine_batch_size)), so k still governs the total requested.

        Sampling is random and without replacement on both sides. If either
        pool has fewer items than its allocated quota, you get however many
        are available — there is no backfill from the other pool. Callers
        that strictly need k examples should check the returned length.

        Args:
            k: Total number of examples requested. Actual count may be less
                if either pool has insufficient data.
            seed_fraction: Fraction of k to draw from the seed datastore.
                Remainder is drawn from machine_data.
                - 1.0: seeds only
                - 0.0: synthetic only
                - None (default): split derived from seed_batch_size /
                  (seed_batch_size + machine_batch_size) task config ratio.
            reload: If True, bypass the seed examples cache and reload from
                disk before sampling. Passed through to get_seed_examples().

        Returns:
            List of randomly sampled examples (INPUT_DATA_TYPE | OUTPUT_DATA_TYPE).
            Seeds are marked is_seed=True.
        """
        if seed_fraction is not None and not (0.0 <= seed_fraction <= 1.0):
            raise ValueError(f"seed_fraction must be in [0.0, 1.0], got {seed_fraction}")

        if seed_fraction is None:
            total = self._seed_batch_size + self._machine_batch_size
            fraction = self._seed_batch_size / total if total > 0 else 0.0
        else:
            fraction = seed_fraction

        n_seed = round(k * fraction)
        n_synthetic = k - n_seed

        results: List[Any] = []

        # Random sample from seed pool (no state mutation on main dataloader).
        if n_seed > 0:
            pool = self.get_seed_examples(reload=reload)
            n = min(n_seed, len(pool))
            results.extend(random.sample(pool, k=n) if n < len(pool) else pool)

        # Random sample from machine-generated data.
        if n_synthetic > 0 and self.machine_data:
            n = min(n_synthetic, len(self.machine_data))
            results.extend(random.sample(self.machine_data, k=n))

        return results

    def get_batch_examples(self) -> List[DataPoint]:
        """Returns batch of examples from dataloader. Mixes examples from seed data and machine-generated data.

        FRAMEWORK-INTERNAL: Called exclusively by the framework execution loop
        (_spawn_futures in ConcurrentGenerationDataBuilder, call_with_task_list
        in GenerationDataBuilder). Advances the main dataloader state — do not
        call from __call__, stages, or any user code. Use sample_examples()
        instead for ICL sampling or any other purpose outside the framework loop.

        Returns:
            List[DataPoint]: List of examples to be used by SDG process.
        """
        outputs = []

        # get outputs from seed data loader sequentially
        for _ in range(self._seed_batch_size):
            example = self.get_example()
            if example is None:
                break
            outputs.append(example)

        # get outputs from machine batch randomly
        m_data = self.machine_data
        if m_data and len(m_data) > self._machine_batch_size:
            m_data = random.sample(m_data, k=self._machine_batch_size)

        outputs.extend(m_data)

        return outputs


# ===========================================================================
#                       TRANSFORMATION
# ===========================================================================
class TransformationTask(Task):
    """TransformTask is a subclass of Task that has default values that are more conducive to transformation tasks."""

    def __init__(
        self,
        *args,
        runner_config: Mapping | TransformationTaskRunnerConfig,
        data: List[Any] | Dict,
        dataloader: Optional[Dict] = None,
        **kwargs,
    ):
        """Initializes transformation task object.

        Args:
            runner_config (Union[Mapping, GenerationTaskRunnerConfig]): Config specifying the run settings of the transformation task.
            data (Union[List[Any], Dict]): A list of examples to transform OR a dictionary containing the configuration for the transformation data datastore.
            dataloader (Optional[Dict]): A dictionary containing the configuration for the transformation data dataloader.

        """
        # Initialize parent
        super().__init__(
            *args,
            runner_config=init_dataclass_from_dict(runner_config, TransformationTaskRunnerConfig),
            **kwargs,
        )

        # Transformation tasks always restart with a clean slate because the
        # cardinality of the transformation (1→1, M→N, etc.) is unknown, so
        # resuming a partial output is never safe.
        self._restart_generation = True
        self._post_init()

        # Extract required variables from the runner configuration
        self._transform_batch_size = self._runner_config.transform_batch_size

        # Initialize transformation data datastore
        self._transformation_data_datastore_config = {
            **self._minimum_datastore_config,
            **(data if isinstance(data, dict) else {TYPE_KEY: "default"}),
        }
        self._transformation_data_datastore = get_datastore(
            self._transformation_data_datastore_config.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "transformation_data"),
                "data": data if isinstance(data, list) else None,
                **self._transformation_data_datastore_config,
                "restart": False,
            },
        )

        # Initialize transformation data dataloader
        self._dataloader = None
        self._transformation_data_dataloader_config = (
            dataloader if dataloader is not None else {TYPE_KEY: "default", "loop_over": False}
        )
        self._dataloader_state_datastore: Datastore = None
        self._dataloader_state: Any = None
        self._init_dataloader()

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def transform_batch_size(self) -> int:
        """Number of examples to pass as input to round of transformation.

        Returns:
            int: Number of examples to transform in a single batch
        """
        return self._transform_batch_size

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def _init_dataloader(self) -> None:
        """Initialize dataloader to iterate over seed data"""
        # Initialize dataloader state datastore (should be same as base datastore)
        self._dataloader_state_datastore = get_datastore(
            self._datastore_cfg.get(TYPE_KEY),
            **{
                "store_name": os.path.join(self._store_name, "dataloader_state"),
                **self._datastore_cfg,
            },
        )

        # Initialize dataloader
        self._dataloader = get_dataloader(
            self._transformation_data_dataloader_config.get(TYPE_KEY),
            datastore=self._transformation_data_datastore,
            **self._transformation_data_dataloader_config,
        )

    def save_dataloader_state(self) -> None:
        """Saves the state of the dataloader"""
        curr_state = self._dataloader.get_state()
        if self._dataloader_state != curr_state:
            self._dataloader_state = curr_state
            self._dataloader_state_datastore.save_data([curr_state])

    def load_dataloader_state(self) -> None:
        """Loads the state of the dataloader"""
        prev_state = self._dataloader_state_datastore.load_data()
        if prev_state:
            self._dataloader.set_state(prev_state[-1])
            self._dataloader_state = prev_state

    def finish(self) -> None:
        """Method for wrapping up task execution. Called after `is_complete` signals task has completed"""
        # close dataloader state datastores, which may involve writing any buffered data
        self._dataloader_state_datastore.close()

        super().finish()

    def is_complete(self):
        """Indicates whether task has completed.

        Returns:
            bool: Whether task is complete or not.
        """
        return True

    def get_example(self) -> DataPoint:
        """Returns single example from dataloader.

        Returns:
            DataPoint: example to be transformed.
        """
        try:
            return self.instantiate_input_example(**next(self._dataloader))
        except StopIteration:
            return None

    def get_batch_examples(self) -> List[DataPoint]:
        """Returns batch of examples from dataloader. Mixes examples from seed data and machine-generated data.

        Returns:
            List[DataPoint]: List of examples to be used by SDG process.
        """
        outputs = []

        # get outputs from seed data loader sequentially
        for _ in range(self._transform_batch_size):
            example = self.get_example()
            if example is None:
                break
            outputs.append(example)

        return outputs

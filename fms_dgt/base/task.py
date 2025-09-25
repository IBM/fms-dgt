# Standard
from abc import abstractmethod
from logging import FileHandler
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union
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
from fms_dgt.constants import TYPE_KEY
from fms_dgt.utils import (
    DGT_LOG_FORMATTER,
    dgt_logger,
    group_data_by_attribute,
    init_dataclass_from_dict,
)

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
    return group_data_by_attribute(data_list, "task_name")


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
        self._restart_generation = self._runner_config.restart_generation

        # Initialize necessary state variables
        self._post_proc_id = 0  # Tracks Post processor IDs
        self.machine_data = []  # Tracks machine generated/transformed data

        # Determine store name from __init__ OR datastore's property, if defined OR task's name
        self._store_name = store_name or (datastore or dict()).pop("store_name", None) or self._name

        # Datastore configurations
        self._minimum_datastore_config = {
            "restart": self.restart_generation,
            "output_dir": self._output_dir,
        }
        self._datastore_cfg = {
            **self._minimum_datastore_config,
            **(datastore if datastore is not None else {TYPE_KEY: "default"}),
        }
        self._final_datastore_config = {
            **self._minimum_datastore_config,
            **(final_datastore if final_datastore is not None else self._datastore_cfg),
        }
        self._formatted_datastore_config = {
            **self._minimum_datastore_config,
            **(formatted_datastore if formatted_datastore is not None else self._datastore_cfg),
        }
        self._task_card_datastore_cfg = {
            **self._minimum_datastore_config,
            **self._datastore_cfg,
        }

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

        # Save task card, initialize datastores and logger
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

    def _init_datastores(self):
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
        # Initialize logger only when default datastore is used
        if self._datastore_cfg.get(TYPE_KEY) == "default":

            # Create logs directory in output_dir
            logs_dir = os.path.join(
                self._datastore_cfg.get("output_dir", "output"),
                self._store_name,
                "logs",
            )
            os.makedirs(logs_dir, exist_ok=True)

            # Clean up previous logs, if restart requested
            if self._restart_generation:
                for existing_log_file in Path(logs_dir).glob("*.log"):
                    os.remove(existing_log_file)

            # Set up a new log file
            log_file_handler = FileHandler(
                filename=os.path.join(logs_dir, f"{os.getpid()}.log"),
            )
            log_file_handler.setFormatter(DGT_LOG_FORMATTER)
            dgt_logger.addHandler(log_file_handler)

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
            dgt_logger.info("Saving final data to %s", self.final_datastore.output_path)
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
                dgt_logger.info("Saving formatted data to %s", self.formatted_datastore.output_path)
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

        # Extract required variables from the runner configuration
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

    def get_seed_examples(self) -> List[DataPoint]:
        """Gets all seed examples and returns them in a list.

        Returns:
            List[DataPoint]: List of all seed examples
        """
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
        return seed_data

    def get_batch_examples(self) -> List[DataPoint]:
        """Returns batch of examples from dataloader. Mixes examples from seed data and machine-generated data.

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

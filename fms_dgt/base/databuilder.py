# Standard
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import json
import logging
import os
import time

# Local
from fms_dgt.base.block import Block, get_row_name
from fms_dgt.base.data_objects import DataBuilderConfig, DataPoint
from fms_dgt.base.registry import get_block, get_block_class
from fms_dgt.base.task import GenerationTask, Task, TransformationTask
from fms_dgt.base.telemetry import Span, _NoOpSpanWriter
from fms_dgt.constants import (
    BLOCKS_KEY,
    DATASET_TYPE,
    DATASTORES_KEY,
    NAME_KEY,
    STORE_NAME_KEY,
    TASK_NAME_KEY,
    TYPE_KEY,
)
from fms_dgt.log import FanOutHandler
from fms_dgt.utils import (
    all_annotations,
    convert_byte_size,
    init_dataclass_from_dict,
    merge_dictionaries,
)


# ===========================================================================
#                       BASE
# ===========================================================================
class DataBuilder:
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: Task = Task

    def __init__(
        self,
        config: Mapping | DataBuilderConfig | None = None,
        max_stalled_attempts: int = 5,
        verify_block_type: bool = False,
        task_kwargs: List[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes data builder object.

        Args:
            config (Union[Mapping, DataBuilderConfig], optional): Config specifying all databuilder settings.
            max_stalled_attempts (int, optional): Maximum number of data generation/transformation loop iterations that do not return new data before terminating.
            task_kwargs (List[Dict], optional): List of task_kwargs for each task to be executed by this data builder.
        """
        # Initialize necessary variables
        self._config = init_dataclass_from_dict(config, DataBuilderConfig)
        self._max_stalled_attempts = (
            max_stalled_attempts
            if max_stalled_attempts and isinstance(max_stalled_attempts, int)
            else 5
        )

        # Initialize tasks
        self._tasks: List[Task] = [self.TASK_TYPE(**entry) for entry in task_kwargs]

        # just grab first task's build_id
        self._build_id = self._tasks[0].task_card.build_id

        # Initialize builder-scoped logger with fan-out to active task logs.
        # Must come before _init_blocks() so the FanOutHandler is available
        # to inject into each block's configuration.
        self._init_logger()

        # Initialize blocks
        self._block_datastores_per_task = {}
        self._blocks: List[Block] = self._init_blocks(verify_block_type=verify_block_type)

        # Initialize epoch counter (always start with 1)
        self._epoch = 1

        # SpanWriter set by configure_telemetry() in generate_data.py before
        # execute_tasks() is called. Defaults to a no-op so execute_tasks()
        # can reference it safely even if telemetry is disabled or not yet wired.

        self._span_writer = _NoOpSpanWriter()

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def name(self) -> str:
        """Returns the name of the data builder

        Returns:
            str: name string
        """
        return self._config.name

    @property
    def config(self) -> DataBuilderConfig:
        """Returns the DataBuilderConfig associated with this class.

        Returns:
            DataBuilderConfig: Config specifying data builder settings
        """
        return self._config

    @property
    def blocks(self) -> List[Block]:
        """Returns the blocks associated with this class.

        Returns:
            List[Block]: List of blocks to be used in this data builder
        """
        return self._blocks

    @property
    def tasks(self) -> List[Task]:
        """Returns the tasks associated with this class.

        Returns:
            List[SdgTask]: List of tasks to be used in this data builder
        """
        return self._tasks

    @property
    def build_id(self) -> str:
        """Returns the build ID for this run.

        Returns:
            str: Build ID
        """
        return self._build_id

    @property
    def run_id(self) -> str:
        """Returns the run ID for this execution (from the first task card).

        Returns:
            str: Run ID
        """
        return self._tasks[0].task_card.run_id if self._tasks else None

    @property
    def span_writer(self):
        """Returns the SpanWriter for this run (set by configure_telemetry)."""
        return self._span_writer

    @span_writer.setter
    def span_writer(self, writer) -> None:
        """Set the SpanWriter and propagate it to all blocks."""
        self._span_writer = writer
        for block in self._blocks:
            block.span_writer = writer

    # ===========================================================================
    #                       HELPER FUNCTIONS
    # ===========================================================================
    def _init_blocks(self, verify_block_type: bool = False) -> List[Block]:
        """This method does two things:

        (1) It initializes each block object specified in self._config.blocks
        (2) It sets the block-attributes for a DataBuilder to be those initialized blocks (where the block is assumed to be assigned to `obj_name`)
            - In the process of doing this, it checks that the type specified in the DataBuilder class's attribute matches the block type that was initialized

        This method is intended to be overloaded when type checking is not necessary (e.g., in the case of the Pipeline class).
        """
        # Initialize necessary variables
        blocks: List[Block] = []
        type_annotations = {
            k: v
            for k, v in all_annotations(type(self)).items()
            if isinstance(v, type) and issubclass(v, Block)
        }
        found_annotations = []
        required_annotations = [k for k in type_annotations if getattr(self, k, True) is not None]

        # Verify all necessary details are available
        if len(self._config.blocks) != len([b.get("name") for b in self._config.blocks]):
            raise ValueError(f"Duplicate block in '{self.name}' data builder detected")

        # TODO: need to handle nested blocks
        for block_configuration in self._config.blocks:
            # Verify "name" and "type" properties are provided for each block
            for mandatory_property in [NAME_KEY, TYPE_KEY]:
                if mandatory_property not in block_configuration:
                    raise KeyError(
                        f"'{mandatory_property}' field missing in data builder config from block with args:\n{json.dumps(block_configuration, indent=4)}"
                    )

            # Extract block's name and type from the configuration and determine associated class
            block_name = block_configuration.get(NAME_KEY)
            block_type = block_configuration.get(TYPE_KEY)
            block_class = get_block_class(block_type)

            # Check types for all blocks specified in the databuilder definition
            if block_name in type_annotations:
                requested_block_type = type_annotations[block_name]
                is_same_type = issubclass(block_class, requested_block_type)

                # verify block type, if requested
                if verify_block_type and not is_same_type:
                    raise TypeError(
                        f"Retrieved block type ({block_class}) != retrieved block type ({requested_block_type}) for {block_name} in DataBuilder {self.__class__}"
                    )
                elif not is_same_type:
                    self.logger.warning(
                        "Retrieved block type (%s) for %s does not match type (%s) specified in DataBuilder %s",
                        block_class,
                        block_name,
                        requested_block_type,
                        self.__class__,
                    )

                # Add to found annotation list
                found_annotations.append(block_name)

            # Extend block configuration to include databuilder name and the
            # shared FanOutHandler so block log records route to the same task
            # log files as builder records.
            block_configuration = {
                "builder_name": self.name,
                "fanout_handler": self._fanout_handler,
                **block_configuration,
            }

            # Extend block configuration with datastore configuration, if necessary
            if DATASTORES_KEY not in block_configuration or not block_configuration[DATASTORES_KEY]:
                block_configuration[DATASTORES_KEY] = []
                self._block_datastores_per_task[block_name] = defaultdict(list)
                for task in self._tasks:
                    # Create block store name
                    block_store_name = os.path.join(task.store_name, BLOCKS_KEY, block_name)
                    self._block_datastores_per_task[block_name][task.name].append(block_store_name)
                    block_configuration[DATASTORES_KEY].append(
                        {
                            **task.datastore_configuration,
                            STORE_NAME_KEY: block_store_name,
                        }
                    )
            else:
                if not isinstance(block_configuration[DATASTORES_KEY], list):
                    raise ValueError(
                        f'"{DATASTORES_KEY}" property in block configuration YAML must be provided as a list.'
                    )
                for task in self._tasks:
                    self._block_datastores_per_task[block_name][task.name] = [
                        datastore_cfg[STORE_NAME_KEY]
                        for datastore_cfg in block_configuration[DATASTORES_KEY]
                    ]

            # Initialize block
            block = get_block(block_type, **block_configuration)

            # Add block to databuilder's member variables for ease of access
            setattr(self, block_name, block)

            # Add to global list
            blocks.append(block)

        # Report missing block configuration
        missing_block_specs = set(required_annotations).difference(found_annotations)
        if missing_block_specs:
            raise ValueError(
                f"Some blocks - [{', '.join(missing_block_specs)}] - did not have a definition in the databuilder config"
            )

        # Return
        return blocks

    # ===========================================================================
    #                       LOGGER
    # ===========================================================================
    def _init_logger(self) -> None:
        """Initialize a builder-scoped logger backed by a FanOutHandler.

        All task file handlers are pre-registered here so that run-level
        events emitted before execute_tasks() (run_started, etc.) are captured
        in the task log files. execute_tasks() calls _register_task_log_handler
        again for active tasks, which is a no-op thanks to idempotent register().
        The builder logger is a child of dgt_logger so records also propagate
        to the root stdout handler.
        """
        self._logger = logging.getLogger(f"fms_dgt.builder.{self.name}")
        self._fanout_handler = FanOutHandler()
        self._logger.addHandler(self._fanout_handler)

        # Pre-register all task handlers so pre-execute_tasks log records
        # (run_started, etc.) are written to every task log file.
        for task in self._tasks:
            self._register_task_log_handler(task)

    @property
    def logger(self) -> logging.Logger:
        """Returns the builder-scoped logger.

        Records emitted here are written to all currently-active task log files
        via the FanOutHandler, making each task log self-contained.

        Returns:
            logging.Logger: Builder-scoped logger
        """
        return self._logger

    def _register_task_log_handler(self, task: Task) -> None:
        """Register a task's file handler with the FanOutHandler.

        Call this just before a task enters the active set so that subsequent
        log records are written to its log file.
        """
        if task.log_handler is not None:
            self._fanout_handler.register(task.name, task.log_handler)

    def _unregister_task_log_handler(self, task: Task) -> None:
        """Unregister a task's file handler from the FanOutHandler.

        Call this before task.finish() so that the handler is removed from
        the fan-out set before it is closed by task teardown.
        """
        self._fanout_handler.unregister(task.name)

    def _pretty_print(self, block: Block, prefix: str = None):
        execution_times = []
        peak_memory = []
        for entry in block.profiler_data["executions"]:
            execution_times.append(entry["time"])
            peak_memory.append(entry["peak_memory"])

        execution_time_str = (
            f"{round(mean(execution_times), 2)} ± {round(stdev(execution_times), 2) if len(execution_times) > 1 else float(0)}"
            if execution_times
            else "-"
        )
        peak_memory_str = (
            f"{convert_byte_size(mean(peak_memory))} ± {convert_byte_size(stdev(peak_memory) if len(peak_memory) > 1 else 0)}"
            if peak_memory
            else "-"
        )
        completion_token_usage_str = (
            f'{block.profiler_data["usage"]["tokens"]["completion"]:,}'
            if "usage" in block.profiler_data
            and block.profiler_data["usage"]
            and "tokens" in block.profiler_data["usage"]
            and block.profiler_data["usage"]["tokens"]
            and "completion" in block.profiler_data["usage"]["tokens"]
            else "-"
        )
        prompt_token_usage_str = (
            f'{block.profiler_data["usage"]["tokens"]["prompt"]:,}'
            if "usage" in block.profiler_data
            and block.profiler_data["usage"]
            and "tokens" in block.profiler_data["usage"]
            and block.profiler_data["usage"]["tokens"]
            and "prompt" in block.profiler_data["usage"]["tokens"]
            else "-"
        )

        block_name = f"{prefix}.{block.name}" if prefix else block.name
        report_str = f"{block_name if len(block_name) <= 17 else block_name[:17]+'...':<20}\t{execution_time_str:17}\t{peak_memory_str:24}\t{completion_token_usage_str:^19}\t{prompt_token_usage_str:^19}"
        self.logger.info(report_str)

        return

    def _report_block_wise_profiling_information(self, block: Block, prefix: str = None):
        if block.blocks:
            for child_block in block.blocks:
                self._report_block_wise_profiling_information(
                    block=child_block,
                    prefix=f"{prefix}.{block.name}" if prefix else block.name,
                )
        else:
            self._pretty_print(
                block=block,
                prefix=prefix,
            )

    def _report_profiling_information(self):
        self.logger.info("*" * 99)
        self.logger.info('\t\tEXECUTION PROFILER FOR DATABUILDER "%s"', self.name)
        self.logger.info("*" * 99)
        self.logger.info(
            "Block%s\tTime (mean ± std)\tPeak Memory (mean ± std)\tTokens (Completion)\tTokens (Prompt)",
            " " * 15,
        )
        for block in self._blocks:
            self._report_block_wise_profiling_information(block=block)

        self.logger.info("*" * 99)

    def close(self):
        # Step 1: Report profiling information
        self._report_profiling_information()

        # Step 2: Close and free up resource consumed by blocks
        for block in self._blocks:
            block.close()

        # Step 3: Record run completion
        self.record_run_results(
            update={"status": "completed", "end_time": int(datetime.now().timestamp())}
        )

        # Step 4: Unregister and close task log handlers now that all run-level
        # events (run_finished / run_errored) have been written by the caller.
        # _unregister_task_log_handler is NOT called inside execute_tasks() so
        # that the FanOutHandler can still route those final events to each
        # task's log file.
        for task in self._tasks:
            self._unregister_task_log_handler(task)
            task.close_log_handler()

    def execute_postprocessing(self, completed_tasks: List[Task]):
        """Executes any postprocessing required after tasks have completed.

        Args:
            completed_tasks (List[SdgTask]): tasks that have been completed and can undergo postprocessing
        """
        if self._config.postprocessors:
            # TODO: This could potentially be very expensive,
            # we could make this more efficient by using datastore.load_iterators()
            post_processed_data = []
            for task in completed_tasks:
                data = task.datastore.load_data()

                for block_info in self._config.postprocessors:
                    block_info = dict(block_info)
                    block_name = block_info.pop(NAME_KEY)
                    block = next(iter([b for b in self.blocks if b.name == block_name]))

                    # execute postprocessing
                    data = block(
                        [
                            {
                                **data_point,
                                "store_names": self.get_block_store_names(
                                    block_name=block_name,
                                    task_name=data_point[TASK_NAME_KEY],
                                ),
                            }
                            for data_point in data
                        ],
                        **block_info,
                    )

                # Collate post processed data
                post_processed_data.extend(data)

            # write results
            self._write_postprocessing(completed_tasks, post_processed_data)

    def _write_postprocessing(self, completed_tasks: List[Task], data: DATASET_TYPE):
        # write outputs to datastore
        for task in completed_tasks:
            # update pointer to current datastore
            task.set_new_postprocessing_datastore()

        # TODO: make this more efficient
        tasks: Dict[str, Tuple[Task, int]] = {task.name: [task, 0] for task in completed_tasks}
        for d in data:
            task_name = d[TASK_NAME_KEY]
            # have to cast this to OUTPUT_TYPE
            d = {
                k: v
                for k, v in d.items()
                if k in tasks[task_name][0].OUTPUT_DATA_TYPE.get_field_names()
            }
            tasks[task_name][0].save_intermediate_data(d)
            tasks[task_name][1] += 1

        self.logger.info("*" * 99)
        self.logger.info("\t[EPOCH %d]\tPOST-PROCESSING RESULTS", self._epoch)
        self.logger.info("*" * 99)
        self.logger.info(
            "Task%s\tBefore\t\t\tAfter",
            " " * 36,
        )
        for task_name, (task, ct) in tasks.items():
            report_str = f"{task_name if len(task_name) <= 37 else task_name[:37]+'...':<40}\t{len(task.machine_data):^10}\t{ct:^20}"
            self.logger.info(report_str)

        self.logger.info("*" * 99)

        # load_intermediate_data loads from postprocess datastore
        for task in completed_tasks:
            task.machine_data = task.load_intermediate_data()

    def record_run_results(self, update: Dict[str, Any] = None) -> None:
        """Record the results of data generation for all tasks. It is guaranteed to only be called once SDG is complete"""
        # Step 1: Iterate over each task
        for task in self._tasks:
            # Step 1.a: Initialize run result
            run_result = {}

            # Step 1.b: Load existing run results
            existing_run_results = task.task_results_datastore.load_data()

            # Step 1.c: Set run result to latest existing run result
            if existing_run_results:
                run_result = deepcopy(existing_run_results[-1])

            # Step 1.d: Merge with update
            # Step 1.d.i: If 'metrics' key found in update, warn and ignore
            if update and "metrics" in update:
                self.logger.warning(
                    '"metrics" is a protected field and will be set by task.record_task_results().'
                )
                del update["metrics"]

            # Step 1.d.ii: Merge
            run_result = merge_dictionaries(run_result, update if update else {})

            # Step 1.e: Calculate task specific metrics
            # Step 1.e.i: Load task's intermediate data
            interm_data = task.load_intermediate_data()

            # Step 1.e.ii: Initialize metrics dictionary
            run_result["metrics"] = {
                "Number of data produced": len(interm_data),
            }

            # Step 1.e.iii: Extend metrics dictionary using task reported results
            for k, v in task.record_task_results(interm_data).items():
                run_result["metrics"][k] = v

            # Step 1.f: Save
            task.task_results_datastore.save_data([run_result])

    def get_block_store_names(self, block_name: str, task_name: str) -> List[str]:
        return self._block_datastores_per_task[block_name][task_name]

    @abstractmethod
    def execute_tasks(self):
        """
        Main entry point for task execution.
        """

        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def call_with_task_list(self, tasks: List[Task], *args, **kwargs) -> Iterable[DataPoint]:
        """Executes data builder __call__ function for all in-progress tasks. Is executed in the inner loop of `execute_tasks`

        Args:
            tasks (List[Task]): List of in-progress tasks

        Returns:
            Iterable[DataPoint]: List of data instances generated by the __call__ function
        """

        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[DataPoint]:
        """Contains the main logic of a data builder. Returns a list of data objects that reflect new instances

        Returns:
            List[DataPoint]: List of new data objects
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )


# ===========================================================================
#                       GENERATION
# ===========================================================================
class GenerationDataBuilder(DataBuilder):
    """A data builder represents a means of constructing data for a set of tasks"""

    TASK_TYPE: GenerationTask = GenerationTask

    def __init__(
        self,
        *args,
        num_attempts_to_complete: int = 1000000,
        **kwargs: Any,
    ) -> None:
        """Initializes data builder object.

        Args:
            num_attempts_to_complete (int, optional): Maximum number of attempts (generation loop iterations) to execute before terminating.

        """
        # Initialize parent
        super().__init__(*args, **kwargs)

        self._num_attempts_to_complete = (
            num_attempts_to_complete
            if num_attempts_to_complete and isinstance(num_attempts_to_complete, int)
            else 1000000
        )

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def execute_tasks(self):
        """
        Main entry point for task execution.
        Default behavior executes a loop until all tasks are complete, where each loop generates synthetic data.
        """

        # Load existing machine data, if available
        for task in self._tasks:
            task.machine_data = task.load_intermediate_data()
            if task.machine_data:
                self.logger.debug("Loaded %s machine-generated data", len(task.machine_data))
            task.load_dataloader_state()

        # Identify active and completed task
        active_tasks = []
        for task in self._tasks:
            if task.is_complete():
                # Task was already complete before this run started (resume path).
                # Emit task_finished immediately — no task_started because the work
                # happened in a prior run.
                self.logger.info(
                    "Task '%s' finished.",
                    task.name,
                    extra={
                        "event": "task_finished",
                        "task_name": task.name,
                        "reason": "already_complete",
                    },
                )
                task.finish()
            else:
                self._register_task_log_handler(task)
                self.logger.info(
                    "Task '%s' started.",
                    task.name,
                    extra={
                        "event": "task_started",
                        "task_name": task.name,
                    },
                )
                active_tasks.append(task)

        # Initialize necessary variables
        start_time = time.time()
        remaining_unstalled_generation_attempts_per_task = {
            task.name: self._max_stalled_attempts for task in active_tasks
        }
        remaining_unstalled_epochs_per_task = {
            task.name: self._max_stalled_attempts for task in active_tasks
        }
        attempt = 0
        tasks_in_generation_phase: List[GenerationTask] = (
            active_tasks + []
        )  # short-hand to create a new list

        # Generate in a loop till
        # - All tasks finished generating and postprocessing requested number of datapoints OR
        # - Maximum number of attempts to complete tasks is reached
        while active_tasks and attempt <= self._num_attempts_to_complete:
            _epoch_span = Span(
                "dgt.epoch",
                self._span_writer,
                parent_span_name="dgt.run",
                epoch=self._epoch,
                active_task_names=",".join(t.name for t in active_tasks),
                active_task_count=len(active_tasks),
            )
            _epoch_span.__enter__()
            self.logger.info("*" * 99)
            self.logger.info("\t\t\t\tEPOCH: %s", self._epoch)
            self.logger.info("*" * 99)
            self.logger.info(
                "Epoch %s started with %s active task(s): %s",
                self._epoch,
                len(active_tasks),
                ", ".join(t.name for t in active_tasks),
                extra={
                    "event": "epoch_started",
                    "epoch": self._epoch,
                    "active_task_names": [t.name for t in active_tasks],
                    "active_task_count": len(active_tasks),
                },
            )

            # Reset tasks in postprocessing
            tasks_in_postprocessing_phase: List[GenerationTask] = (
                []
            )  # short-hand to create a new list

            # Generate in a loop till
            # - No tasks in generation phase OR
            # - Maximum number of attempts to complete tasks is reached
            attempt_within_epoch = 0
            while tasks_in_generation_phase and attempt <= self._num_attempts_to_complete:
                # Increment attempt counter
                attempt += 1
                attempt_within_epoch += 1

                # Generate data for all active tasks
                generated_data_counter_per_task = {
                    active_task.name: 0 for active_task in active_tasks
                }
                for generated_datapoint in self.call_with_task_list(
                    tasks_in_generation_phase, attempt
                ):
                    # Identify relevant task using "task_name"
                    relevant_task = next(
                        task
                        for task in tasks_in_generation_phase
                        if get_row_name(generated_datapoint) == task.name
                    )
                    relevant_task.save_intermediate_data(generated_datapoint)
                    relevant_task.save_dataloader_state()

                    # Add to machine data
                    relevant_task.machine_data.append(generated_datapoint)

                    # Increment generated data counter for the relevant task
                    generated_data_counter_per_task[relevant_task.name] += 1

                # Report generation statistics
                self.logger.info("*" * 99)
                self.logger.info(
                    "\t[EPOCH %d]\tGENERATION RESULTS AFTER ATTEMPT %d (TOTAL ATTEMPTS: %d)",
                    self._epoch,
                    attempt_within_epoch,
                    attempt,
                )
                self.logger.info("*" * 99)
                self.logger.info(
                    "Task%s\tCurrent\t\t\tTotal",
                    " " * 36,
                )
                for task in tasks_in_generation_phase:
                    report_str = f"{task.name if len(task.name) <= 37 else task.name[:37]+'...':<40}\t{generated_data_counter_per_task[task.name]:^10}\t{len(task.machine_data):^20}"
                    self.logger.info(report_str)

                self.logger.info("*" * 99)

                # Reset remaining unstalled attempts
                for task_name, count in generated_data_counter_per_task.items():
                    if count > 0:
                        remaining_unstalled_generation_attempts_per_task[task_name] = (
                            self._max_stalled_attempts
                        )
                    else:
                        remaining_unstalled_generation_attempts_per_task[task_name] -= 1

                # Move stalled or completed task to post-processing
                remaining_tasks_in_generation_phase = []
                for task in tasks_in_generation_phase:
                    if (
                        task.is_complete()
                        or remaining_unstalled_generation_attempts_per_task[task.name] <= 0
                    ):
                        tasks_in_postprocessing_phase.append(task)
                    else:
                        remaining_tasks_in_generation_phase.append(task)

                # Reset tasks in generation phase
                tasks_in_generation_phase = (
                    remaining_tasks_in_generation_phase + []
                )  # short-hand to create a new list

            # Launch postprocessing
            self.logger.info("Launch postprocessing")
            self.execute_postprocessing(tasks_in_postprocessing_phase)
            for task in tasks_in_postprocessing_phase:
                if task.machine_data:
                    remaining_unstalled_epochs_per_task[task.name] = self._max_stalled_attempts
                else:
                    remaining_unstalled_epochs_per_task[task.name] -= 1
            self.logger.info("Postprocessing completed")

            # Remove stalled or completed task
            for task in tasks_in_postprocessing_phase:
                if (
                    task.is_complete()
                    or remaining_unstalled_generation_attempts_per_task[task.name] <= 0
                    or remaining_unstalled_epochs_per_task[task.name] <= 0
                ):
                    # Issue warning for stalled tasks in generation phase
                    if remaining_unstalled_generation_attempts_per_task[task.name] <= 0:
                        self.logger.warning(
                            "Task %s has not generated any data in the last %s attempts, terminating task",
                            task.name,
                            self._max_stalled_attempts,
                        )

                    # Issue warning for stalled task in post-processing phase
                    if remaining_unstalled_epochs_per_task[task.name] <= 0:
                        self.logger.warning(
                            "Task %s has not produced any data in the last %s attempts after post-processing, terminating task",
                            task.name,
                            self._max_stalled_attempts,
                        )

                    # Determine finish reason for the structured event.
                    if task.is_complete():
                        _reason = "complete"
                    elif remaining_unstalled_generation_attempts_per_task[task.name] <= 0:
                        _reason = "stalled_generation"
                    else:
                        _reason = "stalled_postprocessing"
                    self.logger.info(
                        "Task '%s' finished.",
                        task.name,
                        extra={
                            "event": "task_finished",
                            "task_name": task.name,
                            "reason": _reason,
                        },
                    )
                    # Do NOT unregister here — _log_handler must stay in the
                    # FanOutHandler so run_finished (emitted after execute_tasks
                    # returns) reaches this task's log file.  DataBuilder.close()
                    # handles both unregister and close_log_handler().
                    task.finish()
                else:
                    tasks_in_generation_phase.append(task)

            # Reset active tasks
            active_tasks = tasks_in_generation_phase + []

            # Report need of a new epoch and increament epoch counter, if necessary
            if active_tasks and attempt <= self._num_attempts_to_complete:
                report_str = f"Triggering new epoch since {len(active_tasks)} task{'s are' if len(active_tasks) > 1 else ' is'} still pending."
                self.logger.info(report_str)
                self._epoch += 1

            _epoch_span.__exit__(None, None, None)
            self.logger.info("*" * 99)

        # Report generation duration
        self.logger.info("Generation took %.2fs", time.time() - start_time)

    def call_with_task_list(
        self, tasks: List[GenerationTask], request_idx: int
    ) -> Iterable[DataPoint]:
        """Executes data builder __call__ function for all in-progress tasks. Is executed in the inner loop of `execute_tasks`

        Args:
            tasks (List[SdgTask]): List of in-progress tasks
            request_idx (int): The iteration of `execute_tasks` this method was called at

        Returns:
            Iterable[DataPoint]: List of data instances generated by the __call__ function
        """

        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        args = [request_idx, data_pool]
        kwargs = dict()
        return self(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[DataPoint],
    ) -> List[DataPoint]:
        """Contains the main logic of a data builder. Takes in a list of data objects to be used as seed data and returns a list of data objects that reflect new instances

        Args:
            request_idx (int): The iteration of `execute_tasks` this method was called at
            instruction_data (List[DataPoint]): List of data objects to be used as seed data

        Returns:
            List[DataPoint]: List of new data objects that can be used for instruction-tuning
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )


# ===========================================================================
#                       TRANSFORMATION
# ===========================================================================


class TransformationDataBuilder(DataBuilder):
    """This class is designed to have sensible default methods for transformation use cases"""

    def execute_tasks(self):
        """Main entry point for task execution. Default behavior iterates over all tasks and applies the transformation to each task's data."""
        tasks = self._tasks + []

        # Load tasks
        for task in tasks:
            if not isinstance(task, TransformationTask):
                raise TypeError(
                    f"Task {task.name} must inherit from TransformTask class to be used with TransformationDataBuilder"
                )

            task.load_dataloader_state()
            self._register_task_log_handler(task)
            self.logger.info(
                "Task '%s' started.",
                task.name,
                extra={
                    "event": "task_started",
                    "task_name": task.name,
                },
            )

        # Initialize necessary variables
        start_time = time.time()

        # Transform tasks
        for transformed_datapoint in self.call_with_task_list(tasks):
            # save incrementally
            task = next(task for task in tasks if get_row_name(transformed_datapoint) == task.name)
            task.save_intermediate_data(transformed_datapoint)
            task.save_dataloader_state()

            task.machine_data.append(transformed_datapoint)

        # Report performance
        self.logger.info("*" * 99)
        self.logger.info(
            "%s\t\tTotal",
            " " * 40,
        )
        for task in tasks:
            report_str = f"{task.name if len(task.name) <= 37 else task.name[:37]+'...':<40}\t\t{len(task.machine_data):^20}"
            self.logger.info(report_str)

        self.logger.info("*" * 99)

        # Launch postprocessing
        self.logger.info("Launch postprocessing")
        self.execute_postprocessing(tasks)
        self.logger.info("Postprocessing completed")

        # Terminate completed tasks
        for task in tasks:
            self.logger.info(
                "Task '%s' finished.",
                task.name,
                extra={
                    "event": "task_finished",
                    "task_name": task.name,
                    "reason": "complete",
                },
            )
            # Do NOT unregister here — see generation execute_tasks() note above.
            task.finish()

        # Report transformation duration
        self.logger.info("Transformation took %.2fs", time.time() - start_time)

    def call_with_task_list(self, tasks: List[TransformationTask]) -> Iterable[DataPoint]:
        """Executes data builder __call__ function for all in-progress tasks.

        Args:
            tasks (List[SdgTask]): List of in-progress tasks

        Returns:
            Iterable[DataPoint]: List of data instances generated by the __call__ function
        """
        # Create pool of data points to transform
        data_pool = [e for task in tasks for e in task.get_batch_examples()]

        # Iterate till all data points are transformed
        while data_pool:
            args = [data_pool]
            kwargs = dict()

            # Trigger transformation
            for output in self(*args, **kwargs):
                yield output

            # Reset data pool with next set of examples to transform
            data_pool = [e for task in tasks for e in task.get_batch_examples()]

    def __call__(
        self,
        data_points: List[DataPoint],
    ) -> List[DataPoint]:
        """Contains the main logic of a data builder. Takes in a list of data objects to transform and returns a list of transformed data objects

        Args:
            data_points (List[DataPoint]): List of data objects to transform

        Returns:
            List[DataPoint]: List of transformed data objects
        """
        raise NotImplementedError(
            f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
        )

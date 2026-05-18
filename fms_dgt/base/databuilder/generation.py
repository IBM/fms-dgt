# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Iterable, List
import time

# Local
from fms_dgt.base.block import get_row_name
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.base.databuilder.base import DataBuilder
from fms_dgt.base.task import GenerationTask
from fms_dgt.base.telemetry import Span


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
            _counts_before_postproc = {
                task.name: len(task.machine_data) for task in tasks_in_postprocessing_phase
            }
            with Span(
                "dgt.postprocessing",
                self._span_writer,
                parent_span_name="dgt.epoch",
                epoch=self._epoch,
                task_count=len(tasks_in_postprocessing_phase),
            ):
                self.execute_postprocessing(tasks_in_postprocessing_phase)
            for task in tasks_in_postprocessing_phase:
                if task.machine_data:
                    remaining_unstalled_epochs_per_task[task.name] = self._max_stalled_attempts
                else:
                    remaining_unstalled_epochs_per_task[task.name] -= 1
            self.logger.info(
                "Postprocessing completed",
                extra={
                    "event": "postprocessing_finished",
                    "epoch": self._epoch,
                    "task_counts": {
                        task.name: {
                            "before": _counts_before_postproc[task.name],
                            "after": len(task.machine_data),
                        }
                        for task in tasks_in_postprocessing_phase
                    },
                },
            )

            # Remove stalled or completed task
            _epoch_finish_reasons: dict = {}
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
                    _epoch_finish_reasons[task.name] = _reason
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

            self.logger.info(
                "Epoch %s finished.",
                self._epoch,
                extra={
                    "event": "epoch_finished",
                    "epoch": self._epoch,
                    "generation_attempts": attempt_within_epoch,
                    "task_counts": {
                        task.name: len(task.machine_data) for task in tasks_in_postprocessing_phase
                    },
                    "finish_reasons": _epoch_finish_reasons,
                },
            )
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

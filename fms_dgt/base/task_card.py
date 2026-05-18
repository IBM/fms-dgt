# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict, dataclass
from typing import Dict, Optional
import os
import uuid

_DEFAULT_BUILD_ID = "exp"


@dataclass
class TaskRunCard:
    """Identity and provenance record for a single task execution.

    Three concepts govern how DiGiT identifies work across runs:

    ``build_id`` — experiment identity.
        Set by the user (via ``--build-id`` on the CLI, defaults to ``"exp"``).
        Stable across all runs of the same experiment. Shared by every task
        in a single ``generate_data()`` call. Use this to group everything
        belonging to "the ABC experiment."

    ``run_id`` — execution-chain identity, per task.
        A UUID generated fresh on the first run of a task. On resume
        (``restart_generation=False``), inherited from the previous run of the
        same task under the same ``build_id``, so the chain stays unbroken.
        On restart (``restart_generation=True``), a new UUID is assigned.
        Purpose: link datapoints back to the process invocation that created
        them ("datapoints 50-75 came from this execution chain"). Not a
        sequential counter. Not shared across tasks — each task has its own.

    ``task_name`` — task discriminator.
        The human-readable name of the task (from the task config YAML).
        Shared ``build_id`` and per-task ``run_id`` together do not uniquely
        identify a task within a run — ``task_name`` is the key that does.
        Datapoints carry it explicitly. Log records inside ``DataBuilder``
        execution carry it via the datapoint. Setup-time log records (e.g.
        enrichments) must inject it explicitly at the call site.

    In structured logging and OTel, the natural grouping keys are:
        ``build_id`` + ``task_name``  →  all records for one task across runs
        ``build_id`` + ``run_id``     →  all records for one execution chain
    """

    task_name: str  # name of task
    databuilder_name: str  # name of databuilder associated with task
    task_spec: Optional[Dict] = None  # json string for task settings
    databuilder_spec: Optional[Dict] = None  # json string for databuilder settings
    build_id: Optional[str] = None  # id of entity executing the task
    run_id: Optional[str] = None  # unique ID for the experiment
    save_formatted_output: Optional[bool] = None  # will save formatted output
    process_id: Optional[int] = None  # unique process ID for the experiment

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())
        if self.build_id is None:
            self.build_id = _DEFAULT_BUILD_ID  #  default to something generic
        if self.save_formatted_output is None:
            self.save_formatted_output = False
        if self.process_id is None:
            self.process_id = os.getpid()

    def to_dict(self):
        return asdict(self)

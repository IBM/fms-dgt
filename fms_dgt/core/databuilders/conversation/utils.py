# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Type, TypeVar
import random

# Local
from fms_dgt.core.databuilders.conversation.constants import DEVELOPER_ROLE, SYSTEM_ROLE
from fms_dgt.core.databuilders.conversation.data_objects import Step


def get_instruction_role(model_id_or_path: str):
    if "gpt" in model_id_or_path:
        return DEVELOPER_ROLE
    else:
        return SYSTEM_ROLE


T = TypeVar("T")


def get_last_step_of_type(steps: List[Step], tgt_class: Type[T]) -> T | None:
    if steps:
        # Find last step for specified target class, if requested
        if tgt_class:
            for step in reversed(list(steps)):
                if isinstance(step, tgt_class):
                    return step
            return None
        # Overall last step
        return steps[-1]
    else:
        return None


def get_first_step_of_type(steps: List[Step], tgt_class: Type[T]) -> T | None:
    # Find first step for specified target class
    for step in steps:
        if isinstance(step, tgt_class):
            return step


def get_random_step_of_type(steps: List[Step], tgt_class: Type[T]) -> T | None:
    # Find random step for specified target class
    opts = []
    for step in steps:
        if isinstance(step, tgt_class):
            opts.append(step)
    if opts:
        return random.choice(opts)


def get_all_steps_of_type(steps: List[Step], tgt_class: Type[T]) -> List[T]:
    matches = []
    # Find first step for specified target class
    for step in steps:
        if isinstance(step, tgt_class):
            matches.append(step)
    return matches

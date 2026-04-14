# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Iterable, Literal
import os

# Third Party
from datasets import Dataset
import pandas as pd

DATASET_ROW_TYPE = Dict[str, Any] | pd.Series
DATASET_TYPE = Iterable[DATASET_ROW_TYPE] | pd.DataFrame | Dataset

# these are variable names that we try to reuse across the codebase
# Universal
TYPE_KEY = "type"
NAME_KEY = "name"
STORE_NAME_KEY = "store_name"

# Task specific
RUNNER_CONFIG_KEY = "runner_config"
TASK_NAME_KEY = "task_name"
DATABUILDER_KEY = "data_builder"

# Databuilder specific
BLOCKS_KEY = "blocks"
DATASTORES_KEY = "datastores"
RAY_CONFIG_KEY = "ray_config"
TOOLS_KEY = "tools"
ENGINE_KEY = "engine"

DGT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

# Logger name for the root fms-dgt logger; all child loggers use this as a prefix
BASE_LOGGER_NAME = "fms_dgt"

# environment variables that should be known to DGT
DGT_ENV_VARS = {
    "DGT_DATA_DIR": "data",
    "DGT_OUTPUT_DIR": "output",
    "DGT_TELEMETRY_DIR": "telemetry",
    "DGT_TELEMETRY_DISABLE": "",
    "DGT_CACHE_DIR": ".cache",
}


# general types
class NotGiven:
    """
    Type that can be used in cases where `None` is not an appropriate default value
    """

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()

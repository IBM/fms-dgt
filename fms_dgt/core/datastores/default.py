# Standard
from typing import Iterator, List, TypeVar
import glob
import os

# Third Party
import datasets
import pandas as pd

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.registry import register_datastore
import fms_dgt.utils as utils

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
T = TypeVar("T")
_LAZY_LOAD_BUFFER_SIZE = 1000


# ===========================================================================
#                       HELPER FUNCTION
# ===========================================================================


def _read_file(data_format: str, data_path: str, **file_kwargs) -> Iterator:
    if data_format in [".txt", ".md"]:
        yield from [utils.read_file(data_path)]
    elif data_format == ".jsonl":
        yield from utils.read_jsonl(data_path, lazy=True)
    elif data_format == ".json":
        content = utils.read_json(data_path)
        if isinstance(content, list):
            yield from content
        else:
            yield from [content]
    elif data_format == ".yaml":
        content = utils.read_yaml(data_path)
        if isinstance(content, list):
            yield from content
        else:
            yield from [content]
    elif data_format == ".parquet":
        yield from utils.read_parquet(data_path, lazy=True, **file_kwargs)
    else:
        raise ValueError(f"Unhandled data format: {data_format}")


def _join_data(dataset: DATASET_TYPE, seed_data: List):
    if seed_data:
        if isinstance(dataset, list):
            dataset = dataset + seed_data
        elif isinstance(dataset, datasets.Dataset):
            seed_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=seed_data))
            dataset = datasets.concatenate_datasets([dataset, seed_dataset])
        else:
            raise ValueError(
                f"Data used for default 'load_dataset' method must be one of {DATASET_TYPE}"
            )
    return dataset


def _is_glob_path(path: str):
    """
    Checks if a given path string is a glob pattern.

    Args:
        path (str): The path string to check.

    Returns:
        bool: True if the path is a glob pattern, False otherwise.
    """
    return any(char in path for char in ["*", "?", "["])


# ===========================================================================
#                       MAIN CLASS
# ===========================================================================
@register_datastore("default")
class DefaultDatastore(Datastore):
    """Base Class for all data stores"""

    def __init__(
        self,
        output_dir: str = None,
        data_formats: List[str] = None,
        output_data_format: str = "jsonl",
        data: List[T] = None,
        data_path: str = None,
        data_split: str = "train",
        buffer_size: int = _LAZY_LOAD_BUFFER_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._output_dir = output_dir
        self._output_path = (
            os.path.join(output_dir, self.store_name + "." + output_data_format)
            if output_dir
            else None
        )
        self._data_formats = data_formats
        self._data_path = os.path.expandvars(data_path) if data_path else None
        self._data_split = data_split
        self._buffer_size = buffer_size
        self._data = data or []
        if self._restart and os.path.exists(self._output_path):
            os.remove(self._output_path)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # ===========================================================================
    #                       PROPERTIES
    # ===========================================================================
    @property
    def output_dir(self) -> str | None:
        return os.path.dirname(self._output_path) if self._output_path else None

    @property
    def output_path(self) -> str | None:
        return self._output_path if self._output_path else None

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def save_data(self, data_to_save: DATASET_TYPE | Iterator) -> None:

        output_data_format = os.path.splitext(self._output_path)[-1]

        if isinstance(data_to_save, pd.DataFrame):
            data_to_save = data_to_save.to_dict("records")

        if output_data_format == ".jsonl":
            utils.write_jsonl(data_to_save, self._output_path)
        elif output_data_format == ".yaml":
            raise NotImplementedError(
                f"Missing implementation in {self.__module__}.{self.__class__.__name__}"
            )
        elif output_data_format == ".parquet":
            utils.write_parquet(data_to_save, self._output_path)
        else:
            raise ValueError(f"Unhandled data format: {output_data_format}")

    def load_iterators(self) -> List[Iterator]:
        # ===========================================================================
        #                       HELPER FUNCTION
        # ===========================================================================
        def _get_iterator(data_path: str):
            if os.path.exists(data_path):
                # special handling for huggingface datasets
                if os.path.isdir(data_path):
                    return (
                        item
                        for item in utils.read_huggingface(
                            [data_path], self._data_split, lazy=True
                        )
                    )

                data_format = os.path.splitext(data_path)[-1]
                # for local files
                file_kwargs = {}
                if data_format == ".parquet":
                    file_kwargs = {"buffer_size": self._buffer_size}

                return (
                    item for item in _read_file(data_format, data_path, **file_kwargs)
                )

            return None

        # Initialize necessary variables
        all_iterators = []

        # Add data in the form of iterator, if applicable
        if self._data:
            all_iterators.append((data for data in self._data))

        # Process data path
        data_path = self._data_path if self._data_path else self._output_path

        # List of data paths referring to Huggingface dataset
        if isinstance(data_path, list):
            return [
                (
                    item
                    for item in utils.read_huggingface(
                        data_path, self._data_split, lazy=True
                    )
                )
            ]
        # Data path expressed via glob pattern
        elif _is_glob_path(data_path):
            data_paths = glob.glob(data_path, recursive=True)
            for data_path in data_paths:
                if self._data_formats:
                    if any(
                        [
                            data_path.endswith(data_format)
                            for data_format in self._data_formats
                        ]
                    ):
                        all_iterators.append(_get_iterator(data_path))
                else:
                    all_iterators.append(_get_iterator(data_path))
        else:
            iterator = _get_iterator(data_path)
            if iterator:
                all_iterators.append(iterator)

        return all_iterators

    def load_data(self) -> List[T]:

        loaded_data = []
        all_iterators = self.load_iterators()
        for iterator in all_iterators:
            data = [item for item in iterator]
            loaded_data = _join_data(loaded_data, data)

        return loaded_data

    def close(self):
        pass

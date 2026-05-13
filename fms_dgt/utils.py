# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import ChainMap
from dataclasses import MISSING
from dataclasses import fields as _dc_fields
from dataclasses import is_dataclass
from json import JSONDecodeError
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import ast
import copy
import csv
import fnmatch
import glob
import importlib.util
import json
import logging
import math
import os
import signal
import socket
import threading

# Third Party
import datasets
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Local
from fms_dgt.constants import BASE_LOGGER_NAME, NAME_KEY

# ===========================================================================
#                       LOGGER CONFIGURATION
# ===========================================================================


# Step 1: Create default log formatter
DGT_LOG_FORMATTER = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)

# Step 2: Initialize logger
dgt_logger = logging.getLogger(BASE_LOGGER_NAME)

# Step 3: Set up logging level
dgt_logger.setLevel(level=getattr(logging, os.getenv("LOG_LEVEL", "info").upper()))

# Step 4: Create and add stream handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(DGT_LOG_FORMATTER)
dgt_logger.propagate = False
dgt_logger.addHandler(_stream_handler)

# Local
# Step 5: Attach the contextvars-based RunContextFilter so that ALL records
# emitted on any fms_dgt logger — including module-level loggers in LLM
# connectors and utilities — carry build_id and run_id automatically when a
# run_context() is active.  Import is deferred here to avoid a circular
# import (fms_dgt.log.context imports nothing from utils).
from fms_dgt.log.context import RunContextFilter as _RunContextFilter  # noqa: E402

dgt_logger.addFilter(_RunContextFilter())


# ===========================================================================
#                       CONSTANTS
# ===========================================================================


T = TypeVar("T")


# ===========================================================================
#                       GENERIC HELPER FUNCTIONS
# ===========================================================================


def init_dataclass_from_dict(d_obj: Dict, inp_type: T) -> T:
    if isinstance(d_obj, inp_type):
        return d_obj
    elif isinstance(d_obj, dict):
        return inp_type(**d_obj)
    elif d_obj is None:
        return inp_type()
    else:
        raise ValueError(f"Unhandled input type {type(d_obj)}, cannot convert to type {inp_type}")


def merge_dictionaries(*args: List[dict]) -> Dict[str, Any]:
    # Step 1: Define update function
    def _update(d, u):
        for k, v in u.items():
            if k in d and isinstance(d[k], dict) and isinstance(v, dict):
                d[k] = _update(d[k], v)
            else:
                d[k] = v
        return d

    # Step 2: Set 1st dictionary as merged dictionary
    merged_dict = copy.deepcopy(args[0])

    # Step 3: Iterate add remaining dictionaries into merged dictionary
    for new_dict in args[1:]:
        _update(merged_dict, new_dict)

    # Step 4: Return merged dictionary
    return merged_dict


def sanitize_path(path: str) -> str:
    """
    Sanitize a path against directory traversals
    """
    return os.path.relpath(os.path.normpath(os.path.join(os.sep, path)), os.sep)


# Timeouts
class TimeoutException(Exception):
    pass


def execute_with_timeout(timeout: int, func: Callable, *args: Any, **kwargs: Any):

    def timeout_handler(signum: int, frame: Any):
        raise TimeoutException(f"Execution of {func} has exceeded time limit of {timeout}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # disable the alarm
        return result
    except Exception as e:
        signal.alarm(0)
        raise e


def get_all_subclasses(cls: T) -> List[T]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


# ===========================================================================
#                       PARSING FUNCTIONS
# ===========================================================================
def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict


# ===========================================================================
#                       DATA BUILDER HELPER FUNCTIONS
# ===========================================================================


def validate_block_sequence(block_list: List[Dict]):
    for block in block_list:
        if not isinstance(block, dict):
            raise ValueError("Block in block sequence must be a dictionary")
        if block.get(NAME_KEY) is None:
            raise ValueError(f"Must specify {NAME_KEY} in block {block}")


def all_annotations(cls) -> ChainMap:
    return ChainMap(
        *(c.__annotations__ for c in cls.__mro__ if getattr(c, "__annotations__", False))
    )


def pattern_match(patterns, source_list):
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


# ===========================================================================
#                       VLLM/ HF-TUNING
# ===========================================================================


def get_open_port(host: str, address_range: Tuple[int, int] = (8000, 8100)):
    for port in range(*address_range):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.close()
            dgt_logger.info("Port [%s] is available for host [%s]", port, host)
            return port
        # pylint: disable=broad-exception-caught
        except Exception:
            sock.close()

    # pylint: disable=broad-exception-raised
    raise Exception(
        f"Could not find available port for host [{host}] in address range {address_range}"
    ) from None


def get_one_line_from_process(process: Type[psutil.Popen]):
    return "\n".join(
        [
            proc.readline().decode("utf-8").strip()
            for proc in [
                process.stdout,
                process.stderr,
            ]
        ]
    ).strip()


# ===========================================================================
#                       REGISTRY HELPER FUNCTIONS
# ===========================================================================


def dynamic_import(import_module: str, throw_top_level_error: bool = False):
    """This function will attempt to import the module specified by `import_module`"""
    try:
        dgt_logger.debug("Attempting dynamic import of %s", import_module)
        importlib.import_module(import_module)
        return True
    except ModuleNotFoundError as e:
        if f"No module named '{import_module}" not in str(e) or throw_top_level_error:
            raise e
        return False


# ===========================================================================
#                       YAML HELPER FUNCTIONS
# ===========================================================================


def ignore_constructor(_, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, f"{module_name}.py"))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(
    yaml_path: str = None,
    yaml_config: str = None,
    yaml_dir: str = None,
    simple_mode: bool = False,
    encoding: str = "utf-8",
):
    constructor_fn = ignore_constructor if simple_mode else import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, mode="r", encoding=encoding) as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    if not yaml_dir:
        raise ValueError("YAML directory must be specified.")

    return process_yaml_config(yaml_config, yaml_dir, simple_mode, encoding=encoding)


def process_yaml_config(
    yaml_config: Dict,
    yaml_dir: str = None,
    simple_mode: bool = False,
    encoding: str = "utf-8",
):
    """Processes a provided yaml config.

    Args:
        yaml_config (Dict): Config to process.
        yaml_dir (str, optional): Directory where yaml_config was loaded from. Defaults to None.
    """

    def load_file(path):
        path = os.path.expandvars(path)
        if path.endswith(".yaml"):
            data = load_yaml_config(yaml_path=path, simple_mode=simple_mode)
        elif path.endswith(".jsonl"):
            data = []
            with open(path, "r", encoding=encoding) as file:
                for line in file:
                    json_obj = json.loads(line)
                    data.append(json_obj)
        else:
            with open(path, "r", encoding=encoding) as f:
                data = f.read()
        return data

    def _include(to_include: Any):
        if isinstance(to_include, list):
            ret_lst = []
            for x in to_include:
                contents = _include(x)
                if isinstance(x, str) and isinstance(contents, list):
                    ret_lst.extend(contents)
                else:
                    ret_lst.append(contents)
            return ret_lst
        elif isinstance(to_include, dict):
            return {k: _include(v) for k, v in to_include.items()}
        elif isinstance(to_include, str):
            to_include = os.path.expandvars(to_include)
            if os.path.isfile(to_include):  # check absolute
                return load_file(to_include)
            elif yaml_dir and os.path.isfile(os.path.join(yaml_dir, to_include)):  # check relative
                return load_file(os.path.join(yaml_dir, to_include))
            abs_matching_files = glob.glob(to_include)
            if abs_matching_files:  # check absolute w/ pattern
                return [load_file(x) for x in abs_matching_files]
            if yaml_dir:
                rel_matching_files = glob.glob(os.path.join(yaml_dir, to_include))
                if rel_matching_files:  # check relative w/ pattern
                    return [load_file(x) for x in rel_matching_files]
        raise ValueError(f"Unhandled input format in 'include' directive: {to_include}")

    if "include" in yaml_config:
        to_include = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(to_include, str):
            to_include = [to_include]

        final_yaml_config = dict()
        to_add = _include(to_include)
        if isinstance(to_include, list):
            new_entry = merge_dictionaries(*to_add)
            final_yaml_config.update(new_entry)
        elif isinstance(to_include, dict):
            final_yaml_config.update(to_add)
        else:
            raise ValueError(f"Unhandled input format in 'include' directive: {to_include}")

        final_yaml_config.update(yaml_config)
        return final_yaml_config

    return yaml_config


# ===========================================================================
#                       TASK HELPER FUNCTIONS
# ===========================================================================


def group_by(data: List[T], key: Callable[[T], Any]) -> Dict[Any, List[T]]:
    """Group a list of items by an arbitrary key function.

    Args:
        data: List of items to group.
        key: Callable that returns the grouping key for each item.

    Returns:
        Dict mapping each distinct key value to the list of items that produced it.
        Insertion order of first occurrence is preserved.
    """
    result: Dict[Any, List[T]] = {}
    for item in data:
        result.setdefault(key(item), []).append(item)
    return result


# ===========================================================================
#                       GENERATE DATA HELPER FUNCTIONS
# ===========================================================================


def read_task_file(file_path: str):
    if file_path.endswith(".yaml"):
        contents = load_yaml_config(file_path)

        if not contents:
            dgt_logger.warning("Skipping %s because it is empty!", file_path)
            return None

        if file_path.startswith("." + os.sep):
            file_path = file_path[len("." + os.sep) :]

        # get seed instruction data
        task = {
            **{
                "data_builder": "simple",
                "created_by": "",
                # FIXME: We should remove this since it is not required for transformation tasks
                # "seed_examples": [],
            },
            **contents,
        }

        return task


def read_tasks(data):
    tasks = []
    if os.path.isfile(data):  # data is file
        task = read_task_file(data)
        tasks.append(task)
    else:
        # TODO: fix this once done testing
        for directory, _, files in os.walk(data):
            for file_name in files:
                if file_name in ["task.yaml", "qna.yaml"]:
                    file_path = os.path.join(directory, file_name)
                    data = read_task_file(file_path)
                    if data:
                        tasks.append(data)

    return tasks


def import_builder(inp_dir: str) -> None:

    imp_path = inp_dir.replace(os.sep, ".")

    import_path = f"{imp_path}.generate"
    # we try both, but we will overwrite with include path
    try:
        dynamic_import(import_path)
    except ModuleNotFoundError as e:
        # we try both, but we will overwrite with include path
        if f"No module named '{imp_path}" not in str(e):
            raise e


def load_joint_config(yaml_path: str, encoding: str = "utf-8"):

    with open(yaml_path, mode="r", encoding=encoding) as f:
        config: dict = yaml.full_load(f)

    data_paths, db_overrides, task_overrides = ([], dict(), dict())

    for k, v in config.items():
        if k in ["databuilders", "tasks"]:
            if not isinstance(v, dict):
                raise ValueError(
                    f"'{k}' field in config must be provided as a dictionary where keys are the names of databuilders to override"
                )
            if k == "databuilders":
                db_overrides = v
            else:
                task_overrides = {
                    task_name: process_yaml_config(task_cfg) for task_name, task_cfg in v.items()
                }
        elif k == "task_files":
            if not isinstance(v, list):
                raise ValueError(f"'{k}' field in config must be provided as a list")
            data_paths = v
        else:
            raise ValueError("Config must only specify 'databuilders' and 'tasks' fields")

    return data_paths, db_overrides, task_overrides


def load_nested_paths(inp: Dict, base_dir: str = None):
    def _is_file(text: str) -> bool:
        return any([text.endswith(ext) for ext in [".json", ".yaml", ".txt"]])

    def _load_file(path: str, encoding: str = "utf-8"):
        if path.endswith(".json"):
            with open(path, mode="r", encoding=encoding) as f:
                return json.load(f)
        elif path.endswith(".yaml"):
            with open(path, mode="r", encoding=encoding) as f:
                return yaml.safe_load(f)
        elif path.endswith(".txt"):
            with open(path, mode="r", encoding=encoding) as f:
                return str(f.read())
        return path

    def _get_path(fname: str, parent_dir: str):
        if os.path.isfile(fname):
            return os.path.normpath(fname)
        elif parent_dir and os.path.isfile(os.path.join(parent_dir, fname)):
            return os.path.normpath(os.path.join(parent_dir, fname))

    def _pull_paths(d: Union[List, Dict, str], parent_dir: str):
        if isinstance(d, dict):
            for k in d.keys():
                d[k] = _pull_paths(d[k], parent_dir)
        elif isinstance(d, list):
            for i, entry in enumerate(d):
                d[i] = _pull_paths(entry, parent_dir)
        elif isinstance(d, str) and d and _is_file(d):
            # assigns file_path then checks that file_path is not 'None'
            if (
                file_path := _get_path(d, parent_dir)
            ) not in checked_files and file_path is not None:
                checked_files.add(file_path)
                return _pull_paths(_load_file(file_path), os.path.dirname(file_path))
        return d

    checked_files = set()
    new_dict = _pull_paths(copy.deepcopy(inp), base_dir)

    return new_dict


def try_parse_json_string(json_string: str):
    if not isinstance(json_string, str):
        return None
    try:
        return json.loads(json_string)
    except json.decoder.JSONDecodeError:
        try:
            json_string = (
                json_string.replace(": true", ": True")
                .replace(": false", ": False")
                .replace(": null", ": None")
            )
            return json.loads(json.dumps(ast.literal_eval(json_string)))
        except (json.decoder.JSONDecodeError, SyntaxError, ValueError, TypeError):
            return None


# ===========================================================================
#                       GENERIC FILE LOADING FUNCTIONS
# ===========================================================================


def read_file(file_path: str, encoding: str = "utf-8"):
    with open(file_path, mode="r", encoding=encoding) as fp:
        return fp.read()


def read_yaml(file_path: str, encoding: str = "utf-8"):
    file_path = os.path.expandvars(file_path)
    with open(file_path, mode="r", encoding=encoding) as fp:
        data = yaml.safe_load(fp)
    return data


def read_json(file_path: str, encoding: str = "utf-8"):
    file_path = os.path.expandvars(file_path)
    with open(file_path, mode="r", encoding=encoding) as fp:
        try:
            data = json.load(fp)
        except ValueError:
            data = []
    return data


def read_jsonl(file_path: str, encoding: str = "utf-8", lazy: bool = False):
    file_path = os.path.expandvars(file_path)

    def _yield(file_path: str, encoding: str = "utf-8"):
        with open(file_path, mode="r", encoding=encoding) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except JSONDecodeError as err:
                        dgt_logger.warning("Decoding error %s for line: %s", str(err), line)

    if lazy:
        return _yield(file_path=file_path, encoding=encoding)
    else:
        data = []
        with open(file_path, mode="r", encoding=encoding) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as err:
                        dgt_logger.warning("Decoding error %s for line: %s", str(err), line)

        return data


def read_parquet(
    file_path: str,
    lazy: bool = False,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    buffer_size: int = 1024,
):
    def _yield(file_path: str):
        parquet_file = pq.ParquetFile(file_path)
        for record_batch in parquet_file.iter_batches(batch_size=buffer_size):
            yield from record_batch.to_pylist()

    if lazy:
        return _yield(file_path=file_path)
    else:
        return pd.read_parquet(file_path, engine=engine).apply(dict, axis=1).to_list()


def read_csv(
    file_path: str,
    encoding: str = "utf-8",
    lazy: bool = False,
    has_header: bool = False,
    delimiter: str = ",",
    quotechar: str = '"',
    lineterminator: str = "\r\n",
    skipinitialspace: bool = False,
):
    with open(file_path, mode="r", encoding=encoding) as fp:
        if has_header:
            reader = csv.DictReader(
                fp,
                delimiter=delimiter,
                quotechar=quotechar,
                lineterminator=lineterminator,
                skipinitialspace=skipinitialspace,
            )
        else:
            reader = csv.reader(
                fp,
                delimiter=delimiter,
                quotechar=quotechar,
                lineterminator=lineterminator,
                skipinitialspace=skipinitialspace,
            )

        if lazy:
            yield from reader
        else:
            return list(reader)


def read_huggingface(dataset_args: List[str], split: str, lazy=False):
    if lazy:
        yield from datasets.load_dataset(
            *dataset_args,
            split=split,
            streaming=True,
        )
    else:
        data = datasets.load_dataset(
            *dataset_args,
            split=split,
        )
        return data


def write_yaml(data_to_write: List[T], file_path: str, mode: str = "w", encoding: str = "utf-8"):
    file_path = os.path.expandvars(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode=mode, encoding=encoding) as fp:
        yaml.safe_dump(data_to_write, fp, sort_keys=False)


def write_json(data_to_write: List[T], file_path: str, mode: str = "w", encoding: str = "utf-8"):
    file_path = os.path.expandvars(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode=mode, encoding=encoding) as f:
        json.dump(data_to_write, f, indent=4)


def write_jsonl(
    data_to_write: List[T] | Iterator,
    file_path: str,
    mode: str = "a",
    encoding: str = "utf-8",
):
    file_path = os.path.expandvars(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode=mode, encoding=encoding) as f:
        for d in data_to_write:
            f.write(json.dumps(d) + "\n")


def write_parquet(
    data_to_write: List[T] | Iterator,
    file_path: str,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    buffer_size: int = 1024,
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if isinstance(data_to_write, list):
        pd.DataFrame(data_to_write).to_parquet(
            file_path,
            engine=engine,
            append=os.path.isfile(file_path),
        )
    else:
        writer = None

        def _write_batch(batch: List[T]):
            nonlocal writer
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(file_path, schema=table.schema)
            writer.write_table(table)

        batch = []
        for item in data_to_write:
            batch.append(item)
            if len(batch) >= buffer_size:
                _write_batch(batch)
                batch = []

        if batch:
            _write_batch(batch)
            batch = []

        if writer:
            writer.close()


# ===========================================================================
#                       ROTATING JSONL WRITER
# ===========================================================================
class RotatingJsonlWriter:
    """Append-only JSONL writer with size-based rotation and age-based retention.

    Rotation: when the active file exceeds ``max_bytes``, it is renamed to
    ``{stem}.1.jsonl``, existing numbered files are shifted up by one, and a
    fresh ``{stem}.jsonl`` is opened.

    Retention: on each rotation and at construction time, any rotated file
    whose mtime is older than ``max_age_days`` is deleted.  The active file is
    never deleted by retention.

    Thread safety: all writes and rotations are serialized by a single lock.

    Args:
        path: Full path to the active file (e.g. ``telemetry/traces.jsonl``).
        max_bytes: Rotate when the active file reaches this size in bytes.
        max_age_days: Delete rotated files older than this many days.
            Set to 0 to disable age-based retention entirely.

    Example::

        writer = RotatingJsonlWriter("telemetry/traces.jsonl",
                                     max_bytes=100 * 1024 * 1024,
                                     max_age_days=14)
        writer.write('{"span": "dgt.block", "duration_ms": 42}')
    """

    def __init__(self, path: str, max_bytes: int, max_age_days: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._max_age_days = max_age_days
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._purge_old_files()

    def write(self, line: str) -> None:
        """Append ``line`` (without trailing newline) to the active file,
        rotating first if the file is at or above ``max_bytes``."""
        with self._lock:
            if self._should_rotate():
                self._rotate()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _should_rotate(self) -> bool:
        try:
            return os.path.getsize(self._path) >= self._max_bytes
        except FileNotFoundError:
            return False

    def _rotate(self) -> None:
        stem, _ = os.path.splitext(self._path)
        existing = sorted(
            glob.glob(f"{stem}.*.jsonl"),
            key=lambda p: int(p[len(stem) + 1 : -len(".jsonl")]),
            reverse=True,
        )
        for fpath in existing:
            n = int(fpath[len(stem) + 1 : -len(".jsonl")])
            os.rename(fpath, f"{stem}.{n + 1}.jsonl")
        if os.path.exists(self._path):
            os.rename(self._path, f"{stem}.1.jsonl")
        self._purge_old_files()

    def _purge_old_files(self) -> None:
        if self._max_age_days <= 0:
            return
        # Standard
        from datetime import datetime, timezone

        stem, _ = os.path.splitext(self._path)
        cutoff = datetime.now(tz=timezone.utc).timestamp() - (self._max_age_days * 86400)
        for fpath in glob.glob(f"{stem}.*.jsonl"):
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
            except OSError:
                pass


# ===========================================================================
#                       BYTE CONVERTER
# ===========================================================================
def convert_byte_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# ===========================================================================
#                       DICTIONARY PROCESSOR
# ===========================================================================
# Path DSL used by ``from_dict`` / ``to_dict``:
#
#   ``a.b``           — plain key traversal.
#   ``a[0]``          — integer index into a list.
#   ``a[:n]`` / ``a[n:]`` — slice (read-only; ``to_dict`` rejects).
#   ``a[+]``          — append (``to_dict`` only; ``from_dict`` rejects).
#
# Missing-key semantics on reads: intermediate segments always raise
# ``KeyError``; the terminal segment raises when ``strict=True`` (default)
# and returns ``dataclasses.MISSING`` when ``strict=False`` so callers can
# distinguish "absent" from "present and ``None``".


def _parse_segment(segment: str) -> tuple[str, str | None]:
    """Split a path segment into ``(name, modifier)``.

    ``modifier`` is the raw content between the brackets (e.g. ``"0"``,
    ``"+"``, ``":2"``) or ``None`` when the segment has no bracket.

    Raises ``ValueError`` on malformed syntax (unclosed bracket, empty name,
    nested brackets).
    """
    if not segment:
        raise ValueError("Empty path segment")
    if "[" not in segment:
        return segment, None
    if not segment.endswith("]"):
        raise ValueError(f"Malformed segment '{segment}': unclosed bracket")
    name, _, rest = segment.partition("[")
    if not name:
        raise ValueError(f"Malformed segment '{segment}': missing key before '['")
    modifier = rest[:-1]
    if "[" in modifier or "]" in modifier:
        raise ValueError(f"Malformed segment '{segment}': nested brackets")
    return name, modifier.strip()


def _apply_read_modifier(container: Any, name: str, modifier: str) -> Any:
    """Apply a bracket modifier to a value fetched by ``name``.

    Handles ``[i]``, ``[:n]``, and ``[n:]``. Rejects ``[+]`` (write-only).
    """
    if modifier == "+":
        raise ValueError(f"Append modifier '[+]' is not valid on reads (segment '{name}[+]')")
    if not hasattr(container, "__getitem__"):
        raise TypeError(f"Expected indexable object but got {type(container)} for '{name}'")
    if modifier.startswith(":"):
        return container[: int(modifier[1:])]
    if modifier.endswith(":"):
        return container[int(modifier[:-1]) :]
    return container[int(modifier)]


def from_dict(dictionary: Dict[str, Any], key: str, *, strict: bool = True):
    """Read a value from ``dictionary`` at the DSL path ``key``.

    Args:
        dictionary: The source dictionary to read from.
        key: Dot/bracket path (see module-level DSL reference above).
        strict: When ``True`` (default), raises ``KeyError`` if the terminal
            segment is absent. When ``False``, returns ``dataclasses.MISSING``
            for an absent terminal so callers can distinguish "missing" from
            a present-but-``None`` value. Intermediate segments always raise
            regardless of ``strict``.

    Returns:
        The value at the path, or ``dataclasses.MISSING`` when the terminal
        is absent and ``strict=False``.

    Raises:
        KeyError: Intermediate segment missing, or terminal missing with
            ``strict=True``.
        TypeError: Path traverses a non-indexable value (e.g. ``.key`` on a
            list, ``[i]`` on a non-indexable).
        ValueError: Malformed path syntax.
    """
    key_segments = key.split(".")
    name, modifier = _parse_segment(key_segments[0])
    is_terminal = len(key_segments) == 1

    # Presence check at this level (distinguishes "absent" from "None value").
    if name not in dictionary:
        if is_terminal and not strict:
            return MISSING
        raise KeyError(
            f"Missing key '{name}' while resolving path '{key}'"
            + ("" if is_terminal else " (intermediate segment)")
        )

    current = dictionary[name]

    if is_terminal:
        if modifier is None:
            return current
        return _apply_read_modifier(current, name, modifier)

    # Intermediate segment: descend.
    if modifier is not None:
        if ":" in modifier:
            raise ValueError(
                f"Slice notation '[{modifier}]' is not allowed on intermediate segment '{name}'"
            )
        if modifier == "+":
            raise ValueError(f"Append modifier '[+]' is not valid on reads (segment '{name}[+]')")
        if not isinstance(current, list):
            raise TypeError(f"Expected list at '{name}' but got {type(current).__name__}")
        current = current[int(modifier)]

    # ``None``-as-absent at intermediate level: a dict with ``None`` value is
    # semantically equivalent to "no child here" for path traversal. Raise a
    # clean KeyError rather than letting the recursive call crash on ``NoneType``.
    if current is None:
        raise KeyError(f"Intermediate value at '{name}' is None while resolving path '{key}'")
    if not isinstance(current, dict):
        raise TypeError(f"Expected dict at intermediate '{name}' but got {type(current).__name__}")

    return from_dict(current, ".".join(key_segments[1:]), strict=strict)


def to_dict(dictionary: Dict[str, Any], key: str, value: Any) -> None:
    """Write ``value`` into ``dictionary`` at the DSL path ``key``.

    Intermediate containers are auto-created when missing: plain segments
    create ``dict``s, bracketed segments create ``list``s. An intermediate
    value of ``None`` is treated as absent (a common dataclass default),
    and the appropriate container is materialized in its place.

    Args:
        dictionary: The dictionary to mutate in place.
        key: Dot/bracket path (see module-level DSL reference above).
        value: The value to write at the terminal segment.

    Raises:
        TypeError: Traversal encounters a non-container at an intermediate
            segment, or a bracket modifier targets a non-list.
        IndexError: List index is out of range for an existing list.
        ValueError: Malformed path, or slice notation used (write-only DSL
            rejects ``[:n]`` / ``[n:]``).
    """
    key_segments = key.split(".")
    name, modifier = _parse_segment(key_segments[0])
    is_terminal = len(key_segments) == 1

    if modifier is None:
        # Plain-name segment.
        if is_terminal:
            dictionary[name] = value
            return

        # Intermediate: auto-create dict if missing or None.
        if dictionary.get(name) is None:
            dictionary[name] = {}
        elif not isinstance(dictionary[name], dict):
            raise TypeError(
                f"Expected dict at intermediate '{name}' but got {type(dictionary[name]).__name__}"
            )
        to_dict(dictionary[name], ".".join(key_segments[1:]), value)
        return

    # Bracketed segment.
    if ":" in modifier:
        raise ValueError(
            f"Slice notation '[{modifier}]' is not allowed in writes (segment '{name}')"
        )

    existing = dictionary.get(name)
    absent = existing is None

    if modifier == "+":
        # Append: always produces a new slot.
        if absent:
            dictionary[name] = []
            existing = dictionary[name]
        elif not isinstance(existing, list):
            raise TypeError(
                f"Expected list for '{name}' to use '[+]' append, got {type(existing).__name__}"
            )
        if is_terminal:
            existing.append(value)
            return
        # Intermediate [+]: append a fresh {} and descend into it.
        existing.append({})
        to_dict(existing[-1], ".".join(key_segments[1:]), value)
        return

    # Numeric index.
    pos_idx = int(modifier)
    if absent:
        # Preserve legacy quirk (documented): terminal ``[i]`` on a missing
        # key creates ``[None]`` and writes at position 0; intermediate
        # ``[i]`` creates ``[{}]`` and descends.
        placeholder = None if is_terminal else {}
        dictionary[name] = [placeholder]
        existing = dictionary[name]
        pos_idx = 0
    else:
        if not isinstance(existing, list):
            raise TypeError(f"Expected list for '{name}' but got {type(existing).__name__}")
        if len(existing) <= pos_idx:
            raise IndexError(
                f"Cannot assign at index {pos_idx} for list '{name}' of length {len(existing)}"
            )

    if is_terminal:
        existing[pos_idx] = value
        return
    if existing[pos_idx] is None:
        existing[pos_idx] = {}
    elif not isinstance(existing[pos_idx], dict):
        raise TypeError(
            f"Expected dict at '{name}[{pos_idx}]' but got {type(existing[pos_idx]).__name__}"
        )
    to_dict(existing[pos_idx], ".".join(key_segments[1:]), value)


# ===========================================================================
#                       DATACLASS PROCESSOR
# ===========================================================================
# ``from_dataclass`` / ``to_dataclass`` walk heterogeneous object graphs
# (dataclass → dict → list → ...) using the same DSL as ``from_dict`` /
# ``to_dict``. Attribute creation on dataclasses is intentionally disallowed:
# the whole point of declaring fields is that typos are bugs, not silent
# attribute additions. Nested dicts / lists that live inside dataclass fields
# still auto-create per the dict rules.


def _declared_fields(obj: Any) -> set[str]:
    """Return the set of declared field names on a dataclass instance."""
    return {f.name for f in _dc_fields(obj)}


def _get_node_child(node: Any, name: str, modifier: str | None, path: str, segment: str) -> Any:
    """Fetch ``node[name]`` (dataclass/dict) then apply ``modifier`` if present.

    Shared read helper for ``from_dataclass``. Raises consistent errors naming
    the failing segment and the full path.
    """
    if is_dataclass(node):
        if name not in _declared_fields(node):
            raise AttributeError(
                f"Dataclass {type(node).__name__} has no field '{name}' "
                f"(segment '{segment}' of path '{path}')"
            )
        current = getattr(node, name)
    elif isinstance(node, dict):
        if name not in node:
            raise KeyError(f"Missing key '{name}' while resolving path '{path}'")
        current = node[name]
    else:
        raise TypeError(
            f"Cannot traverse '{name}' on {type(node).__name__} "
            f"(segment '{segment}' of path '{path}')"
        )

    if modifier is None:
        return current
    if modifier == "+":
        raise ValueError(f"Append modifier '[+]' is not valid on reads (segment '{segment}')")
    if not isinstance(current, list):
        raise TypeError(
            f"Expected list at '{name}' but got {type(current).__name__} "
            f"(segment '{segment}' of path '{path}')"
        )
    if modifier.startswith(":"):
        return current[: int(modifier[1:])]
    if modifier.endswith(":"):
        return current[int(modifier[:-1]) :]
    return current[int(modifier)]


def from_dataclass(obj: Any, path: str, *, strict: bool = True):
    """Read a value from a dataclass (or mixed dataclass/dict/list graph) by path.

    Args:
        obj: The root object. Typically a dataclass instance; dicts and lists
            encountered as intermediate nodes are walked using the same DSL
            semantics as :func:`from_dict`.
        path: Dot/bracket path.
        strict: When ``True`` (default), a missing terminal raises. When
            ``False``, returns ``dataclasses.MISSING``. Intermediate segments
            always raise regardless of ``strict``. Dataclass field names that
            are not declared always raise ``AttributeError`` — "not declared"
            is a schema error, not a missing value.

    Returns:
        The value at the path, or ``dataclasses.MISSING`` when the terminal
        is absent on a dict node and ``strict=False``.

    Raises:
        AttributeError: Dataclass field not declared at any segment.
        KeyError: Dict intermediate key missing, or terminal missing when strict.
        TypeError: Path traverses a non-traversable value.
        ValueError: Malformed path syntax.
    """
    segments = path.split(".")
    name, modifier = _parse_segment(segments[0])
    is_terminal = len(segments) == 1

    # Dataclass field not declared → always raise, even when not strict.
    if is_dataclass(obj) and name not in _declared_fields(obj):
        raise AttributeError(
            f"Dataclass {type(obj).__name__} has no field '{name}' "
            f"(segment '{segments[0]}' of path '{path}')"
        )

    # Dict terminal miss with strict=False → sentinel.
    if isinstance(obj, dict) and is_terminal and name not in obj and not strict:
        return MISSING

    current = _get_node_child(obj, name, modifier, path, segments[0])

    if is_terminal:
        return current

    if current is None:
        raise KeyError(f"Intermediate value at '{name}' is None while resolving path '{path}'")
    return from_dataclass(current, ".".join(segments[1:]), strict=strict)


def _descend_for_write(node: Any, name: str, modifier: str | None, path: str, segment: str) -> Any:
    """Return the child container at ``node[name]`` (+ modifier), auto-creating
    intermediate ``None`` dataclass fields and dict entries per design rules.

    Used by ``to_dataclass``. The caller decides what to do at the terminal;
    this helper only handles intermediate descent.
    """
    if is_dataclass(node):
        if name not in _declared_fields(node):
            raise ValueError(
                f"Dataclass {type(node).__name__} has no field '{name}' "
                f"(segment '{segment}' of path '{path}'). "
                f"Attributes cannot be auto-created on dataclasses."
            )
        current = getattr(node, name)
        # ``None``-valued dataclass field behaves as "absent": materialize the
        # correct container based on the modifier shape.
        if current is None:
            current = [] if modifier is not None else {}
            setattr(node, name, current)
    elif isinstance(node, dict):
        current = node.get(name)
        if current is None:
            current = [] if modifier is not None else {}
            node[name] = current
    else:
        raise TypeError(
            f"Cannot descend into '{name}' on {type(node).__name__} "
            f"(segment '{segment}' of path '{path}')"
        )

    if modifier is None:
        if not isinstance(current, dict) and not is_dataclass(current):
            raise TypeError(
                f"Expected dict or dataclass at intermediate '{name}' "
                f"but got {type(current).__name__}"
            )
        return current

    # Bracketed modifier: descend into list.
    if ":" in modifier:
        raise ValueError(
            f"Slice notation '[{modifier}]' is not allowed in writes (segment '{segment}')"
        )
    if not isinstance(current, list):
        raise TypeError(f"Expected list for '{name}' but got {type(current).__name__}")
    if modifier == "+":
        current.append({})
        return current[-1]
    pos_idx = int(modifier)
    if len(current) <= pos_idx:
        raise IndexError(
            f"Cannot descend into index {pos_idx} for list '{name}' of length {len(current)}"
        )
    child = current[pos_idx]
    if child is None:
        child = {}
        current[pos_idx] = child
    elif not isinstance(child, (dict,)) and not is_dataclass(child):
        raise TypeError(
            f"Expected dict or dataclass at '{name}[{pos_idx}]' but got {type(child).__name__}"
        )
    return child


def to_dataclass(obj: Any, path: str, value: Any) -> None:
    """Write ``value`` into a dataclass (or mixed graph) at DSL ``path``.

    Mutates ``obj`` in place. Rules:

    - Dataclass fields must be declared. Undeclared paths raise ``ValueError``.
      This is a schema contract, not a missing-value condition.
    - Dict keys and list slots auto-create per the :func:`to_dict` rules
      (plain segment → ``{}``; bracket segment → ``[]``).
    - An intermediate dataclass field that is ``None`` is treated as absent
      and replaced with the appropriate container, keeping the common
      ``Optional[Dict] = None`` pattern writable without manual initialization.
    - ``SRC_DATA`` is reserved; writing to it at the terminal raises.
    - No leaf-type enforcement. Python dataclasses do not enforce field types
      at runtime and neither does this helper.

    Args:
        obj: The root object to mutate (dataclass, dict, or mixed graph).
        path: Dot/bracket path.
        value: The value to assign at the terminal.

    Raises:
        ValueError: Undeclared dataclass field, slice notation in path, or
            write targets ``SRC_DATA`` at the terminal.
        TypeError: Path traverses an incompatible type, or ``[+]`` targets a
            non-list field.
        IndexError: List index is out of range.
    """
    segments = path.split(".")
    name, modifier = _parse_segment(segments[0])
    is_terminal = len(segments) == 1

    if is_terminal and name == "SRC_DATA":
        raise ValueError("Cannot write to reserved field 'SRC_DATA'")

    if modifier is None:
        if is_terminal:
            if is_dataclass(obj):
                if name not in _declared_fields(obj):
                    raise ValueError(
                        f"Dataclass {type(obj).__name__} has no field '{name}' "
                        f"(path '{path}'). Attributes cannot be auto-created on dataclasses."
                    )
                setattr(obj, name, value)
            elif isinstance(obj, dict):
                obj[name] = value
            else:
                raise TypeError(f"Cannot set '{name}' on {type(obj).__name__} (path '{path}')")
            return
        child = _descend_for_write(obj, name, None, path, segments[0])
        to_dataclass(child, ".".join(segments[1:]), value)
        return

    # Bracketed segment.
    if ":" in modifier:
        raise ValueError(
            f"Slice notation '[{modifier}]' is not allowed in writes (segment '{segments[0]}')"
        )

    # Resolve the list that holds this position.
    if is_dataclass(obj):
        if name not in _declared_fields(obj):
            raise ValueError(
                f"Dataclass {type(obj).__name__} has no field '{name}' "
                f"(path '{path}'). Attributes cannot be auto-created on dataclasses."
            )
        lst = getattr(obj, name)
        if lst is None:
            lst = []
            setattr(obj, name, lst)
        elif not isinstance(lst, list):
            raise TypeError(
                f"Field '{name}' on {type(obj).__name__} is {type(lst).__name__}, "
                f"cannot use bracket modifier"
            )
    elif isinstance(obj, dict):
        lst = obj.get(name)
        if lst is None:
            lst = []
            obj[name] = lst
        elif not isinstance(lst, list):
            raise TypeError(f"Expected list for '{name}' but got {type(lst).__name__}")
    else:
        raise TypeError(f"Cannot descend into '{name}' on {type(obj).__name__} (path '{path}')")

    if modifier == "+":
        if is_terminal:
            lst.append(value)
            return
        lst.append({})
        to_dataclass(lst[-1], ".".join(segments[1:]), value)
        return

    pos_idx = int(modifier)
    if is_terminal:
        if len(lst) <= pos_idx:
            raise IndexError(
                f"Cannot assign at index {pos_idx} for list '{name}' of length {len(lst)}"
            )
        lst[pos_idx] = value
        return

    if len(lst) <= pos_idx:
        raise IndexError(
            f"Cannot descend into index {pos_idx} for list '{name}' of length {len(lst)}"
        )
    child = lst[pos_idx]
    if child is None:
        child = {}
        lst[pos_idx] = child
    elif not isinstance(child, dict) and not is_dataclass(child):
        raise TypeError(
            f"Expected dict or dataclass at '{name}[{pos_idx}]' but got {type(child).__name__}"
        )
    to_dataclass(child, ".".join(segments[1:]), value)

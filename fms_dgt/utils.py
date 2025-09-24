# Standard
from collections import ChainMap
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
import fnmatch
import glob
import importlib.util
import json
import logging
import math
import os
import signal
import socket

# Third Party
import datasets
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Local
from fms_dgt.constants import NAME_KEY

# ===========================================================================
#                       LOGGER CONFIGURATION
# ===========================================================================


# Step 1: Create default log formatter
DGT_LOG_FORMATTER = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)

# Step 2: Initialize logger
dgt_logger = logging.getLogger("fms_dgt")

# Step 3: Set up logging level
dgt_logger.setLevel(level=getattr(logging, os.getenv("LOG_LEVEL", "info").upper()))

# Step 4: Create and add stream handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(DGT_LOG_FORMATTER)
dgt_logger.propagate = False
dgt_logger.addHandler(_stream_handler)


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


def group_data_by_attribute(data_list: List[T], attr: str) -> List[List[T]]:
    attr_values = set([getattr(data_item, attr) for data_item in data_list])
    return [
        [data_item for data_item in data_list if getattr(data_item, attr) == attr_value]
        for attr_value in attr_values
    ]


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
    with open(file_path, mode="r", encoding=encoding) as fp:
        data = yaml.safe_load(fp)
    return data


def read_json(file_path: str, encoding: str = "utf-8"):
    with open(file_path, mode="r", encoding=encoding) as fp:
        try:
            data = json.load(fp)
        except ValueError:
            data = []
    return data


def read_jsonl(file_path: str, encoding: str = "utf-8", lazy: bool = False):
    def _yield(file_path: str, encoding: str = "utf-8"):
        with open(file_path, mode="r", encoding=encoding) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as err:
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode=mode, encoding=encoding) as fp:
        yaml.safe_dump(data_to_write, fp, sort_keys=False)


def write_json(data_to_write: List[T], file_path: str, mode: str = "w", encoding: str = "utf-8"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode=mode, encoding=encoding) as f:
        json.dump(data_to_write, f, indent=4)


def write_jsonl(
    data_to_write: List[T] | Iterator,
    file_path: str,
    mode: str = "a",
    encoding: str = "utf-8",
):
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
def from_dict(dictionary: Dict[str, Any], key: str):
    """
    Fetching value for key from the dictionary.

    NOTE:
    - Nested key are allowed. Support for "." notation, "[:?\\d+:?]" notation

    Args:
        dictionary (Dict[str, Any]): _description_
        key (str): _description_

    Returns:
        _type_: _description_
    """
    # Step 1: Split into individual key segments
    key_segments = key.split(".")

    # Step 2: Get value, if last key segment
    if len(key_segments) == 1:
        # Step 2.a: key with list notation
        if key_segments[0].endswith("]"):
            dict_key, pos_idx = key_segments[0].split("[")

            if not hasattr(dictionary.get(dict_key), "__getitem__"):
                raise TypeError(
                    f"Expected indexable object but got {type(dictionary.get(dict_key))} for {dict_key}"
                )

            pos_idx = pos_idx.rstrip("]").strip()
            if pos_idx.startswith(":"):
                return dictionary.get(dict_key)[: int(pos_idx[1:])]
            elif pos_idx.endswith(":"):
                return dictionary.get(dict_key)[: int(pos_idx[:-1])]
            else:
                return dictionary.get(dict_key)[int(pos_idx)]
        else:
            # Step 2.b: dictionary key
            return dictionary.get(key_segments[0])
    else:
        # Step 3.a: key with list notation
        if key_segments[0].endswith("]"):
            dict_key, pos_idx = key_segments[0].split("[")

            if ":" in pos_idx.rstrip("]").strip():
                raise ValueError("List notation ([:n], [n:]) is not allowed for intermediate keys.")

            if not isinstance(dictionary.get(dict_key), list):
                raise TypeError(
                    f"Expected list but got {type(dictionary.get(dict_key))} for {dict_key}"
                )

            return from_dict(
                dictionary=dictionary.get(dict_key)[int(pos_idx.rstrip("]").strip())],
                key=".".join(key_segments[1:]),
            )
        else:
            # Step 3.b: dictionary key
            return from_dict(
                dictionary=dictionary.get(key_segments[0]),
                key=".".join(key_segments[1:]),
            )


def to_dict(dictionary: Dict[str, Any], key: str, value: Any):
    # Step 1: Split into individual key segments
    key_segments = key.split(".")

    # Step 2: Set value, if last key segment
    if len(key_segments) == 1:
        # Step 2.a: key with list notation
        if key_segments[0].endswith("]"):
            # Step 2.a.i: Identify dictionary key and position index
            dict_key, pos_idx = key_segments[0].split("[")

            # Step 2.a.ii: Create empty list and set position index to 0, if necessary
            if dict_key not in dictionary:
                dictionary[dict_key] = [None]
                pos_idx = 0
            else:
                # Step 2.a.ii.*: Raise error, if position index refers to multiple multiple indices
                if ":" in pos_idx:
                    raise ValueError("List notation ([:n], [n:]) is not allowed for  keys.")
                else:
                    # Step 2.a.ii.**: Parse position index
                    pos_idx = int(pos_idx.rstrip("]").strip())

                    # Step 2.a.ii.***: Raise error, if value for the key of interest is not of type list
                    if not isinstance(dictionary[dict_key], list):
                        raise TypeError(
                            f"Expected list for {dict_key} key but got {type(dictionary[dict_key])} type."
                        )

                    # Step 2.a.ii.****: Raise error, if position index is out of index range
                    if len(dictionary[dict_key]) < pos_idx:
                        raise IndexError(
                            f"Cannot insert at index {pos_idx} for list with lengh = {len(dictionary[dict_key])}"
                        )

            # Step 2.a.iii: Set value
            dictionary[dict_key][pos_idx] = value
        else:
            dictionary[key_segments[0]] = value
    else:
        # Step 2.a: key with list notation
        if key_segments[0].endswith("]"):
            # Step 2.a.i: Identify dictionary key and position index
            dict_key, pos_idx = key_segments[0].split("[")

            # Step 2.a.ii: Create empty list and set position index to 0, if necessary
            if dict_key not in dictionary:
                dictionary[dict_key] = [{}]
                pos_idx = 0
            else:
                # Step 2.a.ii.*: Raise error, if position index refers to multiple multiple indices
                if ":" in pos_idx:
                    raise ValueError("List notation ([:n], [n:]) is not allowed for  keys.")
                else:
                    # Step 2.a.ii.**: Parse position index
                    pos_idx = int(pos_idx.rstrip("]").strip())

                    # Step 2.a.ii.***: Raise error, if existing value for the key of interest is not of type list
                    if not isinstance(dictionary[dict_key], list):
                        raise TypeError(
                            f"Expected list for {dict_key} key but got {type(dictionary[dict_key])} type."
                        )

                    # Step 2.a.ii.****: Raise error, if position index is out of index range
                    if len(dictionary[dict_key]) < pos_idx:
                        raise IndexError(
                            f"Cannot insert at index {pos_idx} for list with lengh = {len(dictionary[dict_key])}"
                        )

            # Step 2.a.iii: Continue recursive call
            to_dict(
                dictionary=dictionary[dict_key][pos_idx],
                key=".".join(key_segments[1:]),
                value=value,
            )

        else:
            # Step 2.b: Create empty dictionary with key_segment as the key
            if key_segments[0] not in dictionary:
                dictionary[key_segments[0]] = {}

            # Step 2.c: Continue
            to_dict(
                dictionary=dictionary[key_segments[0]],
                key=".".join(key_segments[1:]),
                value=value,
            )

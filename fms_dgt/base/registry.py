# Standard
from typing import Any
import os
import re

# Local
from fms_dgt import INTERNAL_DGT_DIR, SRC_DGT_DIR
from fms_dgt.base.dataloader import Dataloader
from fms_dgt.base.datastore import Datastore
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.resource import BaseResource
from fms_dgt.utils import dynamic_import

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
REGISTRATION_SEARCHABLE_DIRECTORIES = []
_ADDED_REGISTRATION_DIRECTORIES = set()
REGISTRATION_MODULE_MAP = {}


# ===========================================================================
#                       HELPER FUNCTIONS
# ===========================================================================
def add_namespace_to_searchable_dirs(namespace: str):
    """Adds a provided namespace to the searchable locations the registry will access

    Args:
        namespace (str): Namespace to extend locations accessible to registry
    """
    namespace_dir = os.path.join(SRC_DGT_DIR, namespace)
    if not os.path.isdir(namespace_dir):
        raise ValueError(f"Namespace '{namespace}' must be a subdirectory of fms_dgt")

    searchable = (INTERNAL_DGT_DIR, namespace_dir)
    if searchable not in REGISTRATION_SEARCHABLE_DIRECTORIES:
        REGISTRATION_SEARCHABLE_DIRECTORIES.append(searchable)


def _build_importable_registration_map(registration_func: str):
    def extract_registered_classes(file_contents: str):
        classes = []
        for matching_pattern in re.findall(
            rf"{registration_func}\(.*\)", file_contents
        ):
            # last character is ")"
            matching_pattern = matching_pattern.replace(registration_func + "(", "")[
                :-1
            ]
            classes.extend(
                [
                    pattern.replace('"', "").strip()
                    for pattern in matching_pattern.split(",")
                ]
            )
        return classes

    if registration_func not in REGISTRATION_MODULE_MAP:
        REGISTRATION_MODULE_MAP[registration_func] = dict()

    for base_dir, search_dir in REGISTRATION_SEARCHABLE_DIRECTORIES:
        if (search_dir, registration_func) in _ADDED_REGISTRATION_DIRECTORIES:
            continue
        _ADDED_REGISTRATION_DIRECTORIES.add((search_dir, registration_func))
        for dirpath, _, filenames in os.walk(os.path.join(base_dir, search_dir)):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if filepath.endswith(".py"):
                    import_path = filepath.replace(base_dir, "").replace(os.sep, ".")[
                        :-3
                    ]
                    if import_path.startswith("."):
                        import_path = import_path[1:]
                    with open(filepath, mode="r", encoding="utf-8") as f:
                        class_names = extract_registered_classes(f.read())
                        for class_name in class_names:
                            # we have this be a list to allow conflicts to naturally occur when duplicate names are detected
                            if (
                                not class_name
                                in REGISTRATION_MODULE_MAP[registration_func]
                            ):
                                REGISTRATION_MODULE_MAP[registration_func][
                                    class_name
                                ] = []
                            REGISTRATION_MODULE_MAP[registration_func][
                                class_name
                            ].append(import_path)


def dynamic_registration_import(registration_func: str, class_name: str):
    _build_importable_registration_map(registration_func)
    if (
        registration_func in REGISTRATION_MODULE_MAP
        and class_name in REGISTRATION_MODULE_MAP[registration_func]
    ):
        import_paths = REGISTRATION_MODULE_MAP[registration_func][class_name]
        for import_path in import_paths:
            dynamic_import(import_path, throw_top_level_error=True)


# ===========================================================================
#                       BLOCK REGISTRY
# ===========================================================================
BLOCK_REGISTRY = {}


def register_block(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert (
                name not in BLOCK_REGISTRY
            ), f"Block named '{name}' conflicts with existing block! Please register with a non-conflicting alias instead."

            BLOCK_REGISTRY[name] = cls
        return cls

    return decorate


def get_block_class(block_name):
    if block_name not in BLOCK_REGISTRY:
        dynamic_registration_import("register_block", block_name)

    known_keys = list(BLOCK_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_block", [])
    )
    if block_name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load block '{block_name}', but no block for this name found! Supported block names: {known_keys}"
        )

    return BLOCK_REGISTRY[block_name]


def get_block(block_name, *args: Any, **kwargs: Any):

    # Local
    from fms_dgt.base.multiprocessing import RayBlock
    from fms_dgt.constants import RAY_CONFIG_KEY

    block_class = get_block_class(block_name)

    ret_block = (
        RayBlock(block_class, *args, **kwargs)
        if RAY_CONFIG_KEY in kwargs
        else block_class(*args, **kwargs)
    )

    return ret_block


# ===========================================================================
#                       RESOURCE REGISTRY
# ===========================================================================
RESOURCE_REGISTRY = {}
RESOURCE_OBJECTS = {}


def register_resource(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseResource
            ), f"Resource '{name}' ({cls.__name__}) must extend BaseResource class"

            assert (
                name not in RESOURCE_REGISTRY
            ), f"Resource named '{name}' conflicts with existing resource! Please register with a non-conflicting alias instead."

            RESOURCE_REGISTRY[name] = cls
        return cls

    return decorate


def get_resource(resource_name, *args: Any, **kwargs: Any):
    if resource_name not in RESOURCE_REGISTRY:
        dynamic_registration_import("register_resource", resource_name)

    known_keys = list(RESOURCE_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_resource", [])
    )

    if resource_name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load resource '{resource_name}', but no resource for this name found! Supported resource names: {known_keys}"
        )

    resource: BaseResource = RESOURCE_REGISTRY[resource_name](*args, **kwargs)

    if resource.id not in RESOURCE_OBJECTS:
        RESOURCE_OBJECTS[resource.id] = resource
    return RESOURCE_OBJECTS[resource.id]


# ===========================================================================
#                       DATABUILDER REGISTRY
# ===========================================================================
DATABUILDER_REGISTRY = {}
ALL_DATABUILDERS = set()


def register_data_builder(name):
    def decorate(fn):
        assert (
            name not in DATABUILDER_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        DATABUILDER_REGISTRY[name] = fn
        ALL_DATABUILDERS.add(name)
        return fn

    return decorate


def get_data_builder(name, *args: Any, **kwargs: Any):
    if name not in DATABUILDER_REGISTRY:
        raise KeyError(
            f"Attempted to load data builder '{name}', but no data builder for this name found! Supported data builder names: {', '.join(DATABUILDER_REGISTRY.keys())}"
        )
    return DATABUILDER_REGISTRY[name](*args, **kwargs)


# ===========================================================================
#                       DATALOADER REGISTRY
# ===========================================================================
DATALOADER_REGISTRY = {}


def register_dataloader(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, Dataloader
            ), f"Dataloader '{name}' ({cls.__name__}) must extend BaseDataloader class"

            assert (
                name not in DATALOADER_REGISTRY
            ), f"Dataloader named '{name}' conflicts with existing dataloader! Please register with a non-conflicting alias instead."

            DATALOADER_REGISTRY[name] = cls
        return cls

    return decorate


def get_dataloader(dataloader_name, *args: Any, **kwargs: Any) -> Dataloader:
    if dataloader_name not in DATALOADER_REGISTRY:
        dynamic_registration_import("register_dataloader", dataloader_name)

    known_keys = list(DATALOADER_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_dataloader", [])
    )
    if dataloader_name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load dataloader '{dataloader_name}', but no dataloader for this name found! Supported dataloader names: {known_keys}"
        )

    return DATALOADER_REGISTRY[dataloader_name](*args, **kwargs)


# ===========================================================================
#                       DATASTORE REGISTRY
# ===========================================================================
DATASTORE_REGISTRY = {}


def register_datastore(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, Datastore
            ), f"Datastore '{name}' ({cls.__name__}) must extend Datastore class"

            assert (
                name not in DATASTORE_REGISTRY
            ), f"Datastore named '{name}' conflicts with existing datastore! Please register with a non-conflicting alias instead."

            DATASTORE_REGISTRY[name] = cls
        return cls

    return decorate


def get_datastore(datastore_name, *args: Any, **kwargs: Any) -> Datastore:
    if datastore_name not in DATASTORE_REGISTRY:
        dynamic_registration_import("register_datastore", datastore_name)

    known_keys = list(DATASTORE_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_datastore", [])
    )
    if datastore_name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load datastore '{datastore_name}', but no datastore for this name found! Supported datastore names: {known_keys}"
        )

    return DATASTORE_REGISTRY[datastore_name](*args, **kwargs)


# ===========================================================================
#                       FORMATTER REGISTRY
# ===========================================================================
FORMATTER_REGISTRY = {}


def register_formatter(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            if not issubclass(cls, Formatter):
                raise TypeError(
                    f"Formatter '{name}' ({cls.__name__}) must extend BaseFormatter class"
                )

            if name in FORMATTER_REGISTRY:
                raise ValueError(
                    f"Formatter named '{name}' conflicts with existing formatter! Please register with a non-conflicting alias instead."
                )

            FORMATTER_REGISTRY[name] = cls
        return cls

    return decorate


def get_formatter(formatter_name, *args: Any, **kwargs: Any) -> Dataloader:
    if formatter_name not in FORMATTER_REGISTRY:
        dynamic_registration_import("register_formatter", formatter_name)

    known_keys = list(FORMATTER_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_formatter", [])
    )
    if formatter_name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load formatter '{formatter_name}', but no formatter for this name found! Supported formattter names: {known_keys}"
        )

    return FORMATTER_REGISTRY[formatter_name](*args, **kwargs)


# ===========================================================================
#                       TASK REGISTRY
# ===========================================================================
TASK_REGISTRY = {}
ALL_TASKS = set()


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        return fn

    return decorate


def get_task(name, *args: Any, **kwargs: Any):
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Attempted to load task '{name}', but no task for this name found! Supported task names: {', '.join(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name](*args, **kwargs)

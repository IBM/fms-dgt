# Standard
from typing import Any

# Local
from fms_dgt.base.registry import REGISTRATION_MODULE_MAP, dynamic_registration_import

RETRIEVER_REGISTRY = {}


def register_retriever(*names):
    def decorate(cls):
        for name in names:
            assert (
                name not in RETRIEVER_REGISTRY
            ), f"unstructured_text_retriever named '{name}' conflicts with existing unstructured_text_retriever! Please register with a non-conflicting alias instead."

            RETRIEVER_REGISTRY[name] = cls
        return cls

    return decorate


def get_retriever_class(name):
    if name not in RETRIEVER_REGISTRY:
        dynamic_registration_import("register_retriever", name)

    known_keys = list(RETRIEVER_REGISTRY.keys()) + list(
        REGISTRATION_MODULE_MAP.get("register_retriever", [])
    )
    if name not in known_keys:
        known_keys = ", ".join(known_keys)
        raise KeyError(
            f"Attempted to load unstructured_text_retriever '{name}', but no block for this name found! Supported unstructured_text_retriever names: {known_keys}"
        )

    return RETRIEVER_REGISTRY[name]


def get_unstructured_text_retriever(name, *args: Any, **kwargs: Any):
    req_class = get_retriever_class(name)
    return req_class(*args, **kwargs)

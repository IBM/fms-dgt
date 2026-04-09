# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

# Local
from fms_dgt.core.tools.data_objects import ToolList

logger = logging.getLogger("fms_dgt.tools.loaders")


# ===========================================================================
#                       LOADER REGISTRY
# ===========================================================================

_TOOL_LOADER_REGISTRY: Dict[str, type] = {}


def register_tool_loader(*names: str):
    """Class decorator that registers a ToolLoader subclass under one or more names.

    Usage::

        @register_tool_loader("file")
        class FileToolLoader(ToolLoader):
            ...

    Args:
        *names: One or more string aliases.

    Raises:
        AssertionError: If a name is already registered or the class does not
            extend ``ToolLoader``.
    """

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, ToolLoader
            ), f"Loader '{name}' ({cls.__name__}) must extend ToolLoader"
            assert name not in _TOOL_LOADER_REGISTRY, (
                f"Tool loader named '{name}' conflicts with an existing registration. "
                f"Use a non-conflicting alias."
            )
            _TOOL_LOADER_REGISTRY[name] = cls
        return cls

    return decorate


def get_tool_loader(name: str, **kwargs: Any) -> "ToolLoader":
    """Instantiate a registered ToolLoader by type name.

    This is the resolver called by ``Task.__init__`` when processing the
    ``tools:`` config list.  Each entry in the list has a ``type`` key that
    maps to a registered loader class; the remaining keys are forwarded as
    constructor kwargs.

    Args:
        name: Registry key (e.g. ``"file"``).
        **kwargs: Keyword arguments forwarded to the loader constructor.

    Returns:
        An initialized ``ToolLoader`` instance.

    Raises:
        KeyError: If no loader is registered under ``name``.
    """
    if name not in _TOOL_LOADER_REGISTRY:
        known = ", ".join(_TOOL_LOADER_REGISTRY.keys()) or "<none registered>"
        raise KeyError(f"Tool loader '{name}' not found. Registered loaders: {known}")
    return _TOOL_LOADER_REGISTRY[name](**kwargs)


# ===========================================================================
#                       BASE LOADER
# ===========================================================================


class ToolLoader(ABC):
    """Abstract base class for all tool source adapters.

    A ``ToolLoader`` is responsible for one concern: producing a
    ``list[Tool]`` with ``namespace`` already set on every tool.  It knows
    nothing about the registry, validation, or deduplication — those are
    ``ToolRegistry``'s concerns.

    Loaders are retained by the registry for ``refresh()`` calls.  Stateful
    loaders (e.g. ``MCPToolLoader``) should keep their connection open between
    ``load()`` calls; stateless loaders (e.g. ``FileToolLoader``) can re-open
    on each call.
    """

    @abstractmethod
    def load(self) -> ToolList:
        """Load and return a list of ``Tool`` objects.

        Every returned ``Tool`` must have ``namespace`` set.  This method is
        called once at construction time (via ``ToolRegistry.from_loaders``)
        and again on each ``ToolRegistry.refresh()`` call.

        Returns:
            List of ``Tool`` instances, each with a non-empty ``namespace``.
        """
        raise NotImplementedError

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


# ===========================================================================
#                       SAMPLING ERROR
# ===========================================================================


class SamplingError(Exception):
    """Raised when a topology-aware sampler cannot satisfy the requested k.

    Attributes:
        requested: The k that was requested.
        tools: The partial list collected before the sampler dead-ended.
            For ``tc/chain`` this is a valid prefix chain.  For ``tc/fan_in``
            and ``tc/fan_out`` this may be a structurally incomplete set.
            May be empty.  Use at your own risk; the message states the case.
    """

    def __init__(self, message: str, requested: int, tools: list) -> None:
        super().__init__(message)
        self.requested = requested
        self.tools = tools


# ===========================================================================
#                       SAMPLER REGISTRY
# ===========================================================================

_TOOL_SAMPLER_REGISTRY: Dict[str, type] = {}


def register_tool_sampler(*names: str):
    """Class decorator that registers a ToolSampler subclass under one or more names.

    Usage::

        @register_tool_sampler("tc/random")
        class RandomToolSampler(ToolSampler):
            ...

    Args:
        *names: One or more string aliases.

    Raises:
        AssertionError: If a name is already registered or the class does not
            extend ``ToolSampler``.
    """

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, ToolSampler
            ), f"Sampler '{name}' ({cls.__name__}) must extend ToolSampler"
            assert name not in _TOOL_SAMPLER_REGISTRY, (
                f"Tool sampler named '{name}' conflicts with an existing registration. "
                f"Use a non-conflicting alias."
            )
            _TOOL_SAMPLER_REGISTRY[name] = cls
        return cls

    return decorate


def get_tool_sampler(name: str, **kwargs: Any) -> "ToolSampler":
    """Instantiate a registered ToolSampler by type name.

    Args:
        name: Registry key (e.g. ``"tc/random"``).
        **kwargs: Keyword arguments forwarded to the sampler constructor.

    Returns:
        An initialized ``ToolSampler`` instance.

    Raises:
        KeyError: If no sampler is registered under ``name``.
    """
    if name not in _TOOL_SAMPLER_REGISTRY:
        known = ", ".join(_TOOL_SAMPLER_REGISTRY.keys()) or "<none registered>"
        raise KeyError(f"Tool sampler '{name}' not found. Registered samplers: {known}")
    return _TOOL_SAMPLER_REGISTRY[name](**kwargs)


# ===========================================================================
#                       BASE CLASS
# ===========================================================================


class ToolSampler(ABC):
    """Abstract base class for all tool samplers.

    A sampler selects a subset of tools from a ``ToolRegistry`` for a single
    data generation call.  Samplers run per-scenario at generation time and
    read from ``registry.artifacts`` populated by enrichments at task
    initialization time.

    **Enrichment dependency:** ``required_artifacts`` declares artifact keys
    that must already be present in ``registry.artifacts`` before this sampler
    is constructed.  Missing artifacts raise immediately with a clear message
    pointing to the enrichment that should be added.

    **Constructor vs call-site overrides:** Samplers are configured once via
    YAML (constructor arguments) and called many times (once per scenario).
    ``sample()`` accepts keyword overrides for any constructor parameter so
    that the scenario stage can adjust sampling per call without rebuilding the
    sampler.  Call-site values always win over constructor defaults.

    **Logging:** Subclasses should log via ``self.logger``.

    Class Attributes:
        required_artifacts: Artifact keys that must exist in
            ``registry.artifacts`` before construction.
    """

    required_artifacts: List[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "logger" not in cls.__dict__:
            cls.logger = logging.getLogger(cls.__module__)

    def __init__(self, registry: Any, **kwargs: Any) -> None:
        self.logger = logging.getLogger(type(self).__module__)
        self._registry = registry
        self._validate_artifacts(registry)
        super().__init__(**kwargs)

    def _validate_artifacts(self, registry: Any) -> None:
        """Raise if any required artifact is missing from registry.artifacts."""
        missing = [k for k in self.required_artifacts if k not in registry.artifacts]
        if missing:
            name = next(
                (k for k, v in _TOOL_SAMPLER_REGISTRY.items() if v is type(self)),
                type(self).__name__,
            )
            raise ValueError(
                f"Sampler '{name}' requires artifact(s) {missing} but they are not present "
                f"in registry.artifacts. Add the corresponding enrichment(s) to "
                f"tools.enrichments in your task config."
            )

    @abstractmethod
    def sample(self, k: int | None = None, **kwargs: Any) -> list:
        """Return a list of ``Tool`` objects for one data generation call.

        Args:
            k: Number of tools to sample.  When ``None``, the constructor
                default is used.  Call-site value overrides the constructor.
            **kwargs: Sampler-specific per-call overrides for constructor
                parameters.  Each subclass declares the kwargs it accepts.
                Unrecognized kwargs should be ignored silently so that callers
                can pass a uniform set of arguments across sampler types.

        Returns:
            List of selected ``Tool`` instances.  May be shorter than ``k``
            if the available tool pool is smaller.
        """
        raise NotImplementedError

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


# ===========================================================================
#                       ENRICHMENT REGISTRY
# ===========================================================================

_TOOL_ENRICHMENT_REGISTRY: Dict[str, type] = {}


def register_tool_enrichment(*names: str):
    """Class decorator that registers a ToolEnrichment subclass under one or more names.

    Usage::

        @register_tool_enrichment("output_parameters")
        class OutputParametersEnrichment(ToolEnrichment):
            ...

    Args:
        *names: One or more string aliases.

    Raises:
        AssertionError: If a name is already registered or the class does not
            extend ``ToolEnrichment``.
    """

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, ToolEnrichment
            ), f"Enrichment '{name}' ({cls.__name__}) must extend ToolEnrichment"
            assert name not in _TOOL_ENRICHMENT_REGISTRY, (
                f"Tool enrichment named '{name}' conflicts with an existing registration. "
                f"Use a non-conflicting alias."
            )
            _TOOL_ENRICHMENT_REGISTRY[name] = cls
        return cls

    return decorate


def get_tool_enrichment(name: str, **kwargs: Any) -> "ToolEnrichment":
    """Instantiate a registered ToolEnrichment by type name.

    Args:
        name: Registry key (e.g. ``"embeddings"``).
        **kwargs: Keyword arguments forwarded to the enrichment constructor.

    Returns:
        An initialized ``ToolEnrichment`` instance.

    Raises:
        KeyError: If no enrichment is registered under ``name``.
    """
    if name not in _TOOL_ENRICHMENT_REGISTRY:
        known = ", ".join(_TOOL_ENRICHMENT_REGISTRY.keys()) or "<none registered>"
        raise KeyError(f"Tool enrichment '{name}' not found. Registered enrichments: {known}")
    return _TOOL_ENRICHMENT_REGISTRY[name](**kwargs)


# ===========================================================================
#                       BASE CLASS
# ===========================================================================


class ToolEnrichment(ABC):
    """Abstract base class for all tool enrichment passes.

    An enrichment is an in-place transformation applied to a ``ToolRegistry``
    after all loaders have run.  Enrichments may:

    - Modify ``Tool`` objects in-place (e.g. fill in ``output_parameters``).
    - Write side-channel data to ``registry.artifacts`` (e.g. embeddings).

    **Dependency ordering:** ``depends_on`` declares artifact keys that must
    already be present in ``registry.artifacts`` before this enrichment runs.
    The framework resolves execution order via topological sort at task
    construction time.  A cycle or missing dependency is a hard error.

    **Execution contract:** ``enrich()`` is called once per enrichment pass
    (at task construction time and again on each ``ToolRegistry.refresh()``).
    Enrichments must be idempotent: running them a second time on an already-
    enriched registry must not corrupt state.

    **Logging:** Subclasses should log via ``self.logger`` rather than a
    module-level logger.  ``Task._post_init()`` replaces ``self.logger`` with
    a ``logging.LoggerAdapter`` carrying ``task_name`` so that enrichment-phase
    records are attributable to the task that triggered them.  See the
    three-channel attribution design in ``fms_dgt/log/context.py``.

    Class Attributes:
        depends_on: Artifact keys that must exist in ``registry.artifacts``
            before ``enrich()`` is called.
        artifact_key: Key written to ``registry.artifacts`` by this enrichment,
            or ``None`` if the enrichment only mutates ``Tool`` objects.
    """

    depends_on: List[str] = []
    artifact_key: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Set a class-level logger as a fallback so instances constructed via
        # __new__ (e.g. in tests) still have a valid logger via class lookup.
        if "logger" not in cls.__dict__:
            cls.logger = logging.getLogger(cls.__module__)

    def __init__(self, **kwargs: Any) -> None:
        # Promote to an instance attribute so Task._post_init() can replace it
        # with a LoggerAdapter carrying task_name on a per-instance basis
        # without affecting other instances of the same class.
        self.logger = logging.getLogger(type(self).__module__)
        super().__init__(**kwargs)

    @abstractmethod
    def enrich(self, registry: Any) -> None:
        """Run this enrichment pass on the given registry.

        May read ``registry.all_tools()``, read or write ``registry.artifacts``,
        and modify ``Tool`` objects in-place.  Return type is ``None`` —
        enrichments are in-place passes.

        Args:
            registry: The ``ToolRegistry`` to enrich.
        """
        raise NotImplementedError

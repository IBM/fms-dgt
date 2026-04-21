# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

# Local
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine

# ===========================================================================
#                       SAMPLER REGISTRY
# ===========================================================================

_SAMPLER_REGISTRY: Dict[str, Type["DocumentSampler"]] = {}


def register_document_sampler(*names: str):
    """Class decorator that registers a ``DocumentSampler`` subclass.

    Usage::

        @register_document_sampler("search/random")
        class RandomDocumentSampler(DocumentSampler):
            ...

    Args:
        *names: One or more string aliases for the sampler class.

    Raises:
        AssertionError: If a name is already registered or the class does not
            extend ``DocumentSampler``.
    """

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, DocumentSampler
            ), f"Sampler '{name}' ({cls.__name__}) must extend DocumentSampler"
            assert (
                name not in _SAMPLER_REGISTRY
            ), f"Document sampler '{name}' conflicts with an existing registration."
            _SAMPLER_REGISTRY[name] = cls
        return cls

    return decorate


def get_document_sampler(name: str, *args: Any, **kwargs: Any) -> "DocumentSampler":
    """Instantiate a registered ``DocumentSampler`` by name.

    Args:
        name: Registry key (e.g. ``"search/random"``).
        *args: Positional arguments forwarded to the sampler constructor.
        **kwargs: Keyword arguments forwarded to the sampler constructor.

    Raises:
        KeyError: If no sampler is registered under ``name``.
    """
    if name not in _SAMPLER_REGISTRY:
        known = ", ".join(_SAMPLER_REGISTRY.keys())
        raise KeyError(f"Document sampler '{name}' not found. Registered samplers: {known}")
    return _SAMPLER_REGISTRY[name](*args, **kwargs)


# ===========================================================================
#                       DOCUMENT SAMPLER ABC
# ===========================================================================


class DocumentSampler(ABC):
    """Base class for document sampling strategies.

    A ``DocumentSampler`` drives a ``SearchToolEngine`` to select a set of
    documents for a conversation scenario (Pattern 1 — static context).  The
    sampler owns the selection strategy; the engine owns corpus access.

    Samplers are constructed once per task (at stage initialization, before the
    thread pool starts).  ``sample()`` is called per scenario and must be
    thread-safe: no instance variable may be written during ``sample()``.

    Samplers always call ``engine.simulate()``, not ``engine.execute()`` —
    retrieval is read-only and no session history needs to be retained.

    Args:
        engine: The ``SearchToolEngine`` this sampler drives.
    """

    def __init__(self, engine: SearchToolEngine, **kwargs: Any) -> None:
        self._engine = engine

    @property
    def engine(self) -> SearchToolEngine:
        return self._engine

    @abstractmethod
    def sample(
        self,
        session_id: str,
        k: int,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Select ``k`` documents from the corpus.

        Args:
            session_id: Active session ID — passed to ``engine.simulate()``.
            k: Number of documents to return.
            query: Optional seed query or scenario description.  Samplers that
                use query-based retrieval (``search/topk``, ``search/diverse``)
                use this as the retrieval query.  Samplers that do not
                (``search/random``) ignore it.

        Returns:
            List of ``Document`` objects, length <= ``k``.
        """
        raise NotImplementedError

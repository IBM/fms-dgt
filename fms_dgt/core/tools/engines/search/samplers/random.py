# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Literal, Optional
import random

# Local
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.engines.search.samplers.base import (
    DocumentSampler,
    register_document_sampler,
)
from fms_dgt.utils import from_dict
from fms_dgt.utils import group_by as group_by_util


def _get_group_key(doc: Document, field: str) -> str:
    """Extract a grouping key from a document using dot-notation path.

    Resolved against ``Document.metadata`` first, then top-level fields
    (``id``, ``text``).  Returns ``""`` for documents missing the field so
    they are bucketed together rather than dropped.
    """
    try:
        val = from_dict(doc.metadata, field)
        if val is None:
            val = from_dict({"id": doc.id, "text": doc.text}, field)
        return str(val) if val is not None else ""
    except Exception:
        return ""


def _build_group_pools(
    documents: List[Document],
    field: str,
) -> Dict[str, List[Document]]:
    return group_by_util(documents, key=lambda doc: _get_group_key(doc, field))


def _group_probabilities(
    pools: Dict[str, List[Document]],
    strategy: Literal["uniform", "proportional"],
) -> Dict[str, float]:
    if strategy == "proportional":
        total = sum(len(p) for p in pools.values())
        return {g: len(p) / total for g, p in pools.items()}
    n = len(pools)
    return {g: 1.0 / n for g in pools}


def _resolve_eligible_pools(
    group_pools: Dict[str, List[Document]],
    group_probs: Dict[str, float],
    exclude_groups: Optional[List[str]],
    include_groups: Optional[List[str]],
) -> tuple[Dict[str, List[Document]], Dict[str, float]]:
    """Return (pools, probs) restricted to the eligible group set.

    ``include_groups`` and ``exclude_groups`` are mutually exclusive.

    Raises:
        ValueError: If both are provided, or if the eligible set is empty.
    """
    if include_groups is not None and exclude_groups is not None:
        raise ValueError("search/random: include_groups and exclude_groups are mutually exclusive.")

    if include_groups is not None:
        unknown = set(include_groups) - set(group_pools)
        if unknown:
            raise ValueError(
                f"search/random: include_groups references unknown group(s) {sorted(unknown)}. "
                f"Available: {sorted(group_pools)}"
            )
        eligible = {g: group_pools[g] for g in include_groups if g in group_pools}
    elif exclude_groups is not None:
        eligible = {g: p for g, p in group_pools.items() if g not in exclude_groups}
    else:
        return group_pools, group_probs

    if not eligible:
        raise ValueError(
            f"search/random: no groups remain after applying include/exclude filter. "
            f"Available groups: {sorted(group_pools)}"
        )

    total_w = sum(group_probs[g] for g in eligible)
    renorm_probs = {g: group_probs[g] / total_w for g in eligible}
    return eligible, renorm_probs


@register_document_sampler("search/random")
class RandomDocumentSampler(DocumentSampler):
    """Selects documents uniformly at random from the corpus.

    When ``group_by`` is omitted the sampler delegates directly to the engine,
    which selects documents randomly (baseline behavior, no corpus pre-fetch).

    When ``group_by`` is set the sampler fetches the full corpus once at
    construction time via ``engine.corpus()``, partitions it into groups by the
    value at the given dot-notation field path, then at each ``sample()`` call
    picks one group according to ``strategy`` and draws ``k`` documents from it.
    This ensures all documents in a scenario share a coherent topic domain.

    ``group_by`` is resolved against ``Document.metadata`` first, falling back
    to top-level ``Document`` fields (``id``, ``text``).  Documents that do not
    carry the field are bucketed under an empty-string group key.

    ``group_by`` requires a bounded, enumerable engine (``search/file``,
    ``search/in_memory``, ``search/elasticsearch``).  Pairing it with an
    unbounded engine such as a web-search backend raises ``NotImplementedError``
    at construction time.

    Per-call group filtering is supported via ``include_groups`` and
    ``exclude_groups`` on ``sample()``.  These are mutually exclusive and
    renormalize the group probabilities over the eligible set for that call only.

    Args:
        engine: The ``SearchToolEngine`` this sampler drives.
        group_by: Dot-notation field path used to partition the corpus into
            groups (e.g. ``"domain"`` or ``"source.category"``).  ``None``
            disables grouping.
        strategy: Group selection strategy when ``group_by`` is set.
            ``"uniform"`` (default) gives each group an equal probability.
            ``"proportional"`` weights groups by the number of documents they
            contain.
    """

    def __init__(
        self,
        engine: SearchToolEngine,
        group_by: Optional[str] = None,
        strategy: Literal["uniform", "proportional"] = "uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(engine=engine, **kwargs)
        self._group_by = group_by
        self._strategy = strategy
        self._group_pools: Optional[Dict[str, List[Document]]] = None
        self._group_probs: Optional[Dict[str, float]] = None

        if group_by is not None:
            all_docs = self.corpus()
            self._group_pools = _build_group_pools(all_docs, group_by)
            self._group_probs = _group_probabilities(self._group_pools, strategy)

    def corpus(self) -> List[Document]:
        """Return all documents from the underlying engine."""
        return self._engine.corpus()

    def sample(
        self,
        session_id: str,
        k: int,
        query: Optional[str] = None,
        exclude_groups: Optional[List[str]] = None,
        include_groups: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Select ``k`` documents, optionally restricting which groups are eligible.

        ``exclude_groups`` and ``include_groups`` are only meaningful when
        ``group_by`` was set at construction time.  Passing either without
        ``group_by`` has no effect.  The two are mutually exclusive — passing
        both raises ``ValueError``.

        Args:
            session_id: Active session ID.
            k: Number of documents to return.
            query: Ignored by this sampler.
            exclude_groups: Group keys to exclude from selection this call.
                Probabilities are renormalized over the remaining groups.
            include_groups: Restrict selection to only these group keys this
                call.  Probabilities are renormalized over the included groups.
        """
        if self._group_pools is not None:
            return self._sample_grouped(k, exclude_groups, include_groups)
        return self._sample_engine(session_id, k)

    def _sample_grouped(
        self,
        k: int,
        exclude_groups: Optional[List[str]],
        include_groups: Optional[List[str]],
    ) -> List[Document]:
        pools, probs = _resolve_eligible_pools(
            self._group_pools, self._group_probs, exclude_groups, include_groups
        )
        groups = list(probs.keys())
        weights = [probs[g] for g in groups]
        chosen_group = random.choices(groups, weights=weights, k=1)[0]
        pool = pools[chosen_group]
        return random.sample(pool, min(k, len(pool)))

    def _sample_engine(self, session_id: str, k: int) -> List[Document]:
        tc = ToolCall(name="sample", arguments={"size": k}, call_id=None)
        results = self._engine.simulate(session_id=session_id, tool_calls=[tc])
        if not results or results[0].result is None:
            return []
        raw = results[0].result
        if not isinstance(raw, list):
            return []
        return [
            Document(
                id=str(item.get("id", i)),
                text=str(item.get("text", "")),
                score=item.get("score"),
                metadata={k: v for k, v in item.items() if k not in ("id", "text", "score")},
            )
            for i, item in enumerate(raw)
        ]

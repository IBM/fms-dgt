# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Third Party
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Local
from fms_dgt.core.tools.constants import (
    TOOL_DESCRIPTION,
    TOOL_PROPERTIES,
)
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.base import ToolEnrichment, register_tool_enrichment
from fms_dgt.core.tools.enrichments.cache import (
    compute_fingerprint,
    enrichment_cache_path,
    load_cache,
    save_cache,
)
from fms_dgt.core.tools.enrichments.embeddings import (
    EmbeddingsEnrichment,
    _tool_to_text,
)
from fms_dgt.core.tools.registry import schema_fingerprint

# ===========================================================================
#                       EDGE EMBEDDING HELPERS
# ===========================================================================


def _make_param_string(name: str, info: Any) -> str:
    """Compact text for a single parameter entry."""
    descr = info.get(TOOL_DESCRIPTION, "") if isinstance(info, dict) else ""
    return f"Name: {name}\nDescription: {descr}"


def _build_edge_data(
    tools: List[Tool],
) -> Tuple[List[str], List[Tuple[str, int, List[int], List[int]]]]:
    """Build the sentence list and an index structure for edge embeddings.

    For each tool we produce three kinds of sentences:
    - The tool-level sentence (name + description).
    - One sentence per input parameter property.
    - One sentence per output parameter property.

    Returns:
        sentences: Flat list of strings ready for ``model.encode()``.
        edge_index: List of ``(qualified_name, tool_sentence_pos, [inp_positions], [out_positions])``
            tuples that map back into ``sentences``.
    """
    sentences: List[str] = []
    edge_index: List[Tuple[str, int, List[int], List[int]]] = []

    for tool in tools:
        tool_pos = len(sentences)
        sentences.append(_tool_to_text(tool))

        inp_positions: List[int] = []
        for pname, pinfo in ((tool.parameters or {}).get(TOOL_PROPERTIES) or {}).items():
            inp_positions.append(len(sentences))
            sentences.append(_make_param_string(pname, pinfo))

        out_positions: List[int] = []
        for pname, pinfo in ((tool.output_parameters or {}).get(TOOL_PROPERTIES) or {}).items():
            out_positions.append(len(sentences))
            sentences.append(_make_param_string(pname, pinfo))

        edge_index.append((tool.qualified_name, tool_pos, inp_positions, out_positions))

    return sentences, edge_index


# ===========================================================================
#                       ENRICHMENT
# ===========================================================================


@register_tool_enrichment("neighbors")
class NeighborsEnrichment(ToolEnrichment):
    """Compute output-to-input compatibility scores and store a namespace-bucketed neighbor graph.

    .. warning::
        **EXPERIMENTAL — training objective not fully established.**

        Two open issues:

        1. **Mixed signal.** Pass 1 (candidate selection) uses full-tool
           embeddings — name + description + inputs + outputs — which conflates
           domain proximity with dataflow compatibility.  Pass 2 scores edges by
           comparing composite output vectors against composite input vectors,
           which approximates dataflow but via semantic similarity rather than
           structural parameter matching.  The resulting scores are neither pure
           domain proximity nor pure dataflow compatibility.

        2. **No validated training objective for ``tc/neighbor``.** The sampler
           that consumes this artifact was intended to produce "topic-coherent"
           tool sets, but that objective is covered more cleanly by ``tc/random``
           with namespace filtering.  No training data comparison between
           ``tc/neighbor`` and ``tc/random`` has been done.

        The enrichment is retained because it is working, tested code and
        deleting it before a validated replacement (``dataflow`` enrichment,
        planned) exists carries risk.  Do not build new samplers on top of this
        artifact without first resolving the above.  See the tool subsystem
        design discussion for context.

    For each source tool A, this enrichment finds candidate target tools B
    whose input schema is semantically compatible with the output schema of A.
    The result captures "tool B can consume the output of tool A", which is
    the core signal needed by the ``tc/neighbor`` sampler.

    The artifact written to ``registry.artifacts["neighbors"]`` is a
    ``dict[qualified_name, dict[schema_fp, dict[namespace, list[tuple[name, schema_fp, score]]]]]``
    where the outer two keys uniquely identify the source overload, and each
    namespace bucket holds neighbors as ``(unqualified_name, schema_fp, score)``
    tuples sorted by cosine-similarity edge weight in descending order.  Weights
    are raw cosine scores and are globally comparable across namespaces, so
    samplers can flatten and re-rank freely.  To resolve a neighbor tuple back
    to a ``Tool``, call ``registry.get_by_fingerprint(f"{ns}::{name}", schema_fp)``.

    **Two-pass, namespace-scoped algorithm:**

    1. Candidate pass: for each source tool, collect up to
       ``min(max_candidates, namespace_size)`` closest tools per namespace
       using tool-level embeddings from ``EmbeddingsEnrichment``.
    2. Neighbor pass: score each candidate via composite edge vectors (tool
       embedding + summed parameter embeddings) and retain up to
       ``min(max_neighbors, len(candidates))`` per namespace.

    Results are cached under ``{DGT_CACHE_DIR}/enrichments/neighbors/
    {fingerprint}.json`` and delta-merged on subsequent runs.

    **Requires** ``registry.artifacts["embeddings"]`` to be populated by
    ``EmbeddingsEnrichment`` before this enrichment runs.  Pass both in the
    task YAML and the framework will order them correctly via topological sort.

    Args:
        model: Sentence-transformer model identifier.  Defaults to the same
            model used by ``EmbeddingsEnrichment``.  Only used to embed
            per-parameter sentences; the tool-level embeddings come from
            ``registry.artifacts["embeddings"]``.
        max_candidates: Maximum number of candidate tools to consider per
            namespace in the first pass (default 50).  Capped at the number
            of tools actually present in each namespace.
        max_neighbors: Maximum number of neighbors to retain per namespace
            bucket in the final artifact (default 10).  Capped at the number
            of candidates for that namespace.
        force: If ``True``, bypass the cache and recompute the neighbor graph.
    """

    depends_on: List[str] = [EmbeddingsEnrichment.artifact_key]
    artifact_key: str = "neighbors"

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        max_candidates: int = 50,
        max_neighbors: int = 10,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._max_candidates = max_candidates
        self._max_neighbors = max_neighbors
        self._force = force
        self._model = None

    def _get_model(self):
        if self._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self._model_name).to(
                device=device, dtype=torch.float32
            )
        return self._model

    def _cache_fingerprint(self, tools: List[Tool]) -> str:
        """Fingerprint: sorted (qualified_name, schema_fp, tool_text) triples + model + max_neighbors."""
        tool_ids = sorted(
            (t.qualified_name, schema_fingerprint(t.parameters), _tool_to_text(t)) for t in tools
        )
        return compute_fingerprint(tool_ids, self._model_name, self._max_neighbors)

    @staticmethod
    def _src_key(tool: Tool) -> tuple:
        """``(qualified_name, schema_fp)`` identity tuple for a source tool."""
        return (tool.qualified_name, schema_fingerprint(tool.parameters))

    def enrich(self, registry: Any) -> None:
        """Compute the neighbor graph and store in ``registry.artifacts["neighbors"]``.

        Args:
            registry: ``ToolRegistry`` instance.  Must have
                ``registry.artifacts["embeddings"]`` populated.
        """
        tools = registry.all_tools()
        if not tools:
            registry.artifacts[NeighborsEnrichment.artifact_key] = {}
            return

        # --- Cache lookup ---------------------------------------------------
        fingerprint = self._cache_fingerprint(tools)
        cache_path = enrichment_cache_path("neighbors", fingerprint)

        # Artifact shape: {qname: {schema_fp: {ns: [(name, schema_fp, score)]}}}
        # Cache stores the same shape with tuples serialized as lists.
        cached: Dict[str, Dict[str, Dict[str, List[Tuple[str, str, float]]]]] = {}
        if not self._force:
            raw = load_cache(cache_path)
            if raw:
                cached = {
                    qname: {
                        src_fp: {
                            ns: [tuple(triple) for triple in triples]
                            for ns, triples in ns_buckets.items()
                        }
                        for src_fp, ns_buckets in fp_map.items()
                    }
                    for qname, fp_map in raw.items()
                }
                self.logger.debug(
                    "NeighborsEnrichment: loaded cache from %s (%d entries)",
                    cache_path,
                    len(cached),
                )

        tools_to_compute = [
            t
            for t in tools
            if self._src_key(t) not in {(qn, fp) for qn, fp_map in cached.items() for fp in fp_map}
        ]

        if not tools_to_compute:
            self.logger.info("NeighborsEnrichment: all %d tool(s) satisfied from cache", len(tools))
            registry.artifacts[NeighborsEnrichment.artifact_key] = cached
            return

        self.logger.info(
            "NeighborsEnrichment: computing neighbor graph for %d tool(s) (%d cache hits)",
            len(tools_to_compute),
            len(tools) - len(tools_to_compute),
        )

        # Retrieve tool-level embeddings: {qname: {schema_fp: np.ndarray}}.
        tool_embeddings: Dict[str, Dict[str, np.ndarray]] = registry.artifacts[
            EmbeddingsEnrichment.artifact_key
        ]

        # Build namespace index: namespace -> list of Tool.
        ns_to_tools: Dict[str, List[Tool]] = defaultdict(list)
        for tool in tools:
            ns_to_tools[tool.namespace].append(tool)

        # Build per-parameter sentences and embed them.
        sentences, edge_index = _build_edge_data(tools)
        model = self._get_model()

        with torch.no_grad():
            all_embs = (
                model.encode(
                    sentences,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .cpu()
                .numpy()
            )

        # Tool-level similarity matrix for candidate filtering.
        # One row per tool (including overloads — each overload is a distinct row).
        tool_fp_list = [(t.qualified_name, schema_fingerprint(t.parameters)) for t in tools]
        tool_matrix = np.stack([tool_embeddings[qn][fp] for qn, fp in tool_fp_list], axis=0)
        tool_sim = tool_matrix.dot(tool_matrix.T)
        id_map = {(qn, fp): i for i, (qn, fp) in enumerate(tool_fp_list)}

        # Per-tool edge data keyed by (qualified_name, schema_fp) for fast lookup.
        edge_by_key: Dict[Tuple[str, str], Tuple[int, List[int], List[int]]] = {
            (qname, schema_fingerprint(tools[idx].parameters)): (tool_pos, inp_pos, out_pos)
            for idx, (qname, tool_pos, inp_pos, out_pos) in enumerate(edge_index)
        }

        to_compute_set = {self._src_key(t) for t in tools_to_compute}
        # new_entries mirrors cache shape but keyed by (qname, src_fp) during build.
        new_flat: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str, float]]]] = {}

        for idx, (src_qname, src_tool_pos, _src_inp_pos, src_out_pos) in enumerate(edge_index):
            src_fp = schema_fingerprint(tools[idx].parameters)
            if (src_qname, src_fp) not in to_compute_set:
                continue

            src_row = tool_sim[id_map[(src_qname, src_fp)]]

            # Source composite vector: tool embedding + summed output param embeddings.
            src_rel_emb = all_embs[src_tool_pos : src_tool_pos + 1]
            src_out_embs = [all_embs[p : p + 1] for p in src_out_pos]
            if src_out_embs:
                src_m = src_rel_emb + np.concatenate(src_out_embs, axis=0).sum(
                    axis=0, keepdims=True
                )
            else:
                src_m = src_rel_emb

            ns_buckets: Dict[str, List[Tuple[str, str, float]]] = {}

            for ns, ns_tools in ns_to_tools.items():
                # Exclude self (same qualified_name + schema_fp).
                candidates_pool = [
                    t
                    for t in ns_tools
                    if not (
                        t.qualified_name == src_qname and schema_fingerprint(t.parameters) == src_fp
                    )
                ]
                if not candidates_pool:
                    continue

                # Pass 1: top-min(max_candidates, namespace_size) by tool-level similarity.
                n_candidates = min(self._max_candidates, len(candidates_pool))
                pool_scores = [
                    (src_row[id_map[(t.qualified_name, schema_fingerprint(t.parameters))]], t)
                    for t in candidates_pool
                ]
                pool_scores.sort(key=lambda x: x[0], reverse=True)
                candidates = [t for _, t in pool_scores[:n_candidates]]

                # Pass 2: score by edge weight, retain top-min(max_neighbors, len(candidates)).
                n_neighbors = min(self._max_neighbors, len(candidates))
                edges: List[Tuple[str, str, float]] = []

                for tgt in candidates:
                    tgt_fp = schema_fingerprint(tgt.parameters)
                    tgt_tool_pos, tgt_inp_pos, _ = edge_by_key[(tgt.qualified_name, tgt_fp)]
                    tgt_rel_emb = all_embs[tgt_tool_pos : tgt_tool_pos + 1]
                    tgt_inp_embs = [all_embs[p : p + 1] for p in tgt_inp_pos]
                    if tgt_inp_embs:
                        tgt_m = tgt_rel_emb + np.concatenate(tgt_inp_embs, axis=0).sum(
                            axis=0, keepdims=True
                        )
                    else:
                        tgt_m = tgt_rel_emb

                    weight = float(np.max(cosine_similarity(src_m, tgt_m)))
                    edges.append((tgt.name, tgt_fp, round(weight, 4)))

                edges.sort(key=lambda x: x[2], reverse=True)
                ns_buckets[ns] = edges[:n_neighbors]

            new_flat[(src_qname, src_fp)] = ns_buckets

        # Fold new_flat into the two-level cache shape for persistence.
        new_entries: Dict[str, Dict[str, Dict[str, List[Tuple[str, str, float]]]]] = {}
        for (qname, src_fp), ns_buckets in new_flat.items():
            new_entries.setdefault(qname, {})[src_fp] = ns_buckets

        save_cache(cache_path, new_entries)

        # Merge cached + new at the qname level, then merge fp maps.
        merged: Dict[str, Dict[str, Dict[str, List[Tuple[str, str, float]]]]] = dict(cached)
        for qname, fp_map in new_entries.items():
            if qname in merged:
                merged[qname] = {**merged[qname], **fp_map}
            else:
                merged[qname] = fp_map

        registry.artifacts[NeighborsEnrichment.artifact_key] = merged
        self.logger.info(
            "NeighborsEnrichment: neighbor graph stored (%d source tools)", len(merged)
        )

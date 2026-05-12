# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from heapq import nlargest
from typing import Any, Dict, List, Tuple

# Third Party
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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
    ``dict[qualified_name, dict[schema_fp, dict[namespace, tuple[list, list]]]]``
    where the outer two keys uniquely identify the source overload, and each
    namespace bucket holds a tuple ``(neighbors, near_duplicates)``. Neighbors
    are ``(unqualified_name, schema_fp, score)`` tuples sorted by cosine-similarity
    edge weight in descending order. Near duplicates are tools with tool-level
    cosine similarity >= self._duplicate_sim_thresh, sorted by similarity descending, using the same tuple
    format. Weights are raw cosine scores and are globally comparable across
    namespaces, so samplers can flatten and re-rank freely. To resolve a neighbor
    tuple back to a ``Tool``, call ``registry.get_by_fingerprint(f"{ns}::{name}", schema_fp)``.

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
        duplicate_sim_thresh: float = 0.9,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._max_candidates = max_candidates
        self._max_neighbors = max_neighbors
        self._duplicate_sim_thresh = duplicate_sim_thresh
        self._force = force
        self._model = None

    def _get_model(self):
        if self._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self._model_name).to(
                device=device, dtype=torch.bfloat16
            )
        return self._model

    def _cache_fingerprint(self, tools: List[Tool]) -> str:
        """Fingerprint: sorted (qualified_name, schema_fp, tool_text) triples + model + max_neighbors."""
        tool_ids = sorted(
            (t.qualified_name, schema_fingerprint(t.parameters), _tool_to_text(t)) for t in tools
        )
        return compute_fingerprint(
            tool_ids, self._model_name, self._max_neighbors, self._duplicate_sim_thresh
        )

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

        # Artifact shape: {qname: {schema_fp: {ns: (neighbors, near_duplicates)}}}
        # Cache stores the same shape with tuples serialized as lists.
        cached: Dict[
            str,
            Dict[str, Dict[str, Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]]],
        ] = {}
        if not self._force:
            raw = load_cache(cache_path)
            if raw:
                cached = {
                    qname: {
                        src_fp: {
                            ns: (
                                [tuple(triple) for triple in neighbors],
                                [tuple(triple) for triple in near_dups],
                            )
                            for ns, (neighbors, near_dups) in ns_buckets.items()
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
                    show_progress_bar=True,
                )
                .float()
                .cpu()
                .numpy()
            )

        # Tool-level similarity matrix for candidate filtering.
        # One row per tool (including overloads — each overload is a distinct row).
        tool_fp_list = [(t.qualified_name, schema_fingerprint(t.parameters)) for t in tools]
        tool_matrix = np.stack([tool_embeddings[qn][fp] for qn, fp in tool_fp_list], axis=0)
        tool_sim: np.ndarray = tool_matrix.dot(tool_matrix.T)

        # Precompute composite embeddings for all tools
        self.logger.info("Precomputing composite embeddings for %d tools", len(tools))
        source_composites = np.zeros((len(tools), all_embs.shape[1]), dtype=np.float32)
        target_composites = np.zeros((len(tools), all_embs.shape[1]), dtype=np.float32)

        for idx, (qname, tool_pos, inp_pos, out_pos) in enumerate(edge_index):
            # Source composite: tool + summed output params
            src_emb = all_embs[tool_pos]
            if out_pos:
                src_out = np.sum(all_embs[out_pos], axis=0)
                source_composites[idx] = src_emb + src_out
            else:
                source_composites[idx] = src_emb

            # Target composite: tool + summed input params
            if inp_pos:
                tgt_inp = np.sum(all_embs[inp_pos], axis=0)
                target_composites[idx] = src_emb + tgt_inp
            else:
                target_composites[idx] = src_emb

        # Per-tool edge data keyed by (qualified_name, schema_fp) for fast lookup.
        edge_by_key: Dict[Tuple[str, str], int] = {
            (qname, schema_fingerprint(tools[idx].parameters)): idx
            for idx, (qname, _tool_pos, _inp_pos, _out_pos) in enumerate(edge_index)
        }

        # Precompute full edge similarity matrix (N x N)
        self.logger.info("Computing full edge similarity matrix")
        src_norms = source_composites / (
            np.linalg.norm(source_composites, axis=1, keepdims=True) + 1e-8
        )
        tgt_norms = target_composites / (
            np.linalg.norm(target_composites, axis=1, keepdims=True) + 1e-8
        )
        full_edge_sim: np.ndarray = src_norms @ tgt_norms.T

        to_compute_set = {self._src_key(t) for t in tools_to_compute}
        # new_entries mirrors cache shape but keyed by (qname, src_fp) during build.
        new_flat: Dict[
            Tuple[str, str],
            Dict[str, Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]],
        ] = {}

        ns_info: Dict[str, Tuple] = dict()
        for ns, ns_tools in ns_to_tools.items():
            if not ns_tools:
                continue

            # Get fingerprints once
            ns_tools_w_fingerprints: List[Tuple[Tool, str]] = []
            for tool in ns_tools:
                ns_tools_w_fingerprints.append((tool, schema_fingerprint(tool.parameters)))

            # Get all tool indices for this namespace
            ns_indices = [
                edge_by_key[(t.qualified_name, t_fp)] for t, t_fp in ns_tools_w_fingerprints
            ]
            ns_info[ns] = (ns_indices, ns_tools_w_fingerprints)

        for idx, (src_qname, _, _, _) in enumerate(tqdm(edge_index, desc="Neighbor Extraction")):
            src_fp = schema_fingerprint(tools[idx].parameters)
            if (src_qname, src_fp) not in to_compute_set:
                continue

            # Get precomputed similarity scores
            src_tool_row = tool_sim[idx]
            src_edge_row = full_edge_sim[idx]

            ns_buckets: Dict[
                str, Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]
            ] = {}

            for ns, (ns_indices, ns_tools_w_fingerprints) in ns_info.items():
                # Extract precomputed scores
                tool_sims = src_tool_row[ns_indices]
                edge_weights = src_edge_row[ns_indices]

                # Build scored lists (exclude self after computation)
                near_dups = []
                edge_scored = []
                for i, (tool, tgt_fp) in enumerate(ns_tools_w_fingerprints):
                    # Skip self
                    if tool.qualified_name == src_qname and tgt_fp == src_fp:
                        continue

                    # Only track near duplicates if they meet the threshold
                    dup_sim = float(tool_sims[i])
                    if dup_sim >= self._duplicate_sim_thresh:
                        near_dups.append((dup_sim, tool.name, tgt_fp))

                    edge_scored.append((float(edge_weights[i]), tool.name, tgt_fp))

                # Get top-k neighbors using heapq (O(n log k) instead of O(n log n))
                n_neighbors = min(self._max_neighbors, len(edge_scored))
                top_neighbors = nlargest(n_neighbors, edge_scored, key=lambda x: x[0])
                neighbors = [(name, fp, round(score, 4)) for score, name, fp in top_neighbors]

                # Sort near duplicates by score descending
                near_dups.sort(key=lambda x: x[0], reverse=True)
                near_dups = [(name, fp, round(score, 4)) for score, name, fp in near_dups]

                ns_buckets[ns] = (neighbors, near_dups)

            new_flat[(src_qname, src_fp)] = ns_buckets

        # Fold new_flat into the two-level cache shape for persistence.
        new_entries: Dict[
            str,
            Dict[str, Dict[str, Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]]],
        ] = {}
        for (qname, src_fp), ns_buckets in new_flat.items():
            new_entries.setdefault(qname, {})[src_fp] = ns_buckets

        save_cache(cache_path, new_entries)

        # Merge cached + new at the qname level, then merge fp maps.
        merged: Dict[
            str,
            Dict[str, Dict[str, Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]]],
        ] = dict(cached)
        for qname, fp_map in new_entries.items():
            if qname in merged:
                merged[qname] = {**merged[qname], **fp_map}
            else:
                merged[qname] = fp_map

        registry.artifacts[NeighborsEnrichment.artifact_key] = merged
        self.logger.info(
            "NeighborsEnrichment: neighbor graph stored (%d source tools)", len(merged)
        )

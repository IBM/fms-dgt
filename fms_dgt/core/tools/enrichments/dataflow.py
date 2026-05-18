# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Third Party
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Local
from fms_dgt.core.tools.constants import (
    TOOL_DESCRIPTION,
    TOOL_PROPERTIES,
    TOOL_TYPE,
)
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.base import ToolEnrichment, register_tool_enrichment
from fms_dgt.core.tools.enrichments.cache import (
    compute_fingerprint,
    enrichment_cache_path,
    load_cache,
)
from fms_dgt.core.tools.registry import schema_fingerprint
from fms_dgt.utils import write_json

# ===========================================================================
#                       PARAMETER TEXT HELPER
# ===========================================================================


def _make_param_text(name: str, info: Any) -> str:
    """Compact text for a single parameter: name + type + description.

    Includes type (unlike ``neighbors._make_param_string``) to improve
    discrimination between structurally different parameters.
    """
    if not isinstance(info, dict):
        return f"Name: {name}"
    type_str = info.get(TOOL_TYPE, "")
    descr = info.get(TOOL_DESCRIPTION, "")
    return f"Name: {name}\nType: {type_str}\nDescription: {descr}"


def _iter_out_params(tool: Tool):
    """Yield (param_name, param_info) pairs from tool.output_parameters."""
    yield from ((tool.output_parameters or {}).get(TOOL_PROPERTIES) or {}).items()


def _iter_in_params(tool: Tool):
    """Yield (param_name, param_info) pairs from tool.parameters."""
    yield from ((tool.parameters or {}).get(TOOL_PROPERTIES) or {}).items()


# ===========================================================================
#                       TYPE COMPATIBILITY
# ===========================================================================


def _type_multiplier(out_info: Any, in_info: Any) -> float:
    """Return a [0, 1] multiplier for (output_param, input_param) type compatibility.

    Rules:
    - ``array`` ↔ non-``array``: 0.0 (hard kill — lists cannot feed scalars)
    - ``object`` ↔ ``object``, both have ``properties``, non-empty intersection: 0.9
    - ``object`` ↔ ``object``, both have ``properties``, empty intersection: 0.0
    - ``object`` ↔ ``object``, either lacks ``properties`` (opaque): 0.7
    - Exact scalar match (``string``↔``string``, etc.): 1.0
    - ``integer`` ↔ ``number`` (compatible scalars): 0.8
    - All other mismatches (``string``↔``number``, etc.): 0.0 (hard kill)
    """
    out_t = (out_info or {}).get(TOOL_TYPE, "") if isinstance(out_info, dict) else ""
    in_t = (in_info or {}).get(TOOL_TYPE, "") if isinstance(in_info, dict) else ""

    # Array boundary — one is array, the other is not.
    if (out_t == "array") != (in_t == "array"):
        return 0.0

    if out_t == "object" and in_t == "object":
        out_props = set((out_info or {}).get(TOOL_PROPERTIES, {}).keys())
        in_props = set((in_info or {}).get(TOOL_PROPERTIES, {}).keys())
        if out_props and in_props:
            return 0.9 if (out_props & in_props) else 0.0
        return 0.7  # opaque object — cannot inspect further

    if out_t == in_t:
        return 1.0

    if {out_t, in_t} == {"integer", "number"}:
        return 0.8

    return 0.0  # all other scalar mismatches


# ===========================================================================
#                       PARAM INDEX BUILDER
# ===========================================================================

# Row descriptor: (tool_key, param_name, param_info, sentence_pos)
_RowDesc = Tuple[Tuple[str, str], str, Any, int]


def _build_param_index(
    tools: List[Tool],
) -> Tuple[
    List[str],
    Dict[Tuple[str, str], Tuple[List[_RowDesc], List[_RowDesc]]],
]:
    """Build the flat sentence list and per-tool parameter row descriptors.

    Returns:
        sentences: Flat list of parameter text strings for ``model.encode()``.
        param_index: Maps ``(qualified_name, schema_fp)`` to
            ``(out_rows, in_rows)`` where each row is
            ``(tool_key, param_name, param_info, sentence_pos)``.
    """
    sentences: List[str] = []
    param_index: Dict[Tuple[str, str], Tuple[List[_RowDesc], List[_RowDesc]]] = {}

    for tool in tools:
        key = (tool.qualified_name, schema_fingerprint(tool.parameters))
        out_rows: List[_RowDesc] = []
        in_rows: List[_RowDesc] = []

        for pname, pinfo in _iter_out_params(tool):
            pos = len(sentences)
            sentences.append(_make_param_text(pname, pinfo))
            out_rows.append((key, pname, pinfo, pos))

        for pname, pinfo in _iter_in_params(tool):
            pos = len(sentences)
            sentences.append(_make_param_text(pname, pinfo))
            in_rows.append((key, pname, pinfo, pos))

        param_index[key] = (out_rows, in_rows)

    return sentences, param_index


# ===========================================================================
#                       ENRICHMENT
# ===========================================================================


@register_tool_enrichment("dataflow")
class DataflowEnrichment(ToolEnrichment):
    """Compute pairwise output→input parameter dataflow edges between tools.

    An edge A→B exists when at least one output parameter of A is both
    type-compatible and semantically similar to at least one input parameter
    of B.  The edge score is the maximum cosine similarity over all
    type-compatible parameter pairs (output param of A vs input param of B),
    further scaled by the type-compatibility multiplier for that pair.

    **One artifact written in one pass:**

    ``registry.artifacts["dataflow"]`` — a dict with two sub-keys:

    - ``"out"`` — forward index: for each source tool A, its successors
      (tools that can consume A's output).
    - ``"in"`` — reverse index: for each sink tool B, its predecessors
      (tools whose output can feed B's input).

    Both sub-indexes share the same shape, extended with a matched-pairs
    list per edge::

        {
          src_qname: {
            src_fp: {
              namespace: [
                (tgt_name, tgt_fp, edge_score, [(out_param, in_param, pair_score), ...]),
                ...
              ]
            }
          }
        }

    The matched-pairs list is sorted by pair score descending and contains all
    compatible pairs.  Samplers that do not need matched pairs may ignore the
    field.  ``tc/dag`` uses it to construct prompts.

    **Algorithm:**

    1. Embed all per-parameter sentences in one ``model.encode()`` batch.
    2. Build type compatibility multiplier matrix ``M [out_params, in_params]``
       via vectorized comparisons (no Python loop over pairs).
    3. Compute ``S = out_embs @ in_embs.T`` (cosine similarity — vectors are
       L2-normalized by ``normalize_embeddings=True``).  Multiply element-wise by ``M`` to
       get ``scored``.
    4. For each directed tool pair (A, B), slice the ``[p_out, q_in]``
       submatrix, take the max as edge score, record all positive matched pairs.
    5. Build forward index, then invert to build reverse index.

    **No dependency on EmbeddingsEnrichment.**  Per-parameter embeddings are
    produced independently here; reusing tool-level vectors from
    ``EmbeddingsEnrichment`` would require that enrichment to run first
    without adding any actual benefit.

    **Cache fingerprint (Option B):** Computed over the per-parameter texts
    actually fed to the encoder, not over the full-tool text.  This ensures
    the cache busts correctly if the parameter text format changes even when
    tool qualified names and input schemas stay the same.

    Results are cached under
    ``{DGT_CACHE_DIR}/enrichments/dataflow/{fingerprint}.json`` and
    delta-merged on subsequent runs.

    Args:
        model: Sentence-transformer model identifier.
            Defaults to ``"sentence-transformers/all-mpnet-base-v2"``.
        max_neighbors: Maximum edges to retain per namespace bucket in the
            final artifact (default 10).
        force: If ``True``, bypass cache and recompute.
    """

    depends_on: List[str] = []
    artifact_key: str = "dataflow"

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        max_neighbors: int = 10,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._max_neighbors = max_neighbors
        self._force = force
        self._model = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self._model_name).to(
                device=device, dtype=torch.float32
            )
        return self._model

    def _cache_fingerprint(self, tools: List[Tool]) -> str:
        """Fingerprint over per-parameter texts actually fed to the encoder.

        Includes model name and max_neighbors so cache busts when either
        changes.  Sorted to be stable across insertion order.
        """
        tool_ids = sorted(
            (
                t.qualified_name,
                schema_fingerprint(t.parameters),
                sorted(
                    _make_param_text(n, i)
                    for n, i in list(_iter_out_params(t)) + list(_iter_in_params(t))
                ),
            )
            for t in tools
        )
        return compute_fingerprint(tool_ids, self._model_name, self._max_neighbors)

    @staticmethod
    def _src_key(tool: Tool) -> Tuple[str, str]:
        return (tool.qualified_name, schema_fingerprint(tool.parameters))

    def enrich(self, registry: Any) -> None:
        """Compute dataflow edges and write both artifacts to the registry.

        Args:
            registry: ``ToolRegistry`` instance.
        """
        tools = registry.all_tools()
        if not tools:
            registry.artifacts[self.artifact_key] = {"out": {}, "in": {}}
            return

        # --- Cache lookup ---------------------------------------------------
        fingerprint = self._cache_fingerprint(tools)
        cache_path = enrichment_cache_path(self.artifact_key, fingerprint)

        # Shape on disk: {"out": {qname: {fp: {ns: [[...]]}}},
        #                 "in":  {qname: {fp: {ns: [[...]]}}}}
        # JSON round-trips tuples as lists; we convert back on load.
        cached_fwd: Dict[str, Any] = {}
        cached_rev: Dict[str, Any] = {}
        if not self._force:
            cached = load_cache(cache_path)
            cached_fwd = cached.get("out", {})
            cached_rev = cached.get("in", {})
            if cached_fwd:
                self.logger.debug(
                    "DataflowEnrichment: loaded forward cache from %s (%d entries)",
                    cache_path,
                    len(cached_fwd),
                )

        # Determine which tools still need computation.
        cached_keys = {(qn, fp) for qn, fp_map in cached_fwd.items() for fp in fp_map}
        tools_to_compute = [t for t in tools if self._src_key(t) not in cached_keys]

        if not tools_to_compute:
            self.logger.info("DataflowEnrichment: all %d tool(s) satisfied from cache", len(tools))
            registry.artifacts[self.artifact_key] = {
                "out": self._deserialize(cached_fwd),
                "in": self._deserialize(cached_rev),
            }
            return

        self.logger.info(
            "DataflowEnrichment: computing dataflow graph for %d tool(s) (%d cache hits)",
            len(tools_to_compute),
            len(tools) - len(tools_to_compute),
        )

        # --- Build param index and embed ------------------------------------
        sentences, param_index = _build_param_index(tools)
        model = self._get_model()

        with torch.no_grad():
            all_embs = (
                model.encode(
                    sentences,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                .cpu()
                .numpy()
            )  # shape [N_sentences, dim], L2-normalized

        # Collect all output-param and input-param rows across all tools,
        # with enough bookkeeping to slice per-tool submatrices later.
        #
        # all_out_rows[i] = (tool_key, param_name, param_info)
        # all_out_pos[i]  = sentence index in `sentences`
        all_out_rows: List[Tuple[Tuple[str, str], str, Any]] = []
        all_out_pos: List[int] = []
        all_in_rows: List[Tuple[Tuple[str, str], str, Any]] = []
        all_in_pos: List[int] = []

        # Also track per-tool slice boundaries in the flat row lists.
        # tool_out_slice[key] = (start, end) into all_out_rows
        # tool_in_slice[key]  = (start, end) into all_in_rows
        tool_out_slice: Dict[Tuple[str, str], Tuple[int, int]] = {}
        tool_in_slice: Dict[Tuple[str, str], Tuple[int, int]] = {}

        for tool in tools:
            key = self._src_key(tool)
            out_rows, in_rows = param_index[key]

            out_start = len(all_out_rows)
            for _, pname, pinfo, pos in out_rows:
                all_out_rows.append((key, pname, pinfo))
                all_out_pos.append(pos)
            tool_out_slice[key] = (out_start, len(all_out_rows))

            in_start = len(all_in_rows)
            for _, pname, pinfo, pos in in_rows:
                all_in_rows.append((key, pname, pinfo))
                all_in_pos.append(pos)
            tool_in_slice[key] = (in_start, len(all_in_rows))

        # Build embedding matrices O [n_out, dim] and I [n_in, dim].
        if not all_out_pos or not all_in_pos:
            # No tools have output or input parameters — no edges possible.
            self.logger.info("DataflowEnrichment: no parameterized tools; empty graph.")
            registry.artifacts[self.artifact_key] = {"out": {}, "in": {}}
            return

        out_embs = all_embs[np.array(all_out_pos)]  # [n_out, dim]
        in_embs = all_embs[np.array(all_in_pos)]  # [n_in,  dim]

        # Full pairwise cosine similarity (L2-normalized → dot product).
        # Shape [n_out, n_in].
        S = out_embs @ in_embs.T

        # Build type multiplier matrix M [n_out, n_in] via broadcasting.
        # We need the scalar types for each row — extract into flat lists.
        out_types = [
            (r[2] or {}).get(TOOL_TYPE, "") if isinstance(r[2], dict) else "" for r in all_out_rows
        ]
        in_types = [
            (r[2] or {}).get(TOOL_TYPE, "") if isinstance(r[2], dict) else "" for r in all_in_rows
        ]

        M = self._build_type_matrix(all_out_rows, all_in_rows, out_types, in_types)

        # Element-wise: pair_score = similarity * type_multiplier.
        scored = S * M  # [n_out, n_in]

        # --- Build forward edges for tools_to_compute ----------------------
        to_compute_set = {self._src_key(t) for t in tools_to_compute}

        # Namespace index: key -> namespace (for bucketing edges).
        key_to_ns: Dict[Tuple[str, str], str] = {self._src_key(t): t.namespace for t in tools}
        key_to_name: Dict[Tuple[str, str], str] = {self._src_key(t): t.name for t in tools}

        # new_fwd: {(src_qname, src_fp): {ns: [(tgt_name, tgt_fp, score, pairs)]}}
        new_fwd: Dict[Tuple[str, str], Dict[str, List]] = {}

        for src_key in to_compute_set:
            src_out_start, src_out_end = tool_out_slice.get(src_key, (0, 0))
            if src_out_start == src_out_end:
                # Source has no output params — cannot have successors.
                new_fwd[src_key] = {}
                continue

            ns_buckets: Dict[str, List] = defaultdict(list)

            for tgt_key in tool_out_slice:
                if tgt_key == src_key:
                    continue
                tgt_in_start, tgt_in_end = tool_in_slice.get(tgt_key, (0, 0))
                if tgt_in_start == tgt_in_end:
                    continue  # target has no input params

                # Slice the scored submatrix for this (src, tgt) pair.
                sub = scored[src_out_start:src_out_end, tgt_in_start:tgt_in_end]
                if sub.max() <= 0.0:
                    continue  # no compatible pairs

                edge_score = float(sub.max())

                # All positive matched pairs, sorted by score descending.
                flat = sub.flatten()
                top_idx = np.argsort(flat)[::-1]
                pairs = []
                n_cols = tgt_in_end - tgt_in_start
                for idx in top_idx:
                    ps = float(flat[idx])
                    if ps <= 0.0:
                        break
                    row_i = int(idx) // n_cols
                    col_i = int(idx) % n_cols
                    out_pname = all_out_rows[src_out_start + row_i][1]
                    in_pname = all_in_rows[tgt_in_start + col_i][1]
                    pairs.append((out_pname, in_pname, round(ps, 4)))

                tgt_ns = key_to_ns[tgt_key]
                tgt_name = key_to_name[tgt_key]
                tgt_fp = tgt_key[1]
                ns_buckets[tgt_ns].append((tgt_name, tgt_fp, round(edge_score, 4), pairs))

            # Sort and trim each namespace bucket.
            trimmed: Dict[str, List] = {}
            for ns, edges in ns_buckets.items():
                edges.sort(key=lambda e: e[2], reverse=True)
                trimmed[ns] = edges[: self._max_neighbors]

            new_fwd[src_key] = trimmed

        # --- Build reverse index from all forward edges --------------------
        # Include both cached and newly computed forward entries.
        all_fwd_flat: Dict[Tuple[str, str], Dict[str, List]] = {}

        # Deserialize cached entries into the same structure.
        for qn, fp_map in cached_fwd.items():
            for fp, ns_map in fp_map.items():
                k_ = (qn, fp)
                all_fwd_flat[k_] = {
                    ns: [self._entry_from_list(e) for e in edges] for ns, edges in ns_map.items()
                }

        all_fwd_flat.update(new_fwd)

        new_rev: Dict[Tuple[str, str], Dict[str, List]] = defaultdict(lambda: defaultdict(list))

        for src_key, ns_map in all_fwd_flat.items():
            src_qn, src_fp = src_key
            src_name = key_to_name.get(src_key, src_qn.split("::", 1)[-1])
            for ns, edges in ns_map.items():
                for entry in edges:
                    tgt_name, tgt_fp, edge_score, pairs = entry
                    tgt_qn = f"{ns}::{tgt_name}"
                    tgt_key = (tgt_qn, tgt_fp)
                    # Reversed pairs: swap (out_param, in_param) order.
                    rev_pairs = [(ip, op, ps) for op, ip, ps in pairs]
                    new_rev[tgt_key][key_to_ns.get(src_key, src_qn.split("::", 1)[0])].append(
                        (src_name, src_fp, edge_score, rev_pairs)
                    )

        # Sort and trim reverse buckets.
        trimmed_rev: Dict[Tuple[str, str], Dict[str, List]] = {}
        for tgt_key, ns_map in new_rev.items():
            trimmed_rev[tgt_key] = {}
            for ns, edges in ns_map.items():
                edges.sort(key=lambda e: e[2], reverse=True)
                trimmed_rev[tgt_key][ns] = edges[: self._max_neighbors]

        # --- Merge forward: cached + new at qname level ---------------------
        merged_fwd: Dict[str, Any] = {}
        for qn, fp_map in cached_fwd.items():
            merged_fwd[qn] = {
                fp: {ns: [self._entry_from_list(e) for e in edges] for ns, edges in ns_map.items()}
                for fp, ns_map in fp_map.items()
            }
        for (qn, fp), ns_map in new_fwd.items():
            merged_fwd.setdefault(qn, {})[fp] = ns_map

        # Reverse: fully rebuilt from all_fwd_flat, already in trimmed_rev.
        final_rev: Dict[str, Any] = {}
        for (qn, fp), ns_map in trimmed_rev.items():
            final_rev.setdefault(qn, {})[fp] = ns_map

        # --- Persist cache --------------------------------------------------
        # Serialize merged forward and full reverse together under one key.
        fwd_serial: Dict[str, Any] = {}
        for qn, fp_map in merged_fwd.items():
            fwd_serial[qn] = {
                fp: {ns: [self._entry_to_list(e) for e in edges] for ns, edges in ns_map.items()}
                for fp, ns_map in fp_map.items()
            }
        rev_serial: Dict[str, Any] = {}
        for (qn, fp), ns_map in trimmed_rev.items():
            rev_serial.setdefault(qn, {})[fp] = {
                ns: [self._entry_to_list(e) for e in edges] for ns, edges in ns_map.items()
            }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_json({"out": fwd_serial, "in": rev_serial}, str(cache_path))
        except Exception as exc:
            self.logger.warning(
                "DataflowEnrichment: could not write cache at %s: %s",
                cache_path,
                exc,
            )

        # --- Write artifact -------------------------------------------------
        registry.artifacts[self.artifact_key] = {"out": merged_fwd, "in": final_rev}

        self.logger.info(
            "DataflowEnrichment: graph stored — %d forward sources, %d reverse sinks",
            len(merged_fwd),
            len(final_rev),
        )

    # ------------------------------------------------------------------
    # Type multiplier matrix (vectorized)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_type_matrix(
        out_rows: List[Tuple],
        in_rows: List[Tuple],
        out_types: List[str],
        in_types: List[str],
    ) -> np.ndarray:
        """Build the [n_out, n_in] type compatibility multiplier matrix.

        Uses vectorized string comparisons where possible; the object/properties
        check requires a Python loop only over the object-typed pairs.
        """
        n_out = len(out_rows)
        n_in = len(in_rows)
        M = np.ones((n_out, n_in), dtype=np.float32)

        out_arr = np.array(out_types)  # [n_out]
        in_arr = np.array(in_types)  # [n_in]

        # Array boundary: one side is array, other is not → 0.0.
        out_is_array = (out_arr == "array")[:, None]  # [n_out, 1]
        in_is_array = (in_arr == "array")[None, :]  # [1, n_in]
        array_kill = out_is_array ^ in_is_array
        M[array_kill] = 0.0

        # Exact scalar match → 1.0 (already 1.0 from initialization; no-op).

        # integer ↔ number → 0.8.
        out_int = (out_arr == "integer")[:, None]
        in_num = (in_arr == "number")[None, :]
        out_num = (out_arr == "number")[:, None]
        in_int = (in_arr == "integer")[None, :]
        int_num = (out_int & in_num) | (out_num & in_int)
        M[int_num] = 0.8

        # Scalar mismatch (neither is array, not both object, not same type,
        # not int/num compatible) → 0.0.
        out_is_obj = (out_arr == "object")[:, None]
        in_is_obj = (in_arr == "object")[None, :]
        both_obj = out_is_obj & in_is_obj
        same_type = out_arr[:, None] == in_arr[None, :]
        # Pairs that need scalar-mismatch kill: not array kill, not both object,
        # not same type, not int/num.
        scalar_kill = ~array_kill & ~both_obj & ~same_type & ~int_num
        M[scalar_kill] = 0.0

        # Object ↔ object: check properties intersection.
        obj_out_indices = np.where(out_arr == "object")[0]
        obj_in_indices = np.where(in_arr == "object")[0]
        for oi in obj_out_indices:
            out_info = out_rows[oi][2]
            out_props = (
                set((out_info or {}).get(TOOL_PROPERTIES, {}).keys())
                if isinstance(out_info, dict)
                else set()
            )
            for ii in obj_in_indices:
                in_info = in_rows[ii][2]
                in_props = (
                    set((in_info or {}).get(TOOL_PROPERTIES, {}).keys())
                    if isinstance(in_info, dict)
                    else set()
                )
                if out_props and in_props:
                    M[oi, ii] = 0.9 if (out_props & in_props) else 0.0
                else:
                    M[oi, ii] = 0.7  # opaque object

        return M

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_list(entry: tuple) -> list:
        """Serialize an edge entry tuple to a JSON-safe nested list."""
        tgt_name, tgt_fp, score, pairs = entry
        return [tgt_name, tgt_fp, score, [[op, ip, ps] for op, ip, ps in pairs]]

    @staticmethod
    def _entry_from_list(raw: list) -> tuple:
        """Deserialize a JSON-loaded nested list back to an edge entry tuple."""
        tgt_name, tgt_fp, score, pairs_raw = raw
        return (tgt_name, tgt_fp, score, [(p[0], p[1], p[2]) for p in pairs_raw])

    @staticmethod
    def _deserialize(
        cached: Dict[str, Any],
    ) -> Dict[str, Dict[str, Dict[str, List]]]:
        """Convert a raw loaded cache dict (lists) to the live artifact shape (tuples)."""
        result: Dict[str, Dict[str, Dict[str, List]]] = {}
        for qn, fp_map in cached.items():
            result[qn] = {}
            for fp, ns_map in fp_map.items():
                result[qn][fp] = {
                    ns: [DataflowEnrichment._entry_from_list(e) for e in edges]
                    for ns, edges in ns_map.items()
                }
        return result

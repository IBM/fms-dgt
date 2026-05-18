# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DataflowEnrichment."""

# Standard
from typing import Dict
from unittest.mock import MagicMock

# Third Party
import numpy as np
import torch

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.dataflow import (
    DataflowEnrichment,
    _type_multiplier,
)
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint

DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(
    name: str,
    ns: str = "ns",
    in_params: dict = None,
    out_params: dict = None,
) -> Tool:
    """Build a tool with explicit input and output parameter schemas."""
    parameters = {"type": "object", "properties": in_params} if in_params else {}
    output_parameters = {"type": "object", "properties": out_params} if out_params else {}
    return Tool(
        name=name,
        namespace=ns,
        description=f"Description of {name}.",
        parameters=parameters,
        output_parameters=output_parameters,
    )


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _make_enrichment(dim: int = DIM) -> DataflowEnrichment:
    """Return a DataflowEnrichment with a mocked sentence-transformer."""
    enrichment = DataflowEnrichment.__new__(DataflowEnrichment)
    enrichment._model_name = "mock-model"
    enrichment._max_neighbors = 10
    enrichment._force = True

    mock_model = MagicMock()

    def _fake_encode(texts, convert_to_tensor, normalize_embeddings, show_progress_bar):
        # Return L2-normalized random vectors.
        vecs = torch.rand(len(texts), dim, dtype=torch.float32)
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return vecs

    mock_model.encode.side_effect = _fake_encode
    enrichment._model = mock_model
    return enrichment


def _make_deterministic_enrichment(
    param_vectors: Dict[str, np.ndarray], dim: int = DIM
) -> DataflowEnrichment:
    """Return an enrichment whose encoder returns fixed vectors keyed by text.

    Useful for type-compatibility tests where we need predictable scores.
    ``param_vectors`` maps parameter text (from ``_make_param_text``) to a
    unit vector.  Unknown texts get a zero vector.
    """
    enrichment = DataflowEnrichment.__new__(DataflowEnrichment)
    enrichment._model_name = "mock-model"
    enrichment._max_neighbors = 10
    enrichment._force = True

    mock_model = MagicMock()

    def _fixed_encode(texts, convert_to_tensor, normalize_embeddings, show_progress_bar):
        rows = []
        for t in texts:
            vec = param_vectors.get(t, np.zeros(dim, dtype=np.float32))
            rows.append(vec)
        arr = np.stack(rows, axis=0).astype(np.float32)
        return torch.from_numpy(arr)

    mock_model.encode.side_effect = _fixed_encode
    enrichment._model = mock_model
    return enrichment


# ===========================================================================
#                       Class attributes
# ===========================================================================


class TestDataflowEnrichmentAttributes:
    def test_depends_on_is_empty(self):
        assert DataflowEnrichment.depends_on == []

    def test_artifact_key(self):
        assert DataflowEnrichment.artifact_key == DataflowEnrichment.artifact_key


# ===========================================================================
#                       Empty registry
# ===========================================================================


class TestDataflowEnrichmentEmpty:
    def test_empty_registry_writes_empty_artifact(self):
        reg = _registry()
        _make_enrichment().enrich(reg)
        assert reg.artifacts[DataflowEnrichment.artifact_key] == {"out": {}, "in": {}}


# ===========================================================================
#                       Artifact structure
# ===========================================================================


class TestDataflowEnrichmentStructure:
    def test_artifact_written_with_out_and_in(self):
        t1 = _tool("a", out_params={"city": {"type": "string"}})
        t2 = _tool("b", in_params={"location": {"type": "string"}})
        reg = _registry(t1, t2)
        _make_enrichment().enrich(reg)
        assert DataflowEnrichment.artifact_key in reg.artifacts
        assert "out" in reg.artifacts[DataflowEnrichment.artifact_key]
        assert "in" in reg.artifacts[DataflowEnrichment.artifact_key]

    def test_forward_artifact_structure(self):
        """Shape: {qname: {fp: {ns: [(name, fp, score, pairs)]}}}."""
        t1 = _tool("a", out_params={"city": {"type": "string"}})
        t2 = _tool("b", in_params={"location": {"type": "string"}})
        reg = _registry(t1, t2)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        for src_qname, fp_map in fwd.items():
            assert "::" in src_qname
            for src_fp, ns_map in fp_map.items():
                assert isinstance(src_fp, str)
                for ns, edges in ns_map.items():
                    assert isinstance(ns, str)
                    for entry in edges:
                        tgt_name, tgt_fp, score, pairs = entry
                        assert "::" not in tgt_name  # unqualified
                        assert isinstance(tgt_fp, str)
                        assert isinstance(score, float)
                        assert isinstance(pairs, list)

    def test_reverse_is_inverse_of_forward(self):
        """Every A→B forward edge appears as a B←A reverse edge."""
        t1 = _tool("a", out_params={"city": {"type": "string"}})
        t2 = _tool("b", in_params={"location": {"type": "string"}})
        t3 = _tool("c", in_params={"place": {"type": "string"}})
        reg = _registry(t1, t2, t3)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        rev = reg.artifacts[DataflowEnrichment.artifact_key]["in"]

        for src_qname, fp_map in fwd.items():
            src_ns = src_qname.split("::", 1)[0]
            src_name = src_qname.split("::", 1)[1]
            for src_fp, ns_map in fp_map.items():
                for ns, edges in ns_map.items():
                    for tgt_name, tgt_fp, score, _pairs in edges:
                        tgt_qname = f"{ns}::{tgt_name}"
                        # The reverse artifact must have tgt as key.
                        assert tgt_qname in rev, f"reverse missing sink {tgt_qname}"
                        tgt_rev_fp_map = rev[tgt_qname]
                        assert tgt_fp in tgt_rev_fp_map
                        # src must appear as a predecessor.
                        preds = tgt_rev_fp_map[tgt_fp].get(src_ns, [])
                        pred_names = [p[0] for p in preds]
                        assert (
                            src_name in pred_names
                        ), f"reverse missing predecessor {src_name} for sink {tgt_qname}"

    def test_self_not_in_successors(self):
        tools = [
            _tool(
                f"t{i}", out_params={"x": {"type": "string"}}, in_params={"y": {"type": "string"}}
            )
            for i in range(4)
        ]
        reg = _registry(*tools)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        for src_qname, fp_map in fwd.items():
            src_name = src_qname.split("::", 1)[1]
            for ns_map in fp_map.values():
                for ns, edges in ns_map.items():
                    names = [e[0] for e in edges]
                    assert src_name not in names, f"{src_name} found in its own successors"

    def test_scores_are_floats_in_valid_range(self):
        tools = [
            _tool("a", out_params={"x": {"type": "string"}}),
            _tool("b", in_params={"y": {"type": "string"}}),
        ]
        reg = _registry(*tools)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        for fp_map in fwd.values():
            for ns_map in fp_map.values():
                for edges in ns_map.values():
                    for _, _, score, pairs in edges:
                        assert isinstance(score, float)
                        assert 0.0 <= score <= 1.0 + 1e-6, f"score out of range: {score}"
                        for op, ip, ps in pairs:
                            assert isinstance(ps, float)
                            assert 0.0 <= ps <= 1.0 + 1e-6

    def test_pairs_sorted_by_score_desc(self):
        out_params = {f"out{i}": {"type": "string"} for i in range(4)}
        in_params = {f"in{i}": {"type": "string"} for i in range(4)}
        t1 = _tool("a", out_params=out_params)
        t2 = _tool("b", in_params=in_params)
        reg = _registry(t1, t2)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        for fp_map in fwd.values():
            for ns_map in fp_map.values():
                for edges in ns_map.values():
                    for _, _, _, pairs in edges:
                        scores = [ps for _, _, ps in pairs]
                        assert scores == sorted(scores, reverse=True)


# ===========================================================================
#                       max_neighbors
# ===========================================================================


class TestDataflowEnrichmentMaxNeighbors:
    def test_max_neighbors_respected(self):
        tools = [
            _tool(
                f"t{i}", out_params={"x": {"type": "string"}}, in_params={"y": {"type": "string"}}
            )
            for i in range(6)
        ]
        reg = _registry(*tools)
        enrichment = _make_enrichment()
        enrichment._max_neighbors = 2
        enrichment.enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        for fp_map in fwd.values():
            for ns_map in fp_map.values():
                for edges in ns_map.values():
                    assert len(edges) <= 2


# ===========================================================================
#                       Namespace bucketing
# ===========================================================================


class TestDataflowEnrichmentNamespaces:
    def test_namespace_bucketing(self):
        """Tools from different namespaces appear in separate buckets."""
        t1 = _tool("a", ns="api_a", out_params={"city": {"type": "string"}})
        t2 = _tool("b", ns="api_a", in_params={"location": {"type": "string"}})
        t3 = _tool("c", ns="api_b", in_params={"place": {"type": "string"}})
        reg = _registry(t1, t2, t3)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        fp = schema_fingerprint(t1.parameters)
        ns_map = fwd[t1.qualified_name][fp]
        # t2 is in api_a, t3 is in api_b — should be in separate buckets.
        assert "api_a" in ns_map
        assert "api_b" in ns_map
        for entry in ns_map["api_a"]:
            assert entry[0] == "b"
        for entry in ns_map["api_b"]:
            assert entry[0] == "c"


# ===========================================================================
#                       Overload handling
# ===========================================================================


class TestDataflowEnrichmentOverloads:
    def test_overloads_get_separate_entries(self):
        t1 = Tool(
            name="search",
            namespace="api",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            output_parameters={"type": "object", "properties": {"result": {"type": "string"}}},
        )
        t2 = Tool(
            name="search",
            namespace="api",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            output_parameters={"type": "object", "properties": {"result": {"type": "string"}}},
        )
        t3 = _tool("consume", ns="api", in_params={"result": {"type": "string"}})
        reg = _registry(t1, t2, t3)
        _make_enrichment().enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        fp1 = schema_fingerprint(t1.parameters)
        fp2 = schema_fingerprint(t2.parameters)
        assert fp1 in fwd["api::search"]
        assert fp2 in fwd["api::search"]


# ===========================================================================
#                       Type compatibility
# ===========================================================================


class TestTypeMultiplier:
    """Unit tests for _type_multiplier directly."""

    def test_exact_string_match(self):
        assert _type_multiplier({"type": "string"}, {"type": "string"}) == 1.0

    def test_exact_number_match(self):
        assert _type_multiplier({"type": "number"}, {"type": "number"}) == 1.0

    def test_integer_number_compatible(self):
        assert _type_multiplier({"type": "integer"}, {"type": "number"}) == 0.8
        assert _type_multiplier({"type": "number"}, {"type": "integer"}) == 0.8

    def test_string_number_mismatch(self):
        assert _type_multiplier({"type": "string"}, {"type": "number"}) == 0.0

    def test_string_integer_mismatch(self):
        assert _type_multiplier({"type": "string"}, {"type": "integer"}) == 0.0

    def test_array_vs_string_hard_kill(self):
        assert _type_multiplier({"type": "array"}, {"type": "string"}) == 0.0
        assert _type_multiplier({"type": "string"}, {"type": "array"}) == 0.0

    def test_array_vs_array(self):
        # Both array → exact match → 1.0.
        assert _type_multiplier({"type": "array"}, {"type": "array"}) == 1.0

    def test_object_object_with_property_intersection(self):
        out = {"type": "object", "properties": {"city": {}, "temp": {}}}
        in_ = {"type": "object", "properties": {"city": {}, "humidity": {}}}
        assert _type_multiplier(out, in_) == 0.9

    def test_object_object_no_property_intersection(self):
        out = {"type": "object", "properties": {"city": {}}}
        in_ = {"type": "object", "properties": {"humidity": {}}}
        assert _type_multiplier(out, in_) == 0.0

    def test_object_object_opaque(self):
        # Neither has properties → opaque.
        assert _type_multiplier({"type": "object"}, {"type": "object"}) == 0.7

    def test_none_info_treated_as_empty(self):
        # Missing info dict should not crash.
        assert _type_multiplier(None, {"type": "string"}) == 0.0
        assert _type_multiplier({"type": "string"}, None) == 0.0


class TestDataflowTypeFiltering:
    """Integration tests: type filters produce correct edges (or no edges)."""

    def test_no_edge_when_array_vs_string(self):
        """Tool with array output, tool with string input: no edge should be stored."""
        t1 = _tool("a", out_params={"items": {"type": "array"}})
        t2 = _tool("b", in_params={"item": {"type": "string"}})
        reg = _registry(t1, t2)
        # Use a deterministic enrichment with high similarity to ensure that
        # if the type filter fires, the score is zeroed regardless.
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        texts_a_out = "Name: items\nType: array\nDescription: "
        texts_b_in = "Name: item\nType: string\nDescription: "
        enrichment = _make_deterministic_enrichment({texts_a_out: vec, texts_b_in: vec})
        enrichment.enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        fp_a = schema_fingerprint(t1.parameters)
        ns_map = fwd.get(t1.qualified_name, {}).get(fp_a, {})
        # No successors should be stored because type is incompatible.
        all_successors = [e[0] for edges in ns_map.values() for e in edges]
        assert "b" not in all_successors

    def test_no_edge_when_string_vs_number(self):
        """String output vs number input: no edge."""
        t1 = _tool("a", out_params={"val": {"type": "string"}})
        t2 = _tool("b", in_params={"count": {"type": "number"}})
        reg = _registry(t1, t2)
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        texts_a_out = "Name: val\nType: string\nDescription: "
        texts_b_in = "Name: count\nType: number\nDescription: "
        enrichment = _make_deterministic_enrichment({texts_a_out: vec, texts_b_in: vec})
        enrichment.enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        fp_a = schema_fingerprint(t1.parameters)
        ns_map = fwd.get(t1.qualified_name, {}).get(fp_a, {})
        all_successors = [e[0] for edges in ns_map.values() for e in edges]
        assert "b" not in all_successors

    def test_edge_present_when_string_matches_string(self):
        """String output to string input with identical vectors: edge should exist."""
        t1 = _tool("a", out_params={"city": {"type": "string"}})
        t2 = _tool("b", in_params={"location": {"type": "string"}})
        reg = _registry(t1, t2)
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        texts_a_out = "Name: city\nType: string\nDescription: "
        texts_b_in = "Name: location\nType: string\nDescription: "
        enrichment = _make_deterministic_enrichment({texts_a_out: vec, texts_b_in: vec})
        enrichment.enrich(reg)
        fwd = reg.artifacts[DataflowEnrichment.artifact_key]["out"]
        fp_a = schema_fingerprint(t1.parameters)
        ns_map = fwd.get(t1.qualified_name, {}).get(fp_a, {})
        all_successors = [e[0] for edges in ns_map.values() for e in edges]
        assert "b" in all_successors

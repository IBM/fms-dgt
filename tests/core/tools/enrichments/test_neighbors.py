# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeighborsEnrichment."""

# Standard
from typing import Dict
from unittest.mock import MagicMock

# Third Party
import numpy as np
import torch

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.embeddings import EmbeddingsEnrichment
from fms_dgt.core.tools.enrichments.neighbors import NeighborsEnrichment
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint

DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, ns: str = "ns", params: dict = None, output_params: dict = None) -> Tool:
    return Tool(
        name=name,
        namespace=ns,
        description=f"Description of {name}.",
        parameters=params or {},
        output_parameters=output_params or {},
    )


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _fake_embeddings(tools: list) -> Dict[str, Dict[str, np.ndarray]]:
    artifact = {}
    for t in tools:
        fp = schema_fingerprint(t.parameters)
        artifact.setdefault(t.qualified_name, {})[fp] = np.random.rand(DIM).astype(np.float32)
    return artifact


def _make_enrichment() -> NeighborsEnrichment:
    """Return a NeighborsEnrichment with a mocked sentence-transformer and cache disabled."""
    enrichment = NeighborsEnrichment.__new__(NeighborsEnrichment)
    enrichment._model_name = "mock-model"
    enrichment._max_candidates = 50
    enrichment._max_neighbors = 10
    enrichment._force = True

    mock_model = MagicMock()

    def _fake_encode(texts, convert_to_tensor, show_progress_bar):
        return torch.rand(len(texts), DIM, dtype=torch.float32)

    mock_model.encode.side_effect = _fake_encode
    enrichment._model = mock_model
    return enrichment


# ===========================================================================
#                       NeighborsEnrichment
# ===========================================================================


class TestNeighborsEnrichment:
    def test_requires_embeddings_artifact(self):
        assert "embeddings" in NeighborsEnrichment.depends_on

    def test_artifact_key(self):
        assert NeighborsEnrichment.artifact_key == "neighbors"

    def test_writes_neighbors_artifact(self):
        tools = [_tool("t1"), _tool("t2"), _tool("t3")]
        reg = _registry(*tools)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings(tools)
        _make_enrichment().enrich(reg)
        assert NeighborsEnrichment.artifact_key in reg.artifacts
        assert len(reg.artifacts[NeighborsEnrichment.artifact_key]) == 3

    def test_artifact_structure(self):
        """Artifact shape: {qname: {schema_fp: {ns: [(name, fp, score)]}}}."""
        t1 = _tool("alpha", ns="api")
        t2 = _tool("beta", ns="api")
        reg = _registry(t1, t2)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings([t1, t2])
        _make_enrichment().enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        for src_qname, fp_map in neighbors.items():
            assert "::" in src_qname
            for src_fp, ns_buckets in fp_map.items():
                assert isinstance(src_fp, str)
                for ns, triples in ns_buckets.items():
                    assert isinstance(ns, str)
                    for name, tgt_fp, score in triples:
                        assert "::" not in name  # unqualified
                        assert isinstance(tgt_fp, str)
                        assert isinstance(score, float)

    def test_self_not_in_neighbors(self):
        tools = [_tool(f"t{i}") for i in range(4)]
        reg = _registry(*tools)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings(tools)
        _make_enrichment().enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        for src_qname, fp_map in neighbors.items():
            src_name = src_qname.split("::", 1)[1]
            for ns_buckets in fp_map.values():
                all_names = [n for triples in ns_buckets.values() for n, _, _ in triples]
                assert src_name not in all_names

    def test_neighbor_scores_are_floats(self):
        tools = [_tool("t1"), _tool("t2")]
        reg = _registry(*tools)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings(tools)
        _make_enrichment().enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        for fp_map in neighbors.values():
            for ns_buckets in fp_map.values():
                for triples in ns_buckets.values():
                    for _, _, score in triples:
                        assert isinstance(score, float)

    def test_empty_registry(self):
        reg = _registry()
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = {}
        _make_enrichment().enrich(reg)
        assert reg.artifacts[NeighborsEnrichment.artifact_key] == {}

    def test_max_neighbors_respected(self):
        """With max_neighbors=1, each namespace bucket holds at most 1 neighbor."""
        tools = [_tool(f"t{i}") for i in range(5)]
        reg = _registry(*tools)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings(tools)
        enrichment = _make_enrichment()
        enrichment._max_neighbors = 1
        enrichment.enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        for fp_map in neighbors.values():
            for ns_buckets in fp_map.values():
                for triples in ns_buckets.values():
                    assert len(triples) <= 1

    def test_namespace_bucketing(self):
        """Tools from different namespaces appear in separate buckets."""
        t1 = _tool("t1", ns="ns_a")
        t2 = _tool("t2", ns="ns_a")
        t3 = _tool("t3", ns="ns_b")
        reg = _registry(t1, t2, t3)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings([t1, t2, t3])
        _make_enrichment().enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        fp = schema_fingerprint(t1.parameters)
        src_buckets = neighbors[t1.qualified_name][fp]
        assert "ns_a" in src_buckets
        assert "ns_b" in src_buckets
        for name, _, _ in src_buckets["ns_a"]:
            assert name == "t2"
        for name, _, _ in src_buckets["ns_b"]:
            assert name == "t3"

    def test_overloads_get_separate_entries(self):
        """Two overloads of the same tool produce independent neighbor sets."""
        t1 = Tool(
            name="search",
            namespace="api",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        t2 = Tool(
            name="search",
            namespace="api",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        )
        t3 = _tool("other", ns="api")
        reg = _registry(t1, t2, t3)
        reg.artifacts[EmbeddingsEnrichment.artifact_key] = _fake_embeddings([t1, t2, t3])
        _make_enrichment().enrich(reg)
        neighbors = reg.artifacts[NeighborsEnrichment.artifact_key]
        fp1 = schema_fingerprint(t1.parameters)
        fp2 = schema_fingerprint(t2.parameters)
        assert fp1 in neighbors["api::search"]
        assert fp2 in neighbors["api::search"]

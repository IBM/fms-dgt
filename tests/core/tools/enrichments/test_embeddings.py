# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for EmbeddingsEnrichment and the _tool_to_text helper."""

# Standard
from unittest.mock import MagicMock

# Third Party
import numpy as np
import torch

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.embeddings import (
    EmbeddingsEnrichment,
    _tool_to_text,
)
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint

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


def _make_enrichment(dim: int = 8) -> EmbeddingsEnrichment:
    """Return an EmbeddingsEnrichment with a mocked sentence-transformer and cache disabled."""
    enrichment = EmbeddingsEnrichment.__new__(EmbeddingsEnrichment)
    enrichment._model_name = "mock-model"
    enrichment._force = True

    mock_model = MagicMock()

    def _fake_encode(texts, convert_to_tensor, show_progress_bar):
        return torch.rand(len(texts), dim, dtype=torch.float32)

    mock_model.encode.side_effect = _fake_encode
    enrichment._model = mock_model
    return enrichment


# ===========================================================================
#                       EmbeddingsEnrichment
# ===========================================================================


class TestEmbeddingsEnrichment:
    def test_writes_embeddings_artifact(self):
        tools = [_tool("t1"), _tool("t2")]
        reg = _registry(*tools)
        _make_enrichment().enrich(reg)
        assert EmbeddingsEnrichment.artifact_key in reg.artifacts
        embs = reg.artifacts[EmbeddingsEnrichment.artifact_key]
        assert len(embs) == 2
        for qname, fp_map in embs.items():
            assert isinstance(fp_map, dict)
            for fp, vec in fp_map.items():
                assert isinstance(fp, str)
                assert isinstance(vec, np.ndarray)
                assert vec.shape == (8,)

    def test_empty_registry_produces_empty_artifact(self):
        reg = _registry()
        _make_enrichment().enrich(reg)
        assert reg.artifacts[EmbeddingsEnrichment.artifact_key] == {}

    def test_outer_keys_are_qualified_names(self):
        t1 = _tool("get_weather", ns="weather_api")
        t2 = _tool("lookup_user", ns="hr_api")
        reg = _registry(t1, t2)
        _make_enrichment().enrich(reg)
        embs = reg.artifacts[EmbeddingsEnrichment.artifact_key]
        assert "weather_api::get_weather" in embs
        assert "hr_api::lookup_user" in embs

    def test_inner_key_is_schema_fingerprint(self):
        t = _tool("get_weather", ns="weather_api")
        reg = _registry(t)
        _make_enrichment().enrich(reg)
        embs = reg.artifacts[EmbeddingsEnrichment.artifact_key]
        fp = schema_fingerprint(t.parameters)
        assert fp in embs["weather_api::get_weather"]

    def test_overloads_get_separate_fp_entries(self):
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
        reg = _registry(t1, t2)
        _make_enrichment().enrich(reg)
        embs = reg.artifacts[EmbeddingsEnrichment.artifact_key]
        assert len(embs["api::search"]) == 2

    def test_idempotent(self):
        t = _tool("t1")
        reg = _registry(t)
        enrichment = _make_enrichment()
        enrichment.enrich(reg)
        fp = schema_fingerprint(t.parameters)
        first_shape = reg.artifacts[EmbeddingsEnrichment.artifact_key]["ns::t1"][fp].shape
        enrichment.enrich(reg)
        second_shape = reg.artifacts[EmbeddingsEnrichment.artifact_key]["ns::t1"][fp].shape
        assert first_shape == second_shape

    def test_depends_on_is_empty(self):
        assert EmbeddingsEnrichment.depends_on == []

    def test_artifact_key(self):
        assert EmbeddingsEnrichment.artifact_key == "embeddings"


# ===========================================================================
#                       _tool_to_text
# ===========================================================================


class TestToolToText:
    def test_basic(self):
        t = Tool(
            name="get_weather",
            namespace="ns",
            description="Get weather.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
            },
        )
        text = _tool_to_text(t)
        assert "get_weather" in text
        assert "Get weather." in text
        assert "location" in text

    def test_no_description(self):
        t = Tool(name="tool", namespace="ns")
        text = _tool_to_text(t)
        assert "tool" in text

    def test_output_params_included(self):
        t = Tool(
            name="t",
            namespace="ns",
            output_parameters={
                "type": "object",
                "properties": {"temp": {"type": "number"}},
            },
        )
        text = _tool_to_text(t)
        assert "temp" in text
        assert "Returns:" in text

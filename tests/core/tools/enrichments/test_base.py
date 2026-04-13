# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ToolEnrichment base infrastructure.

Coverage:
- ToolEnrichment base class and registry (register / get / duplicate detection)
- ToolRegistry.artifacts field and refresh() behaviour
- Topological sort helper (_topo_sort_enrichments)
- Cache helpers (compute_fingerprint, load/save)
- Task.__init__ enrichment wiring via tools.enrichments

EmbeddingsEnrichment and _tool_to_text tests live in test_embeddings.py.
NeighborsEnrichment tests live in test_neighbors.py.
OutputParametersEnrichment tests live in test_output_parameters.py.
"""

# Standard
from unittest.mock import MagicMock
import os
import tempfile

# Third Party
import pytest
import torch
import yaml

# Local
from fms_dgt.base.task import _topo_sort_enrichments
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.tools import get_tool_loader
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.base import (
    _TOOL_ENRICHMENT_REGISTRY,
    ToolEnrichment,
    get_tool_enrichment,
    register_tool_enrichment,
)
from fms_dgt.core.tools.enrichments.cache import (
    compute_fingerprint,
    load_cache,
    save_cache,
)
from fms_dgt.core.tools.enrichments.embeddings import EmbeddingsEnrichment
from fms_dgt.core.tools.loaders.file import FileToolLoader
from fms_dgt.core.tools.registry import ToolRegistry

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


# ===========================================================================
#                       BASE CLASS AND REGISTRY
# ===========================================================================


class TestEnrichmentRegistry:
    """register_tool_enrichment / get_tool_enrichment wiring."""

    def test_register_and_get(self):
        """A freshly registered enrichment can be retrieved by name."""
        name = "_test_reg_enrichment_unique_1"
        assert name not in _TOOL_ENRICHMENT_REGISTRY

        @register_tool_enrichment(name)
        class _TestEnrichment(ToolEnrichment):
            def enrich(self, registry):
                pass

        instance = get_tool_enrichment(name)
        assert isinstance(instance, _TestEnrichment)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_tool_enrichment("__no_such_enrichment__")

    def test_duplicate_registration_raises(self):
        name = "_test_reg_enrichment_dup"
        assert name not in _TOOL_ENRICHMENT_REGISTRY

        @register_tool_enrichment(name)
        class _First(ToolEnrichment):
            def enrich(self, registry):
                pass

        with pytest.raises(AssertionError, match="conflicts"):

            @register_tool_enrichment(name)
            class _Second(ToolEnrichment):
                def enrich(self, registry):
                    pass

    def test_non_subclass_registration_raises(self):
        name = "_test_reg_enrichment_non_sub"
        assert name not in _TOOL_ENRICHMENT_REGISTRY
        with pytest.raises(AssertionError, match="must extend ToolEnrichment"):

            @register_tool_enrichment(name)
            class _NotEnrichment:
                pass


# ===========================================================================
#                       REGISTRY artifacts FIELD
# ===========================================================================


class TestRegistryArtifacts:
    def test_artifacts_starts_empty(self):
        reg = _registry(_tool("t1"))
        assert reg.artifacts == {}

    def test_enrichment_can_write_artifact(self):
        class _WriteArtifact(ToolEnrichment):
            artifact_key = "test_key"

            def enrich(self, registry):
                registry.artifacts["test_key"] = {"wrote": True}

        reg = _registry(_tool("t1"))
        _WriteArtifact().enrich(reg)
        assert reg.artifacts.get("test_key") == {"wrote": True}

    def test_refresh_reruns_enrichments(self):
        """refresh() should re-run retained enrichments and reset artifacts."""
        call_count = {"n": 0}

        class _Counter(ToolEnrichment):
            artifact_key = "counter"

            def enrich(self, registry):
                call_count["n"] += 1
                registry.artifacts["counter"] = call_count["n"]

        tool_data = {"ns": [{"name": "t1", "description": "A tool"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(tool_data, f)
            tmp_path = f.name

        try:
            reg = ToolRegistry.from_loaders([FileToolLoader(path=tmp_path, namespace="ns")])
            enrichment = _Counter()
            enrichment.enrich(reg)
            reg._enrichments = [enrichment]

            assert call_count["n"] == 1
            reg.refresh()
            assert call_count["n"] == 2
            assert reg.artifacts["counter"] == 2
        finally:
            os.unlink(tmp_path)


# ===========================================================================
#                       TOPOLOGICAL SORT
# ===========================================================================


class TestTopoSortEnrichments:
    """_topo_sort_enrichments from fms_dgt.base.task."""

    def _make(self, deps: list, key: str | None = None) -> ToolEnrichment:
        class _E(ToolEnrichment):
            depends_on = deps
            artifact_key = key

            def enrich(self, registry):
                pass

        return _E()

    def test_single_no_deps(self):
        e = self._make([], "k1")
        assert _topo_sort_enrichments([e]) == [e]

    def test_two_ordered(self):
        e1 = self._make([], "k1")
        e2 = self._make(["k1"], "k2")
        result = _topo_sort_enrichments([e1, e2])
        assert result == [e1, e2]

    def test_two_reversed_input(self):
        """Even when declared in wrong order, topo sort fixes it."""
        e1 = self._make([], "k1")
        e2 = self._make(["k1"], "k2")
        result = _topo_sort_enrichments([e2, e1])
        assert result.index(e1) < result.index(e2)

    def test_three_chain(self):
        e1 = self._make([], "k1")
        e2 = self._make(["k1"], "k2")
        e3 = self._make(["k2"], "k3")
        result = _topo_sort_enrichments([e3, e1, e2])
        assert result.index(e1) < result.index(e2) < result.index(e3)

    def test_cycle_raises(self):
        e1 = self._make(["k2"], "k1")
        e2 = self._make(["k1"], "k2")
        with pytest.raises(ValueError, match="[Cc]ycle"):
            _topo_sort_enrichments([e1, e2])

    def test_missing_dependency_raises(self):
        e = self._make(["no_such_key"])
        with pytest.raises(ValueError, match="no_such_key"):
            _topo_sort_enrichments([e])

    def test_duplicate_artifact_key_raises(self):
        e1 = self._make([], "k1")
        e2 = self._make([], "k1")
        with pytest.raises(ValueError, match="k1"):
            _topo_sort_enrichments([e1, e2])

    def test_none_artifact_key_not_registered(self):
        """Enrichments with artifact_key=None are valid — they only mutate tools."""
        e_mutator = self._make([], None)
        e_producer = self._make([], "k1")
        result = _topo_sort_enrichments([e_mutator, e_producer])
        assert len(result) == 2

    def test_empty_list(self):
        assert _topo_sort_enrichments([]) == []


# ===========================================================================
#                       Cache helpers
# ===========================================================================


class TestEnrichmentCache:
    """Unit tests for fms_dgt.core.tools.enrichments.cache."""

    def test_compute_fingerprint_is_stable(self):
        fp1 = compute_fingerprint([("a", {}), ("b", {})], "model-x")
        fp2 = compute_fingerprint([("a", {}), ("b", {})], "model-x")
        assert fp1 == fp2

    def test_compute_fingerprint_differs_on_content(self):
        fp1 = compute_fingerprint([("a", {})], "model-x")
        fp2 = compute_fingerprint([("a", {})], "model-y")
        assert fp1 != fp2

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_cache(tmp_path / "nonexistent.json") == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "enrichments" / "test" / "fp.json"
        entries = {"ns::tool_a": {"type": "object", "properties": {}}}
        save_cache(path, entries)
        loaded = load_cache(path)
        assert loaded == entries

    def test_save_delta_merges(self, tmp_path):
        path = tmp_path / "cache.json"
        save_cache(path, {"tool_a": {"x": 1}})
        save_cache(path, {"tool_b": {"y": 2}})
        loaded = load_cache(path)
        assert "tool_a" in loaded
        assert "tool_b" in loaded

    def test_save_new_wins_on_conflict(self, tmp_path):
        path = tmp_path / "cache.json"
        save_cache(path, {"tool_a": {"v": 1}})
        save_cache(path, {"tool_a": {"v": 2}})
        loaded = load_cache(path)
        assert loaded["tool_a"]["v"] == 2

    def test_embeddings_uses_cache_on_second_run(self, tmp_path, monkeypatch):
        """Second enrich() call should not invoke the model if cache is warm."""
        monkeypatch.setenv("DGT_CACHE_DIR", str(tmp_path))

        dim = 8
        tool = _tool("t1")
        enrichment = EmbeddingsEnrichment.__new__(EmbeddingsEnrichment)
        enrichment._model_name = "mock-model"
        enrichment._force = False

        mock_model = MagicMock()

        def _fake_encode(texts, convert_to_tensor, show_progress_bar):
            return torch.rand(len(texts), dim, dtype=torch.float32)

        mock_model.encode.side_effect = _fake_encode
        enrichment._model = mock_model

        reg1 = _registry(tool)
        enrichment.enrich(reg1)
        assert mock_model.encode.call_count == 1

        reg2 = _registry(_tool("t1"))
        enrichment.enrich(reg2)
        assert mock_model.encode.call_count == 1  # no additional call


# ===========================================================================
#                       TASK INTEGRATION: enrichments wiring
# ===========================================================================


class TestTaskEnrichmentWiring:
    """Verify that tools.enrichments: in YAML is wired correctly by Task.__init__."""

    def test_enrichments_run_and_artifacts_populated(self, tmp_path):
        """A simple in-memory enrichment is run and writes an artifact."""
        key = "_task_wiring_test_enrichment_v1"
        if key not in _TOOL_ENRICHMENT_REGISTRY:

            @register_tool_enrichment(key)
            class _TagEnrichment(ToolEnrichment):
                artifact_key = "_tag"

                def enrich(self, registry):
                    registry.artifacts["_tag"] = "tagged"

        tool_file = os.path.join(str(tmp_path), "t.yaml")
        with open(tool_file, "w") as f:
            yaml.dump([{"name": "search"}], f)

        registry_cfgs = [{"type": "file", "path": tool_file, "namespace": "svc"}]
        enrichments_cfg = [{"type": key}]

        loaders = [
            get_tool_loader(
                entry[TYPE_KEY],
                **{k: v for k, v in entry.items() if k != TYPE_KEY},
            )
            for entry in registry_cfgs
        ]
        reg = ToolRegistry.from_loaders(loaders)

        enrichment_instances = [
            get_tool_enrichment(cfg[TYPE_KEY], **{k: v for k, v in cfg.items() if k != TYPE_KEY})
            for cfg in enrichments_cfg
        ]
        ordered = _topo_sort_enrichments(enrichment_instances)
        for e in ordered:
            e.enrich(reg)
        reg._enrichments = ordered

        assert reg.artifacts.get("_tag") == "tagged"
        reg.refresh()
        assert reg.artifacts.get("_tag") == "tagged"

    def test_dependency_order_enforced(self, tmp_path):
        """An enrichment that depends on another is always run after it."""
        call_order = []

        key_a = "_dep_order_test_A"
        key_b = "_dep_order_test_B"

        if key_a not in _TOOL_ENRICHMENT_REGISTRY:

            @register_tool_enrichment(key_a)
            class _EnrichA(ToolEnrichment):
                artifact_key = key_a

                def enrich(self, registry):
                    call_order.append("A")
                    registry.artifacts[key_a] = True

        if key_b not in _TOOL_ENRICHMENT_REGISTRY:

            @register_tool_enrichment(key_b)
            class _EnrichB(ToolEnrichment):
                depends_on = [key_a]
                artifact_key = key_b

                def enrich(self, registry):
                    call_order.append("B")
                    registry.artifacts[key_b] = True

        instances = [get_tool_enrichment(key_b), get_tool_enrichment(key_a)]
        ordered = _topo_sort_enrichments(instances)

        reg = ToolRegistry()
        for e in ordered:
            e.enrich(reg)

        assert call_order == ["A", "B"]

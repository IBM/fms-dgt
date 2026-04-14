# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the tools: -> Task.tool_registry / Task.tool_engine wiring.

Task.__init__ has heavyweight dependencies (datastores, task card, logging).
Rather than constructing a full Task, these tests:

1. Test the Phase-1 and Phase-2 resolution logic end-to-end using real loaders
   and a temp file — this is the critical path.
2. Verify the Task.tool_registry / tool_engine properties and the tools=None
   path using a minimal stub that exercises only the tool-resolution block.
"""

# Standard
from typing import Dict, List
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import yaml

# Local
from fms_dgt.base.task import Task
from fms_dgt.constants import ENGINE_KEY, TYPE_KEY
from fms_dgt.core.tools.constants import TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.engines import (
    MultiServerToolEngine,
    ToolEngine,
    get_tool_engine,
)
from fms_dgt.core.tools.loaders import get_tool_loader
from fms_dgt.core.tools.registry import ToolRegistry


def _qname(ns: str, name: str) -> str:
    return f"{ns}{TOOL_NAMESPACE_SEP}{name}"


# ---------------------------------------------------------------------------
# Phase-1 resolution logic: flat list -> registry only
# ---------------------------------------------------------------------------


class TestPhase1ResolutionLogic:
    """End-to-end test of the Phase-1 flat-list resolution path."""

    def _resolve_flat(self, tools_config, tmp_path):
        """Mimic the Task.__init__ resolution block for a flat list."""
        loaders = [
            get_tool_loader(
                entry[TYPE_KEY],
                **{k: v for k, v in entry.items() if k != TYPE_KEY},
            )
            for entry in tools_config
        ]
        return ToolRegistry.from_loaders(loaders)

    def test_single_file_loader(self, tmp_path):
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"weather_api": [{"name": "get_weather"}]}))
        config = [{"type": "file", "path": str(p)}]
        reg = self._resolve_flat(config, tmp_path)
        assert _qname("weather_api", "get_weather") in reg

    def test_multiple_file_loaders(self, tmp_path):
        p1 = tmp_path / "a.yaml"
        p2 = tmp_path / "b.yaml"
        p1.write_text(yaml.dump({"ns_a": [{"name": "foo"}]}))
        p2.write_text(yaml.dump({"ns_b": [{"name": "bar"}]}))
        config = [
            {"type": "file", "path": str(p1)},
            {"type": "file", "path": str(p2)},
        ]
        reg = self._resolve_flat(config, tmp_path)
        assert _qname("ns_a", "foo") in reg
        assert _qname("ns_b", "bar") in reg

    def test_namespace_override_in_config(self, tmp_path):
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump([{"name": "search"}]))
        config = [{"type": "file", "path": str(p), "namespace": "my_ns"}]
        reg = self._resolve_flat(config, tmp_path)
        assert _qname("my_ns", "search") in reg

    def test_unknown_loader_type_raises(self):
        config = [{"type": "nonexistent_loader_xyz"}]
        with pytest.raises(KeyError, match="not found"):
            self._resolve_flat(config, None)


# ---------------------------------------------------------------------------
# Task stub helpers
# ---------------------------------------------------------------------------


def _make_stub_task(tool_registry, tool_engine=None):
    """Build a minimal Task stub with only the tool attributes set."""
    task = object.__new__(Task)
    task._tool_registry = tool_registry
    task._tool_engine = tool_engine
    return task


# ---------------------------------------------------------------------------
# Task.tool_registry and tool_engine properties
# ---------------------------------------------------------------------------


class TestTaskToolProperties:
    def test_registry_property_returns_registry(self):
        reg = ToolRegistry(tools=[])
        task = _make_stub_task(reg)
        assert task.tool_registry is reg

    def test_registry_property_returns_none_when_no_tools(self):
        task = _make_stub_task(None)
        assert task.tool_registry is None

    def test_engine_property_returns_engine(self):

        reg = ToolRegistry(tools=[Tool(name="t", namespace="ns")])
        with patch("fms_dgt.core.tools.engines.lm.get_block") as mock_get_block:
            mock_get_block.return_value = MagicMock()
            engine = MultiServerToolEngine(registry=reg, engines={})
        task = _make_stub_task(reg, engine)
        assert task.tool_engine is engine

    def test_engine_property_returns_none_when_no_engines(self):
        task = _make_stub_task(None, None)
        assert task.tool_engine is None


# ---------------------------------------------------------------------------
# Phase-2 resolution: registry + engines wiring
# ---------------------------------------------------------------------------


class TestPhase2EngineWiring:
    """Verify that the Task wiring logic correctly builds registry + engine."""

    def _run_task_wiring(self, tools_cfg, tmp_path=None):
        """Execute the tools-wiring block from Task.__init__ in isolation."""
        # Standard

        # Normalise Phase-1 flat list into Phase-2 shape.
        if isinstance(tools_cfg, list):
            tools_cfg = {"registry": tools_cfg}

        registry_cfgs: List[Dict] = tools_cfg.get("registry", [])
        engines_cfg: Dict = tools_cfg.get("engines", {})

        for entry in registry_cfgs:
            engine_name = entry.get(ENGINE_KEY)
            if engine_name:
                if "namespace" not in entry:
                    raise ValueError(
                        f"tools.registry entry referencing engine '{engine_name}' "
                        f"must specify a namespace."
                    )
                if engine_name not in engines_cfg:
                    raise ValueError(
                        f"tools.registry entry references engine '{engine_name}' "
                        f"which is not defined in tools.engines."
                    )

        loaders = [
            get_tool_loader(
                entry[TYPE_KEY],
                **{k: v for k, v in entry.items() if k not in (TYPE_KEY, ENGINE_KEY)},
            )
            for entry in registry_cfgs
        ]
        registry = ToolRegistry.from_loaders(loaders)

        tool_engine = None
        if engines_cfg:
            engine_to_namespaces: Dict[str, List[str]] = {}
            for entry in registry_cfgs:
                engine_name = entry.get(ENGINE_KEY)
                if engine_name:
                    engine_to_namespaces.setdefault(engine_name, []).append(entry["namespace"])

            with patch("fms_dgt.core.tools.engines.lm.get_block") as mock_get_block:
                mock_get_block.return_value = MagicMock()
                built_engines: Dict[str, ToolEngine] = {
                    name: get_tool_engine(
                        cfg[TYPE_KEY],
                        registry=registry,
                        namespaces=engine_to_namespaces.get(name),
                        **{k: v for k, v in cfg.items() if k != TYPE_KEY},
                    )
                    for name, cfg in engines_cfg.items()
                }
            ns_to_engine = {
                entry["namespace"]: built_engines[entry[ENGINE_KEY]]
                for entry in registry_cfgs
                if ENGINE_KEY in entry
            }
            tool_engine = MultiServerToolEngine(registry=registry, engines=ns_to_engine)

        return registry, tool_engine

    def test_registry_only_no_engine(self, tmp_path):
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"ns": [{"name": "search"}]}))
        cfg = {
            "registry": [{"type": "file", "path": str(p)}],
        }
        reg, engine = self._run_task_wiring(cfg)
        assert _qname("ns", "search") in reg
        assert engine is None

    def test_engine_wired_to_namespace(self, tmp_path):
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"weather_api": [{"name": "get_weather"}]}))
        cfg = {
            "registry": [
                {"type": "file", "path": str(p), "namespace": "weather_api", "engine": "my_lm"},
            ],
            "engines": {
                "my_lm": {
                    "type": "lm",
                    "lm_config": {"type": "ollama", "model_id_or_path": "granite:3b"},
                },
            },
        }
        reg, engine = self._run_task_wiring(cfg)
        assert _qname("weather_api", "get_weather") in reg
        assert engine is not None
        assert isinstance(engine, MultiServerToolEngine)
        assert "weather_api" in engine.engines

    def test_two_namespaces_two_engines(self, tmp_path):
        pw = tmp_path / "weather.yaml"
        ph = tmp_path / "hr.yaml"
        pw.write_text(yaml.dump({"weather_api": [{"name": "get_weather"}]}))
        ph.write_text(yaml.dump({"hr_api": [{"name": "get_employee"}]}))
        cfg = {
            "registry": [
                {"type": "file", "path": str(pw), "namespace": "weather_api", "engine": "lm_w"},
                {"type": "file", "path": str(ph), "namespace": "hr_api", "engine": "lm_h"},
            ],
            "engines": {
                "lm_w": {"type": "lm", "lm_config": {"type": "ollama", "model_id_or_path": "m1"}},
                "lm_h": {"type": "lm", "lm_config": {"type": "ollama", "model_id_or_path": "m2"}},
            },
        }
        reg, engine = self._run_task_wiring(cfg)
        assert _qname("weather_api", "get_weather") in reg
        assert _qname("hr_api", "get_employee") in reg
        assert "weather_api" in engine.engines
        assert "hr_api" in engine.engines

    def test_undefined_engine_reference_raises(self, tmp_path):
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"ns": [{"name": "t"}]}))
        cfg = {
            "registry": [
                {"type": "file", "path": str(p), "namespace": "ns", "engine": "nonexistent"},
            ],
            "engines": {},
        }
        with pytest.raises(ValueError, match="nonexistent"):
            self._run_task_wiring(cfg)

    def test_flat_list_is_backwards_compatible(self, tmp_path):
        """Phase-1 flat list must still work after the Phase-2 refactor."""
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"ns": [{"name": "search"}]}))
        cfg = [{"type": "file", "path": str(p)}]
        reg, engine = self._run_task_wiring(cfg)
        assert _qname("ns", "search") in reg
        assert engine is None

    def test_missing_namespace_on_engine_entry_raises(self, tmp_path):
        """Registry entry that references an engine must declare a namespace."""
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump([{"name": "search"}]))
        cfg = {
            "registry": [
                # No namespace key — should raise
                {"type": "file", "path": str(p), "engine": "my_lm"},
            ],
            "engines": {
                "my_lm": {"type": "lm", "lm_config": {"type": "ollama", "model_id_or_path": "m"}},
            },
        }
        with pytest.raises(ValueError, match="must specify a namespace"):
            self._run_task_wiring(cfg)

    def test_two_namespaces_same_engine(self, tmp_path):
        """Two namespaces can share one engine; engine receives both namespaces."""
        pw = tmp_path / "weather.yaml"
        pl = tmp_path / "location.yaml"
        pw.write_text(yaml.dump({"weather_api": [{"name": "get_weather"}]}))
        pl.write_text(yaml.dump({"location_api": [{"name": "get_location"}]}))
        cfg = {
            "registry": [
                {
                    "type": "file",
                    "path": str(pw),
                    "namespace": "weather_api",
                    "engine": "shared_lm",
                },
                {
                    "type": "file",
                    "path": str(pl),
                    "namespace": "location_api",
                    "engine": "shared_lm",
                },
            ],
            "engines": {
                "shared_lm": {
                    "type": "lm",
                    "lm_config": {"type": "ollama", "model_id_or_path": "m"},
                },
            },
        }
        reg, engine = self._run_task_wiring(cfg)
        assert _qname("weather_api", "get_weather") in reg
        assert _qname("location_api", "get_location") in reg
        # Both namespaces route to the same engine instance
        assert engine.engines["weather_api"] is engine.engines["location_api"]
        # Engine knows it covers both namespaces
        assert set(engine.engines["weather_api"].namespaces) == {"weather_api", "location_api"}

    def test_engine_namespaces_property(self, tmp_path):
        """Engine constructed with namespaces exposes them via .namespaces."""
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"ns": [{"name": "t"}]}))
        cfg = {
            "registry": [
                {"type": "file", "path": str(p), "namespace": "ns", "engine": "my_lm"},
            ],
            "engines": {
                "my_lm": {"type": "lm", "lm_config": {"type": "ollama", "model_id_or_path": "m"}},
            },
        }
        _, engine = self._run_task_wiring(cfg)
        sub_engine = engine.engines["ns"]
        assert sub_engine.namespaces == ["ns"]

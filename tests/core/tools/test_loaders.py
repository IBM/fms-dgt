# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
import pytest
import yaml

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE, TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.loaders import (
    FileToolLoader,
    InlineToolLoader,
    ToolLoader,
    get_tool_loader,
    register_tool_loader,
)
from fms_dgt.core.tools.registry import ToolRegistry

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _qname(ns: str, name: str) -> str:
    return f"{ns}{TOOL_NAMESPACE_SEP}{name}"


# ---------------------------------------------------------------------------
# FileToolLoader: Shape 1 — {<namespace>: [...tools...]}
# ---------------------------------------------------------------------------


class TestShape1:
    def test_namespace_from_key(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "shape1_single.yaml")).load()
        assert len(tools) == 1
        assert tools[0].namespace == "my_api"
        assert tools[0].name == "search"

    def test_loader_arg_overrides_file_namespace(self):
        tools = FileToolLoader(
            str(TEST_DATA_DIR / "shape1_single.yaml"), namespace="override_ns"
        ).load()
        assert tools[0].namespace == "override_ns"

    def test_multiple_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "shape1_multi.yaml")).load()
        assert len(tools) == 3
        assert all(t.namespace == "api" for t in tools)
        assert {t.name for t in tools} == {"tool_a", "tool_b", "tool_c"}


# ---------------------------------------------------------------------------
# FileToolLoader: Shape 2 — bare list
# ---------------------------------------------------------------------------


class TestShape2:
    def test_yaml_default_namespace(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "shape2_list.yaml")).load()
        assert all(t.namespace == TOOL_DEFAULT_NAMESPACE for t in tools)

    def test_yaml_namespace_from_loader_arg(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "shape2_list.yaml"), namespace="my_ns").load()
        assert all(t.namespace == "my_ns" for t in tools)

    def test_json_file(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "shape2_list.json"), namespace="ns").load()
        assert tools[0].name == "lookup"
        assert tools[0].namespace == "ns"


# ---------------------------------------------------------------------------
# FileToolLoader: Shape 3 — {ToolName: {name, description, parameters}, ...}
# ---------------------------------------------------------------------------


class TestShape3:
    def test_loads_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "sgd.yaml"), namespace="sgd").load()
        assert len(tools) == 4
        assert all(t.namespace == "sgd" for t in tools)

    def test_default_namespace_without_loader_arg(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "multiwoz.yaml")).load()
        assert all(t.namespace == TOOL_DEFAULT_NAMESPACE for t in tools)

    def test_per_tool_namespace_wins_over_loader_arg(self):
        tools = FileToolLoader(
            str(TEST_DATA_DIR / "shape3_with_tool_namespace.yaml"), namespace="loader_ns"
        ).load()
        assert tools[0].namespace == "tool_ns"


# ---------------------------------------------------------------------------
# FileToolLoader: error cases (dynamic — need tmp_path or monkeypatch)
# ---------------------------------------------------------------------------


class TestFileToolLoaderErrors:
    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "tools.csv"
        p.write_text("name\nfoo")
        with pytest.raises(ValueError, match="Unsupported tool file format"):
            FileToolLoader(str(p)).load()

    def test_env_var_in_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_TOOLS_DIR", str(TEST_DATA_DIR))
        tools = FileToolLoader("${TEST_TOOLS_DIR}/shape2_list.yaml", namespace="ns").load()
        assert tools[0].name == "lookup"


# ---------------------------------------------------------------------------
# InlineToolLoader
# ---------------------------------------------------------------------------


class TestInlineToolLoader:
    _SEARCH_TOOL = {
        "name": "search_documents",
        "description": "Search for relevant document chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    def test_loads_single_tool_with_namespace(self):
        tools = InlineToolLoader(tools=[self._SEARCH_TOOL], namespace="retrieval").load()
        assert len(tools) == 1
        assert tools[0].name == "search_documents"
        assert tools[0].namespace == "retrieval"

    def test_loads_multiple_tools(self):
        tool_dicts = [
            {"name": "search_documents", "description": "Search."},
            {"name": "get_document", "description": "Fetch by ID."},
        ]
        tools = InlineToolLoader(tools=tool_dicts, namespace="retrieval").load()
        assert len(tools) == 2
        assert {t.name for t in tools} == {"search_documents", "get_document"}
        assert all(t.namespace == "retrieval" for t in tools)

    def test_default_namespace_when_omitted(self):
        tools = InlineToolLoader(tools=[self._SEARCH_TOOL]).load()
        assert tools[0].namespace == TOOL_DEFAULT_NAMESPACE

    def test_per_tool_namespace_wins_over_constructor(self):
        tool_dict = {**self._SEARCH_TOOL, "namespace": "tool_ns"}
        tools = InlineToolLoader(tools=[tool_dict], namespace="loader_ns").load()
        assert tools[0].namespace == "tool_ns"

    def test_non_list_tools_raises(self):
        with pytest.raises(TypeError, match="list of tool dicts"):
            InlineToolLoader(tools={"name": "bad"})

    def test_get_tool_loader_inline(self):
        loader = get_tool_loader("inline", tools=[self._SEARCH_TOOL], namespace="ns")
        assert isinstance(loader, InlineToolLoader)
        tools = loader.load()
        assert tools[0].name == "search_documents"
        assert tools[0].namespace == "ns"

    def test_registry_from_inline_loader(self):
        loader = InlineToolLoader(tools=[self._SEARCH_TOOL], namespace="retrieval")
        reg = ToolRegistry.from_loaders([loader])
        assert _qname("retrieval", "search_documents") in reg

    def test_mixed_inline_and_file_loaders(self):
        file_loader = FileToolLoader(str(TEST_DATA_DIR / "shape1_single.yaml"))
        inline_loader = InlineToolLoader(tools=[self._SEARCH_TOOL], namespace="retrieval")
        reg = ToolRegistry.from_loaders([file_loader, inline_loader])
        assert _qname("my_api", "search") in reg
        assert _qname("retrieval", "search_documents") in reg

    def test_parameters_preserved(self):
        tools = InlineToolLoader(tools=[self._SEARCH_TOOL], namespace="ns").load()
        assert tools[0].parameters == self._SEARCH_TOOL["parameters"]

    def test_description_preserved(self):
        tools = InlineToolLoader(tools=[self._SEARCH_TOOL], namespace="ns").load()
        assert tools[0].description == self._SEARCH_TOOL["description"]


# ---------------------------------------------------------------------------
# ToolRegistry.from_loaders and refresh
# ---------------------------------------------------------------------------


class TestFromLoadersAndRefresh:
    def test_from_loaders_single(self):
        reg = ToolRegistry.from_loaders([FileToolLoader(str(TEST_DATA_DIR / "shape1_single.yaml"))])
        assert _qname("my_api", "search") in reg

    def test_from_loaders_multi_source(self):
        reg = ToolRegistry.from_loaders(
            [
                FileToolLoader(str(TEST_DATA_DIR / "sgd.yaml"), namespace="sgd"),
                FileToolLoader(str(TEST_DATA_DIR / "multiwoz.yaml"), namespace="multiwoz"),
            ]
        )
        assert _qname("sgd", "AddEvent") in reg
        assert _qname("multiwoz", "book_hotel") in reg

    def test_refresh_picks_up_new_tools(self, tmp_path):
        # Refresh requires overwriting a file mid-test — tmp_path is appropriate here.
        p = tmp_path / "tools.yaml"
        p.write_text(yaml.dump({"ns": [{"name": "v1"}]}))
        reg = ToolRegistry.from_loaders([FileToolLoader(str(p))])
        assert _qname("ns", "v1") in reg

        p.write_text(yaml.dump({"ns": [{"name": "v2"}]}))
        reg.refresh()

        assert _qname("ns", "v2") in reg
        assert _qname("ns", "v1") not in reg

    def test_refresh_no_loaders_warns(self):
        reg = ToolRegistry(tools=[])
        with pytest.warns(UserWarning, match="no loaders are retained"):
            reg.refresh()


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------


class TestLoaderRegistry:
    def test_get_file_loader(self):
        loader = get_tool_loader(
            "file", path=str(TEST_DATA_DIR / "shape2_list.yaml"), namespace="ns"
        )
        assert isinstance(loader, FileToolLoader)

    def test_unknown_loader_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_tool_loader("nonexistent_xyz_loader")

    def test_register_custom_loader(self):
        @register_tool_loader("_test_custom_loader_xyz")
        class _CustomLoader(ToolLoader):
            def load(self):
                return []

        loader = get_tool_loader("_test_custom_loader_xyz")
        assert isinstance(loader, _CustomLoader)

    def test_duplicate_registration_raises(self):
        with pytest.raises(AssertionError, match="conflicts"):

            @register_tool_loader("file")
            class _Duplicate(ToolLoader):
                def load(self):
                    return []


# ---------------------------------------------------------------------------
# Fixture files — all four benchmark datasets (Shape 3)
# ---------------------------------------------------------------------------


class TestFixtureFiles:
    def test_sgd_loads_all_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "sgd.yaml"), namespace="sgd").load()
        assert len(tools) == 4
        assert all(t.namespace == "sgd" for t in tools)
        assert {"AddEvent", "BuyBusTicket"} <= {t.name for t in tools}

    def test_atis_loads_all_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "atis.yaml"), namespace="atis").load()
        assert len(tools) == 4
        assert all(t.namespace == "atis" for t in tools)
        assert {"atis_flight", "atis_airfare"} <= {t.name for t in tools}

    def test_multiwoz_loads_all_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "multiwoz.yaml"), namespace="multiwoz").load()
        assert len(tools) == 4
        assert all(t.namespace == "multiwoz" for t in tools)
        assert {"book_hotel", "find_train"} <= {t.name for t in tools}

    def test_glaive_loads_all_tools(self):
        tools = FileToolLoader(str(TEST_DATA_DIR / "glaive.yaml"), namespace="glaive").load()
        assert len(tools) == 4
        assert all(t.namespace == "glaive" for t in tools)
        assert {"add", "add_contact"} <= {t.name for t in tools}

    def test_all_fixture_tools_have_parameters(self):
        for fname in ("sgd.yaml", "atis.yaml", "multiwoz.yaml", "glaive.yaml"):
            tools = FileToolLoader(str(TEST_DATA_DIR / fname)).load()
            for t in tools:
                assert t.parameters, f"{fname}: tool '{t.name}' has empty parameters"

    def test_all_fixtures_load_into_single_registry(self):
        reg = ToolRegistry.from_loaders(
            [
                FileToolLoader(str(TEST_DATA_DIR / "sgd.yaml"), namespace="sgd"),
                FileToolLoader(str(TEST_DATA_DIR / "atis.yaml"), namespace="atis"),
                FileToolLoader(str(TEST_DATA_DIR / "multiwoz.yaml"), namespace="multiwoz"),
                FileToolLoader(str(TEST_DATA_DIR / "glaive.yaml"), namespace="glaive"),
            ]
        )
        assert len(reg) == 16
        assert _qname("sgd", "AddEvent") in reg
        assert _qname("atis", "atis_flight") in reg
        assert _qname("multiwoz", "book_hotel") in reg
        assert _qname("glaive", "add_contact") in reg

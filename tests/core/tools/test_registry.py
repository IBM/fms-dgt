# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
import pytest

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE, TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, namespace: str = "ns", params: dict | None = None) -> Tool:
    return Tool(name=name, namespace=namespace, parameters=params or {})


def _make_tool_dict(name: str, params: dict | None = None) -> dict:
    return {"name": name, "parameters": params or {}}


def _qname(namespace: str, name: str) -> str:
    return f"{namespace}{TOOL_NAMESPACE_SEP}{name}"


TEST_DATA_DIR = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# Schema fingerprinting
# ---------------------------------------------------------------------------


class TestSchemaFingerprint:
    def test_identical_schemas_same_fingerprint(self):
        a = {"type": "object", "properties": {"x": {"type": "integer"}}}
        b = {"type": "object", "properties": {"x": {"type": "integer"}}}
        assert schema_fingerprint(a) == schema_fingerprint(b)

    def test_different_schemas_different_fingerprint(self):
        a = {"type": "object", "properties": {"x": {"type": "integer"}}}
        b = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert schema_fingerprint(a) != schema_fingerprint(b)

    def test_key_order_irrelevant(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert schema_fingerprint(a) == schema_fingerprint(b)


# ---------------------------------------------------------------------------
# ToolRegistry: construction and query
# ---------------------------------------------------------------------------


class TestToolRegistryConstruction:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        assert not reg.all_tools()

    def test_single_tool(self):
        reg = ToolRegistry(tools=[_make_tool("search")])
        assert len(reg) == 1
        assert _qname("ns", "search") in reg

    def test_qualified_name_lookup(self):
        reg = ToolRegistry(tools=[_make_tool("search")])
        result = reg.get(_qname("ns", "search"))
        assert len(result) == 1
        assert result[0].name == "search"

    def test_qualified_name_uses_separator(self):
        t = _make_tool("search", namespace="my_api")
        assert t.qualified_name == f"my_api{TOOL_NAMESPACE_SEP}search"

    def test_get_by_name(self):
        reg = ToolRegistry(tools=[_make_tool("search", namespace="ns")])
        result = reg.get_by_name("search", namespace="ns")
        assert len(result) == 1

    def test_namespaces(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2")]
        reg = ToolRegistry(tools=tools)
        assert set(reg.namespaces()) == {"ns1", "ns2"}

    def test_tool_names_all(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2"), _make_tool("a", "ns2")]
        reg = ToolRegistry(tools=tools)
        assert set(reg.tool_names()) == {"a", "b"}

    def test_tool_names_filtered_by_namespace(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2")]
        reg = ToolRegistry(tools=tools)
        assert reg.tool_names("ns1") == ["a"]

    def test_iter(self):
        tools = [_make_tool("a"), _make_tool("b")]
        reg = ToolRegistry(tools=tools)
        assert set(t.name for t in reg) == {"a", "b"}

    def test_tool_missing_namespace_raises(self):
        # Tool.namespace has no default — dataclass enforces this at construction
        with pytest.raises(TypeError):
            Tool(name="no_ns")

    def test_repr(self):
        reg = ToolRegistry(tools=[_make_tool("search", "ns")])
        assert "ToolRegistry" in repr(reg)
        assert "ns" in repr(reg)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    def test_exact_duplicate_raises(self):
        t1 = _make_tool("search", params={"type": "object"})
        t2 = _make_tool("search", params={"type": "object"})
        with pytest.raises(ValueError, match="Duplicate tool"):
            ToolRegistry(tools=[t1, t2])

    def test_overload_allowed(self):
        t1 = _make_tool(
            "search", params={"type": "object", "properties": {"q": {"type": "string"}}}
        )
        t2 = _make_tool(
            "search", params={"type": "object", "properties": {"query": {"type": "string"}}}
        )
        reg = ToolRegistry(tools=[t1, t2])
        assert len(reg) == 2
        assert len(reg.get(_qname("ns", "search"))) == 2

    def test_same_name_different_namespaces_allowed(self):
        t1 = _make_tool("search", namespace="ns1")
        t2 = _make_tool("search", namespace="ns2")
        reg = ToolRegistry(tools=[t1, t2])
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# from_dicts factory
# ---------------------------------------------------------------------------


class TestFromDicts:
    def test_basic(self):
        dicts = [_make_tool_dict("get_weather"), _make_tool_dict("get_time")]
        reg = ToolRegistry.from_dicts(dicts, namespace="weather_api")
        assert len(reg) == 2
        assert _qname("weather_api", "get_weather") in reg

    def test_namespace_assigned(self):
        reg = ToolRegistry.from_dicts([_make_tool_dict("foo")], namespace="myns")
        assert reg.get(_qname("myns", "foo"))[0].namespace == "myns"

    def test_default_namespace(self):
        reg = ToolRegistry.from_dicts([_make_tool_dict("foo")])
        assert _qname(TOOL_DEFAULT_NAMESPACE, "foo") in reg


# ---------------------------------------------------------------------------
# from_file — Shape 1 ({<namespace>: [...tools...]})
# ---------------------------------------------------------------------------


class TestFromFileShape1:
    def test_yaml_namespace_as_key(self):
        reg = ToolRegistry.from_file(str(TEST_DATA_DIR / "shape1_single.yaml"))
        assert _qname("my_api", "search") in reg

    def test_loader_arg_overrides_file_namespace(self):
        reg = ToolRegistry.from_file(
            str(TEST_DATA_DIR / "shape1_single.yaml"), namespace="override_ns"
        )
        assert _qname("override_ns", "search") in reg
        assert _qname("my_api", "search") not in reg

    def test_json_shape1(self):
        reg = ToolRegistry.from_file(str(TEST_DATA_DIR / "shape1_single.json"))
        assert _qname("json_api", "lookup") in reg


# ---------------------------------------------------------------------------
# from_file — Shape 2 (flat list)
# ---------------------------------------------------------------------------


class TestFromFileShape2:
    def test_flat_list_defaults_to_default_namespace(self):
        reg = ToolRegistry.from_file(str(TEST_DATA_DIR / "shape2_list.yaml"))
        assert _qname(TOOL_DEFAULT_NAMESPACE, "lookup") in reg

    def test_flat_list_namespace_override(self):
        reg = ToolRegistry.from_file(
            str(TEST_DATA_DIR / "shape2_list.yaml"), namespace="override_ns"
        )
        assert _qname("override_ns", "lookup") in reg

    def test_json_flat_list(self):
        reg = ToolRegistry.from_file(str(TEST_DATA_DIR / "shape2_list.json"))
        assert _qname(TOOL_DEFAULT_NAMESPACE, "lookup") in reg

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "tools.csv"
        p.write_text("name\nfoo")
        with pytest.raises(ValueError, match="Unsupported tool file format"):
            ToolRegistry.from_file(str(p))


# ---------------------------------------------------------------------------
# ToolRegistry.match
# ---------------------------------------------------------------------------


class TestMatch:
    """Tests for ToolRegistry.match — resolving a ToolCall to a Tool definition."""

    def _make_overload(
        self, name: str, prop_names: list[str], required: list[str] | None = None
    ) -> Tool:
        """Build a tool whose parameters schema has the given property names."""
        props = {p: {"type": "string"} for p in prop_names}
        schema = {"type": "object", "properties": props, "required": required or []}
        return Tool(name=name, namespace="ns", parameters=schema)

    def test_unknown_tool_returns_none(self):
        reg = ToolRegistry()
        tc = ToolCall(name="ns::unknown", arguments={})
        assert reg.match(tc) is None

    def test_single_overload_returned_immediately(self):
        tool = self._make_overload("search", ["query", "limit"])
        reg = ToolRegistry(tools=[tool])
        tc = ToolCall(name="ns::search", arguments={"query": "cats"})
        assert reg.match(tc) is tool

    def test_multiple_overloads_validated_by_schema(self):
        # Overload A: requires "query" only
        tool_a = self._make_overload("search", ["query"], required=["query"])
        # Overload B: requires both "query" and "filter"
        tool_b = self._make_overload("search", ["query", "filter"], required=["query", "filter"])
        reg = ToolRegistry(tools=[tool_a, tool_b])

        # Call supplies "filter" — only B is valid
        tc = ToolCall(name="ns::search", arguments={"query": "cats", "filter": "recent"})
        assert reg.match(tc) is tool_b

    def test_multiple_overloads_required_field_disambiguates(self):
        # Overload A: requires "query" only
        tool_a = self._make_overload("search", ["query"], required=["query"])
        # Overload B: requires both "query" and "filter"
        tool_b = self._make_overload("search", ["query", "filter"], required=["query", "filter"])
        reg = ToolRegistry(tools=[tool_a, tool_b])

        # Call only supplies "query" — B fails validation (missing required "filter"), A passes
        tc = ToolCall(name="ns::search", arguments={"query": "cats"})
        assert reg.match(tc) is tool_a

    def test_tiebreaker_uses_key_overlap(self):
        # Both overloads are optional-only (no required fields), so both validate any dict.
        # Overload A: properties "query", "limit"
        tool_a = self._make_overload("search", ["query", "limit"])
        # Overload B: properties "query", "filter"
        tool_b = self._make_overload("search", ["query", "filter"])
        reg = ToolRegistry(tools=[tool_a, tool_b])

        # Call has "query" and "filter" — overlaps B more (2 vs 1)
        tc = ToolCall(name="ns::search", arguments={"query": "cats", "filter": "recent"})
        assert reg.match(tc) is tool_b

        # Call has "query" and "limit" — overlaps A more
        tc2 = ToolCall(name="ns::search", arguments={"query": "cats", "limit": "10"})
        assert reg.match(tc2) is tool_a

    def test_no_valid_overload_falls_back_to_key_overlap(self):
        # Both overloads require fields that the call doesn't supply.
        tool_a = self._make_overload("search", ["query", "limit"], required=["query", "limit"])
        tool_b = self._make_overload("search", ["query", "filter"], required=["query", "filter"])
        reg = ToolRegistry(tools=[tool_a, tool_b])

        # Call supplies only "query" and "filter" — neither strictly validates,
        # but B has higher key overlap.
        tc = ToolCall(name="ns::search", arguments={"query": "cats", "filter": "recent"})
        assert reg.match(tc) is tool_b

    def test_namespaces_filter_restricts_candidates(self):
        tool_a = Tool(name="search", namespace="weather_api", parameters={})
        tool_b = Tool(name="search", namespace="hr_api", parameters={})
        reg = ToolRegistry(tools=[tool_a, tool_b])

        tc_weather = ToolCall(name="weather_api::search", arguments={})
        tc_hr = ToolCall(name="hr_api::search", arguments={})

        # Scoped to weather_api only — hr call returns None
        assert reg.match(tc_weather, namespaces=["weather_api"]) is tool_a
        assert reg.match(tc_hr, namespaces=["weather_api"]) is None

        # Scoped to both — both resolve
        assert reg.match(tc_weather, namespaces=["weather_api", "hr_api"]) is tool_a
        assert reg.match(tc_hr, namespaces=["weather_api", "hr_api"]) is tool_b

    def test_namespaces_none_sees_all(self):
        tool_a = Tool(name="search", namespace="weather_api", parameters={})
        tool_b = Tool(name="search", namespace="hr_api", parameters={})
        reg = ToolRegistry(tools=[tool_a, tool_b])

        tc = ToolCall(name="weather_api::search", arguments={})
        assert reg.match(tc, namespaces=None) is tool_a


# ---------------------------------------------------------------------------
# ToolRegistry.tool_names with namespaces
# ---------------------------------------------------------------------------


class TestToolNames:
    def test_singular_namespace_filter(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2")]
        reg = ToolRegistry(tools=tools)
        assert reg.tool_names(namespace="ns1") == ["a"]

    def test_plural_namespaces_filter(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2"), _make_tool("c", "ns3")]
        reg = ToolRegistry(tools=tools)
        assert reg.tool_names(namespaces=["ns1", "ns2"]) == ["a", "b"]

    def test_namespaces_takes_precedence_over_namespace(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2")]
        reg = ToolRegistry(tools=tools)
        # namespaces wins when both supplied
        assert reg.tool_names(namespace="ns1", namespaces=["ns2"]) == ["b"]

    def test_no_filter_returns_all(self):
        tools = [_make_tool("a", "ns1"), _make_tool("b", "ns2")]
        reg = ToolRegistry(tools=tools)
        assert reg.tool_names() == ["a", "b"]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dicts_unqualified(self):
        reg = ToolRegistry(tools=[_make_tool("search", namespace="ns")])
        dicts = reg.to_dicts(qualified=False)
        assert dicts[0]["name"] == "search"
        assert dicts[0]["namespace"] == "ns"

    def test_to_dicts_qualified(self):
        reg = ToolRegistry(tools=[_make_tool("search", namespace="ns")])
        dicts = reg.to_dicts(qualified=True)
        assert dicts[0]["name"] == f"ns{TOOL_NAMESPACE_SEP}search"

    def test_round_trip_from_dicts(self):
        reg = ToolRegistry(tools=[_make_tool("search", namespace="ns")])
        dicts = reg.to_dicts(qualified=False)
        reg2 = ToolRegistry.from_dicts(dicts, namespace="ns")
        assert len(reg2) == 1
        assert _qname("ns", "search") in reg2

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MCPToolLoader and MCPToolEngine.

These tests spin up the dummy MCP SSE server in a background thread and
exercise the full load → execute → simulate cycle.

Run with::

    source .venv/bin/activate
    pytest tests/core/tools/engines/test_mcp.py -v
"""

# Standard
import json
import time

# Third Party
import pytest

# Local
from fms_dgt.core.tools.constants import TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.mcp import MCPToolEngine
from fms_dgt.core.tools.loaders.mcp import MCPToolLoader
from fms_dgt.core.tools.registry import ToolRegistry
from tests.core.tools.engines.mcp_server import run_server_in_thread

# ===========================================================================
#                       FIXTURES
# ===========================================================================

_PORT = 18765
_PORT_AUTH = 18766
_BEARER = "test-secret-token"
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_BASE_URL_AUTH = f"http://127.0.0.1:{_PORT_AUTH}"
_NS = "test_mcp"


@pytest.fixture(scope="module")
def mcp_server():
    """Start the unauthenticated MCP server for the test module."""
    handle = run_server_in_thread(port=_PORT)
    time.sleep(0.3)  # give uvicorn a moment to bind
    yield
    handle.stop()


@pytest.fixture(scope="module")
def mcp_server_auth():
    """Start the bearer-protected MCP server for the test module."""
    handle = run_server_in_thread(port=_PORT_AUTH, require_bearer=_BEARER)
    time.sleep(0.3)
    yield
    handle.stop()


@pytest.fixture(scope="module")
def registry(mcp_server):
    """Load tools from the running server into a ToolRegistry."""
    loader = MCPToolLoader(base_url=_BASE_URL, namespace=_NS)
    return ToolRegistry.from_loaders([loader])


@pytest.fixture(scope="module")
def engine(registry):
    """Build an MCPToolEngine backed by the registry."""
    return MCPToolEngine(registry)


def _call(tool: str, **kwargs) -> ToolCall:
    return ToolCall(name=f"{_NS}{TOOL_NAMESPACE_SEP}{tool}", arguments=kwargs)


# ===========================================================================
#                       LOADER TESTS
# ===========================================================================


class TestMCPToolLoader:
    def test_loads_expected_tools(self, registry):
        names = registry.tool_names(namespace=_NS)
        assert set(names) == {"get_weather", "calculator", "echo"}

    def test_tool_has_metadata(self, registry):
        tools = registry.get_by_name("get_weather", _NS)
        assert len(tools) == 1
        meta = tools[0].metadata
        assert meta["mcp_base_url"] == _BASE_URL
        assert meta["mcp_transport"] == "sse"

    def test_tool_has_input_schema(self, registry):
        tools = registry.get_by_name("calculator", _NS)
        assert len(tools) == 1
        props = tools[0].parameters.get("properties", {})
        assert "operation" in props
        assert "a" in props
        assert "b" in props

    def test_bearer_auth_loader(self, mcp_server_auth):
        loader = MCPToolLoader(
            base_url=_BASE_URL_AUTH,
            namespace="auth_ns",
            bearer_token=_BEARER,
        )
        tools = loader.load()
        assert {t.name for t in tools} == {"get_weather", "calculator", "echo"}

    def test_wrong_bearer_raises(self, mcp_server_auth):
        loader = MCPToolLoader(
            base_url=_BASE_URL_AUTH,
            namespace="auth_ns",
            bearer_token="wrong-token",
        )
        with pytest.raises(Exception):
            loader.load()


# ===========================================================================
#                       ENGINE TESTS
# ===========================================================================


class TestMCPToolEngine:
    _SESSION = "sess-mcp-1"

    @pytest.fixture(autouse=True)
    def session(self, engine):
        engine.setup(self._SESSION)
        yield
        engine.teardown(self._SESSION)

    def test_get_weather_execute(self, engine):
        results = engine.execute(self._SESSION, [_call("get_weather", city="Paris")])
        assert len(results) == 1
        r = results[0]
        assert not r.is_error
        # result is a list of content blocks
        assert isinstance(r.result, list)
        text = r.result[0].get("text", "")
        data = json.loads(text)
        assert data["city"] == "Paris"

    def test_calculator_add(self, engine):
        results = engine.execute(self._SESSION, [_call("calculator", operation="add", a=3, b=4)])
        r = results[0]
        assert not r.is_error
        data = json.loads(r.result[0]["text"])
        assert data["result"] == pytest.approx(7)

    def test_calculator_divide_by_zero(self, engine):
        results = engine.execute(self._SESSION, [_call("calculator", operation="divide", a=1, b=0)])
        r = results[0]
        # The server returns a text block with an error key, not isError=true,
        # so ToolResult.result should be set (not .error).
        assert not r.is_error
        data = json.loads(r.result[0]["text"])
        assert "error" in data

    def test_echo(self, engine):
        results = engine.execute(self._SESSION, [_call("echo", message="hello mcp")])
        r = results[0]
        assert not r.is_error
        assert r.result[0]["text"] == "hello mcp"

    def test_unknown_tool_returns_error(self, engine):
        call = ToolCall(name=f"{_NS}{TOOL_NAMESPACE_SEP}nonexistent", arguments={})
        results = engine.execute(self._SESSION, [call])
        assert results[0].is_error
        assert "nonexistent" in results[0].error

    def test_simulate_does_not_persist(self, engine):
        engine.simulate(self._SESSION, [_call("echo", message="sim")])
        # Session state should be unchanged (no history key for MCP engine base).
        state = engine.get_session_state(self._SESSION)
        assert state is not None

    def test_batch_execute(self, engine):
        results = engine.execute(
            self._SESSION,
            [
                _call("get_weather", city="Tokyo"),
                _call("calculator", operation="multiply", a=6, b=7),
                _call("echo", message="batch"),
            ],
        )
        assert len(results) == 3
        assert all(not r.is_error for r in results)

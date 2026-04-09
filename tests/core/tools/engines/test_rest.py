# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for RESTToolLoader and RESTToolEngine.

Uses the open-meteo public API (https://open-meteo.com) — no API key required,
no account, no local server.  Tests are marked ``integration`` and skipped by
default; run them explicitly::

    source .venv/bin/activate
    pytest tests/core/tools/engines/test_rest.py -v -m integration

The open-meteo OpenAPI spec is fetched from their public URL at test time.
A local fallback spec dict is also used in unit-style tests so the loader can
be tested without a network call.
"""

# Standard
from typing import Any, Dict

# Third Party
import pytest

# Local
from fms_dgt.core.tools.constants import TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.rest import RESTToolEngine
from fms_dgt.core.tools.loaders.rest import (
    REST_BASE_URL,
    REST_METHOD,
    REST_PARAM_LOCATIONS,
    REST_PATH,
    RESTToolLoader,
)
from fms_dgt.core.tools.registry import ToolRegistry

# ===========================================================================
#                       CONSTANTS
# ===========================================================================

_OPEN_METEO_BASE_URL = "https://api.open-meteo.com"
_NS = "open_meteo"

# Minimal inline spec covering the /v1/forecast endpoint.
# Used in unit-style tests to avoid a network fetch.
_INLINE_SPEC: Dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {"title": "Open-Meteo", "version": "1"},
    "servers": [{"url": "https://api.open-meteo.com"}],
    "paths": {
        "/v1/forecast": {
            "get": {
                "operationId": "get_forecast",
                "summary": "7-day weather forecast for a location.",
                "parameters": [
                    {
                        "name": "latitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "number"},
                        "description": "Latitude of the location.",
                    },
                    {
                        "name": "longitude",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "number"},
                        "description": "Longitude of the location.",
                    },
                    {
                        "name": "current_weather",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean"},
                        "description": "Include current weather conditions.",
                    },
                    {
                        "name": "temperature_2m_max",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean"},
                        "description": "Include daily max temperature at 2m.",
                    },
                ],
            }
        }
    },
}


# ===========================================================================
#                       HELPERS
# ===========================================================================


def _registry_from_inline(operation_ids=None) -> ToolRegistry:
    """Build a ToolRegistry from the inline spec (no network)."""
    # Write the spec to a temp file so RESTToolLoader can read it.
    # Standard
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as fh:
        json.dump(_INLINE_SPEC, fh)
        path = fh.name

    loader = RESTToolLoader(
        spec=path,
        namespace=_NS,
        operation_ids=operation_ids,
    )
    return ToolRegistry.from_loaders([loader])


def _call(tool: str, **kwargs) -> ToolCall:
    return ToolCall(name=f"{_NS}{TOOL_NAMESPACE_SEP}{tool}", arguments=kwargs)


# ===========================================================================
#                       LOADER UNIT TESTS (no network)
# ===========================================================================


class TestRESTToolLoaderUnit:
    def test_loads_forecast_tool(self):
        registry = _registry_from_inline()
        names = registry.tool_names(namespace=_NS)
        assert "get_forecast" in names

    def test_tool_metadata(self):
        registry = _registry_from_inline()
        tools = registry.get_by_name("get_forecast", _NS)
        assert len(tools) == 1
        meta = tools[0].metadata
        assert meta[REST_BASE_URL] == _OPEN_METEO_BASE_URL
        assert meta[REST_METHOD] == "GET"
        assert meta[REST_PATH] == "/v1/forecast"

    def test_param_locations(self):
        registry = _registry_from_inline()
        tools = registry.get_by_name("get_forecast", _NS)
        locs = tools[0].metadata[REST_PARAM_LOCATIONS]
        assert locs["latitude"] == "query"
        assert locs["longitude"] == "query"

    def test_required_params_in_schema(self):
        registry = _registry_from_inline()
        tools = registry.get_by_name("get_forecast", _NS)
        required = tools[0].parameters.get("required", [])
        assert "latitude" in required
        assert "longitude" in required

    def test_operation_id_filter(self):
        registry = _registry_from_inline(operation_ids=["nonexistent"])
        assert len(registry) == 0

    def test_engine_unknown_tool_returns_error(self):
        registry = _registry_from_inline()
        engine = RESTToolEngine(registry)
        engine.setup("s1")
        try:
            call = ToolCall(name=f"{_NS}{TOOL_NAMESPACE_SEP}missing", arguments={})
            results = engine.execute("s1", [call])
            assert results[0].is_error
            assert "missing" in results[0].error
        finally:
            engine.teardown("s1")


# ===========================================================================
#                       ENGINE INTEGRATION TESTS (network required)
# ===========================================================================


@pytest.mark.integration
class TestRESTToolEngineIntegration:
    """Live tests against https://api.open-meteo.com — skipped unless -m integration."""

    @pytest.fixture(scope="class")
    def registry(self):
        return _registry_from_inline()

    @pytest.fixture(scope="class")
    def engine(self, registry):
        return RESTToolEngine(registry)

    @pytest.fixture(autouse=True)
    def session(self, engine):
        engine.setup("live-sess")
        yield
        engine.teardown("live-sess")

    def test_forecast_london(self, engine):
        """Fetch current weather for London (lat=51.5, lon=-0.12)."""
        results = engine.execute(
            "live-sess",
            [_call("get_forecast", latitude=51.5, longitude=-0.12, current_weather=True)],
        )
        r = results[0]
        assert not r.is_error, r.error
        assert isinstance(r.result, dict)
        assert "current_weather" in r.result or "latitude" in r.result

    def test_forecast_missing_params_returns_empty(self, engine):
        """open-meteo returns 200 with empty body when lat/lon are omitted."""
        results = engine.execute("live-sess", [_call("get_forecast")])
        r = results[0]
        # open-meteo returns 200 with an empty response rather than 4xx.
        assert not r.is_error
        assert r.result == "" or r.result is None or r.result == {}

    def test_simulate_returns_result_without_persisting(self, engine):
        results = engine.simulate(
            "live-sess",
            [_call("get_forecast", latitude=48.85, longitude=2.35)],
        )
        assert len(results) == 1
        assert not results[0].is_error

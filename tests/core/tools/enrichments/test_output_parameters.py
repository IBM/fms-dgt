# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for OutputParametersEnrichment.

Coverage:
- Unit tests (mocked LM, no disk I/O): schema inference, batching, cache behaviour.
- Live test (skipped by default, requires --live): runs against a local Ollama
  instance to inspect the raw schemas produced by the prompt, helping validate
  prompt design without a full pipeline run.

To run the live tests locally:

    pytest tests/core/tools/enrichments/test_output_parameters.py --live -s

Requires Ollama running with a model that supports JSON-mode (e.g. granite4:3b).
"""

# Standard
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import json

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.output_parameters import (
    OutputParametersEnrichment,
)
from fms_dgt.core.tools.registry import ToolRegistry

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"

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
#                       Unit tests (mocked LM)
# ===========================================================================


class TestOutputParametersEnrichment:
    _DEFAULT_SCHEMA = '{"type": "object", "properties": {"temperature": {"type": "number", "description": "Celsius"}}}'

    def _make_lm_mock(self, lm_response: dict | None = None):
        """Return a MagicMock that mirrors LMProvider's pass-through behaviour.

        LMProvider echoes all non-reserved input keys onto each output dict.
        The mock simulates this by merging each input entry with the response
        content, so ``output["tool"]`` works exactly as it does in production.
        """
        mock_lm = MagicMock()
        result_content = lm_response or {"result": [{"content": self._DEFAULT_SCHEMA}]}

        def _side_effect(inputs, **kwargs):
            return [{**entry, **result_content} for entry in inputs]

        mock_lm.side_effect = _side_effect
        return mock_lm

    def _make_enrichment(self, lm_response: dict | None = None):
        """Return an OutputParametersEnrichment with a mocked LM provider.

        Cache is patched out so tests are hermetic (no disk I/O).
        """
        mock_lm = self._make_lm_mock(lm_response)
        with patch("fms_dgt.core.tools.enrichments.output_parameters.get_block") as mock_get_block:
            mock_get_block.return_value = mock_lm
            enrichment = OutputParametersEnrichment(lm_config={"type": "mock"})
        enrichment._lm = mock_lm
        # Disable cache so tests never hit or write disk.
        enrichment._force = True
        return enrichment

    def test_skips_tools_with_existing_output_params(self):
        existing_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = _tool("already_annotated", output_params=existing_schema)
        reg = _registry(tool)
        enrichment = self._make_enrichment()
        enrichment.enrich(reg)
        # LM should not have been called since tool already has output_parameters
        enrichment._lm.assert_not_called()
        assert reg.all_tools()[0].output_parameters == existing_schema

    def test_fills_missing_output_params(self):
        tool = _tool("get_weather")
        assert not tool.output_parameters
        reg = _registry(tool)
        enrichment = self._make_enrichment()
        enrichment.enrich(reg)
        enrichment._lm.assert_called_once()
        updated = reg.all_tools()[0]
        assert updated.output_parameters
        assert "properties" in updated.output_parameters

    def test_handles_unparseable_lm_output_gracefully(self):
        """If LM returns garbage, tool output_parameters stays empty."""
        tool = _tool("bad_tool")
        reg = _registry(tool)
        bad_response = {"result": [{"content": "not valid json at all {{{}"}]}
        enrichment = self._make_enrichment(lm_response=bad_response)
        enrichment.enrich(reg)
        # Tool should still have no output_parameters (empty dict = falsy)
        assert not reg.all_tools()[0].output_parameters

    def test_batches_multiple_tools(self):
        """LM is called once for all tools lacking output_parameters."""
        tool1 = _tool("t1")
        tool2 = _tool("t2")
        reg = _registry(tool1, tool2)

        schema = '{"type": "object", "properties": {"val": {"type": "string"}}}'
        result_content = {"result": [{"content": schema}]}
        mock_lm = MagicMock(
            side_effect=lambda inputs, **kw: [{**e, **result_content} for e in inputs]
        )

        with patch("fms_dgt.core.tools.enrichments.output_parameters.get_block") as mgb:
            mgb.return_value = mock_lm
            enrichment = OutputParametersEnrichment(lm_config={"type": "mock"})
        enrichment._lm = mock_lm
        enrichment._force = True
        enrichment.enrich(reg)

        # Single call, batch of 2
        assert mock_lm.call_count == 1
        call_inputs = mock_lm.call_args[0][0]
        assert len(call_inputs) == 2

    def test_flat_properties_map_wrapped(self):
        """LM returns a flat properties map without the schema envelope.

        Model output: {"field": {"description": "..."}, ...}
        Expected: wrapped into {"type": "object", "properties": {...}}.
        """
        tool = _tool("book_appointment")
        reg = _registry(tool)
        flat = (
            '{"appointmentId": {"type": "string", "description": "Unique ID"},'
            ' "status": {"type": "string", "description": "Booking status"}}'
        )
        enrichment = self._make_enrichment(lm_response={"result": [{"content": flat}]})
        enrichment.enrich(reg)
        updated = reg.all_tools()[0]
        assert updated.output_parameters
        assert updated.output_parameters.get("type") == "object"
        assert "properties" in updated.output_parameters
        assert "appointmentId" in updated.output_parameters["properties"]

    def test_wrapping_envelope_unwrapped(self):
        """LM output wrapped in {"output_parameters": {...}} is accepted."""
        tool = _tool("get_data")
        reg = _registry(tool)
        wrapped = (
            '{"output_parameters": {"type": "object", '
            '"properties": {"rows": {"type": "array"}}}}'
        )
        enrichment = self._make_enrichment(lm_response={"result": [{"content": wrapped}]})
        enrichment.enrich(reg)
        updated = reg.all_tools()[0]
        assert updated.output_parameters
        assert "properties" in updated.output_parameters

    def test_missing_lm_config_raises(self):
        with pytest.raises(AssertionError):
            OutputParametersEnrichment(lm_config=None)

    def test_lm_config_without_type_raises(self):
        with pytest.raises(AssertionError):
            OutputParametersEnrichment(lm_config={"model_id_or_path": "x"})


# ===========================================================================
#                       Cache tests (output_parameters-specific)
# ===========================================================================


class TestOutputParametersCache:
    def test_uses_cache_on_second_run(self, tmp_path, monkeypatch):
        """Second enrich() call should not invoke the LM if cache is warm."""
        monkeypatch.setenv("DGT_CACHE_DIR", str(tmp_path))

        result_content = {
            "result": [{"content": '{"type":"object","properties":{"val":{"type":"string"}}}'}]
        }
        mock_lm = MagicMock(
            side_effect=lambda inputs, **kw: [{**e, **result_content} for e in inputs]
        )

        tool = _tool("get_weather")

        with patch("fms_dgt.core.tools.enrichments.output_parameters.get_block") as mgb:
            mgb.return_value = mock_lm
            enrichment = OutputParametersEnrichment(lm_config={"type": "mock"})
        enrichment._lm = mock_lm

        # First run: LM should be called.
        reg1 = _registry(tool)
        enrichment.enrich(reg1)
        assert mock_lm.call_count == 1

        # Second run on a fresh registry with the same tool: cache should hit.
        tool2 = _tool("get_weather")
        reg2 = _registry(tool2)
        enrichment.enrich(reg2)
        assert mock_lm.call_count == 1  # no additional call

    def test_force_true_bypasses_cache(self, tmp_path, monkeypatch):
        """force=True must skip cache and call the LM even when cache exists."""
        monkeypatch.setenv("DGT_CACHE_DIR", str(tmp_path))

        result_content = {
            "result": [{"content": '{"type":"object","properties":{"x":{"type":"number"}}}'}]
        }
        mock_lm = MagicMock(
            side_effect=lambda inputs, **kw: [{**e, **result_content} for e in inputs]
        )

        with patch("fms_dgt.core.tools.enrichments.output_parameters.get_block") as mgb:
            mgb.return_value = mock_lm
            enrichment = OutputParametersEnrichment(lm_config={"type": "mock"}, force=True)
        enrichment._lm = mock_lm

        # Warm the cache with a first (non-forced) run via a separate enrichment.
        with patch("fms_dgt.core.tools.enrichments.output_parameters.get_block") as mgb2:
            mgb2.return_value = mock_lm
            warm_enrichment = OutputParametersEnrichment(lm_config={"type": "mock"})
        warm_enrichment._lm = mock_lm
        reg0 = _registry(_tool("t1"))
        warm_enrichment.enrich(reg0)
        first_call_count = mock_lm.call_count

        # Forced run must not use the cache.
        reg1 = _registry(_tool("t1"))
        enrichment.enrich(reg1)
        assert mock_lm.call_count > first_call_count


# ===========================================================================
#                       Live integration test (--live, local Ollama)
# ===========================================================================


@pytest.mark.live
class TestOutputParametersEnrichmentLive:
    """Inspect raw schemas produced against a local Ollama instance.

    Skipped by default.  To run:

        pytest tests/core/tools/enrichments/test_output_parameters.py --live -s

    Requires Ollama running locally with the model below pulled.
    Pass ``-s`` to see the printed schemas; this test is primarily a
    prompt-quality probe, not a correctness gate.
    """

    _MODEL = "granite4:3b"

    def _make_enrichment(self, force: bool = True) -> OutputParametersEnrichment:
        return OutputParametersEnrichment(
            lm_config={
                "type": "ollama",
                "model_id_or_path": self._MODEL,
                "temperature": 0.0,
            },
            force=force,
        )

    def _run_and_print(self, tools: List[Tool], label: str) -> Dict[str, Any]:
        reg = ToolRegistry(tools=tools)
        enrichment = self._make_enrichment()
        enrichment.enrich(reg)
        schemas = {t.qualified_name: t.output_parameters for t in reg.all_tools()}
        print(f"\n=== {label} ===")
        for qname, schema in schemas.items():
            print(f"  {qname}:")
            print(f"    {json.dumps(schema, indent=4)}")
        return schemas

    # ------------------------------------------------------------------
    # SGD tools (calendar / appointment domain)
    # ------------------------------------------------------------------

    def test_sgd_add_event(self):
        """AddEvent: calendar tool with date/time/location inputs.

        Expected output schema: something like {event_id, status, ...}.
        """
        tools = [
            Tool(
                name="AddEvent",
                namespace="sgd",
                description="Add event to the user's calendar",
                parameters={
                    "type": "object",
                    "properties": {
                        "event_name": {"type": "string", "description": "Title of event"},
                        "event_date": {"type": "string", "description": "Date of event"},
                        "event_location": {"type": "string", "description": "Location of event"},
                        "event_time": {"type": "string", "description": "Start time of event"},
                    },
                    "required": ["event_name", "event_date", "event_location", "event_time"],
                },
            )
        ]
        schemas = self._run_and_print(tools, "SGD AddEvent")
        schema = schemas.get("sgd::AddEvent", {})
        assert isinstance(schema, dict), "No schema produced for sgd::AddEvent"
        assert "properties" in schema, f"Schema lacks 'properties': {schema}"

    def test_sgd_book_appointment(self):
        """BookAppointment: doctor scheduling tool.

        Expected output schema: something like {appointment_id, confirmation_number, ...}.
        """
        tools = [
            Tool(
                name="BookAppointment",
                namespace="sgd",
                description="Book an appointment with a specific doctor for the given date and time",
                parameters={
                    "type": "object",
                    "properties": {
                        "doctor_name": {"type": "string", "description": "Name of the doctor"},
                        "appointment_time": {
                            "type": "string",
                            "description": "Time for the appointment",
                        },
                        "appointment_date": {
                            "type": "string",
                            "description": "Date for the appointment",
                        },
                    },
                    "required": ["doctor_name", "appointment_time", "appointment_date"],
                },
            )
        ]
        schemas = self._run_and_print(tools, "SGD BookAppointment")
        schema = schemas.get("sgd::BookAppointment", {})
        assert isinstance(schema, dict), "No schema produced for sgd::BookAppointment"
        assert "properties" in schema, f"Schema lacks 'properties': {schema}"

    # ------------------------------------------------------------------
    # MultiWOZ tools (hotel domain)
    # ------------------------------------------------------------------

    def test_multiwoz_find_hotel(self):
        """find_hotel: search tool with multiple filter criteria.

        Expected output schema: hotel-name, hotel-phone, hotel-postcode, ...
        A good prompt should produce domain-appropriate fields.
        """
        tools = [
            Tool(
                name="find_hotel",
                namespace="multiwoz",
                description="Find hotels matching area, price range, and type criteria",
                parameters={
                    "type": "object",
                    "properties": {
                        "hotel-area": {"type": "string", "description": "Area of the city"},
                        "hotel-pricerange": {"type": "string", "description": "Price range"},
                        "hotel-type": {"type": "string", "description": "Type of accommodation"},
                    },
                },
            )
        ]
        schemas = self._run_and_print(tools, "MultiWOZ find_hotel")
        schema = schemas.get("multiwoz::find_hotel", {})
        assert isinstance(schema, dict), "No schema produced for multiwoz::find_hotel"
        assert "properties" in schema, f"Schema lacks 'properties': {schema}"

    # ------------------------------------------------------------------
    # Batch: multiple tools at once
    # ------------------------------------------------------------------

    def test_batch_sgd_tools_raw_lm_output(self):
        """Diagnostic: dump raw LM output dicts for each tool in a batch.

        This bypasses _parse_schema and lets us inspect exactly what `result`
        looks like per tool — useful to debug batch vs. single discrepancies.
        """
        tool_specs = [
            ("AddEvent", "Add event to the user's calendar"),
            ("BookAppointment", "Book an appointment with a specific doctor"),
            ("FindRestaurants", "Find restaurants matching given criteria"),
            ("GetWeather", "Get current weather for a city"),
        ]
        tools = [Tool(name=name, namespace="sgd", description=desc) for name, desc in tool_specs]

        enrichment = self._make_enrichment()
        _ = ToolRegistry(tools=tools)

        # Replicate the batching logic from enrich() but intercept outputs.
        lm_inputs = [
            {
                "input": enrichment._make_messages(t),
                "gen_kwargs": {"response_format": {"type": "json_object"}},
                "tool": t,
                "task_name": "test_batch_sgd_tools_raw_lm_output",
            }
            for t in tools
        ]
        lm_outputs = enrichment._lm(lm_inputs, method=enrichment._lm.CHAT_COMPLETION)

        print("\n=== Raw LM outputs (batch) ===")
        for out in lm_outputs:
            tool = out.get("tool")
            result = out.get("result")
            print(f"\n  tool: {getattr(tool, 'name', '?')}")
            print(f"  type(result): {type(result).__name__}")
            print(
                f"  result: {json.dumps(result, indent=4) if isinstance(result, (dict, list)) else repr(result)}"
            )
            schema = enrichment._parse_schema(out, tool)
            print(f"  parsed schema: {json.dumps(schema, indent=4) if schema else None}")

        # No assertion — this is a pure diagnostic probe.

    def test_batch_sgd_tools(self):
        """Run enrichment over several tools in a single batch call.

        Validates that the LM handles multiple tools coherently and that each
        tool ends up with a distinct, non-empty schema.
        """
        tool_specs = [
            ("AddEvent", "Add event to the user's calendar"),
            ("BookAppointment", "Book an appointment with a specific doctor"),
            ("FindRestaurants", "Find restaurants matching given criteria"),
            ("GetWeather", "Get current weather for a city"),
        ]
        tools = [Tool(name=name, namespace="sgd", description=desc) for name, desc in tool_specs]
        schemas = self._run_and_print(tools, "SGD batch (4 tools)")
        assert len(schemas) == 4, f"Expected 4 schemas, got {len(schemas)}"
        for qname, schema in schemas.items():
            assert isinstance(schema, dict) and schema, f"Empty/missing schema for {qname}"
            assert "properties" in schema, f"Schema for {qname} lacks 'properties': {schema}"

    # ------------------------------------------------------------------
    # Prompt quality probe: check for domain-relevant field names
    # ------------------------------------------------------------------

    def test_get_weather_has_relevant_fields(self):
        """GetWeather output schema should mention temperature or conditions.

        This is a quality probe: if the model produces generic field names
        (e.g. 'result', 'data') the prompt may need tuning.
        """
        tools = [
            Tool(
                name="GetWeather",
                namespace="weather",
                description="Get the current weather conditions for a given city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            )
        ]
        schemas = self._run_and_print(tools, "GetWeather quality probe")
        schema = schemas.get("weather::GetWeather", {})
        assert "properties" in schema, f"Schema lacks 'properties': {schema}"

        props = schema["properties"]
        domain_words = {
            "temperature",
            "temp",
            "condition",
            "description",
            "humidity",
            "wind",
            "weather",
        }
        matched = [p for p in props if any(w in p.lower() for w in domain_words)]
        print(f"\n  Domain-relevant fields found: {matched}")
        # Soft assertion: warn but don't fail if none matched — this is a quality signal.
        if not matched:
            pytest.xfail(
                f"No domain-relevant fields in schema properties {list(props.keys())} — "
                "consider refining the prompt in output_parameters.py"
            )

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines import LMToolEngine
from fms_dgt.core.tools.registry import ToolRegistry

# Local — shared helpers
from tests.core.tools.engines.helpers import (
    _make_call,
    _make_lm_engine,
    _make_registry,
    _set_lm_response,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestLMToolEngineLifecycle:
    def test_setup_and_teardown(self):
        eng = _make_lm_engine()
        eng.setup("s1")
        assert eng.get_session_state("s1") is not None
        eng.teardown("s1")
        assert eng.get_session_state("s1") is None

    def test_duplicate_setup_raises(self):
        eng = _make_lm_engine()
        eng.setup("s1")
        with pytest.raises(ValueError, match="already active"):
            eng.setup("s1")

    def test_teardown_idempotent(self):
        eng = _make_lm_engine()
        eng.teardown("nonexistent")  # must not raise

    def test_execute_before_setup_raises(self):
        eng = _make_lm_engine()
        with pytest.raises(KeyError, match="not active"):
            eng.execute("s1", [_make_call()])

    def test_simulate_before_setup_raises(self):
        eng = _make_lm_engine()
        with pytest.raises(KeyError, match="not active"):
            eng.simulate("s1", [_make_call()])


# ---------------------------------------------------------------------------
# simulate vs execute
# ---------------------------------------------------------------------------


class TestLMToolEngineSimulateVsExecute:
    def test_simulate_does_not_modify_history(self):
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "found it"})
        eng.setup("s1")
        eng.simulate("s1", [_make_call(call_id="c1")])
        state = eng.get_session_state("s1")
        assert state["tool_executions"] == []

    def test_execute_appends_to_history(self):
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "found it"})
        eng.setup("s1")
        eng.execute("s1", [_make_call(call_id="c1")])
        state = eng.get_session_state("s1")
        assert len(state["tool_executions"]) == 1

    def test_execute_appends_error_results_too(self):
        """Failed results must still be appended — assistant must learn to handle errors."""
        eng = _make_lm_engine()
        eng._lm.return_value = [{"result": {"content": "not valid json"}, "addtl": {}}]
        eng.setup("s1")
        results = eng.execute("s1", [_make_call(call_id="c1")])
        assert results[0].is_error
        state = eng.get_session_state("s1")
        assert len(state["tool_executions"]) == 1

    def test_history_accumulates_across_executes(self):
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "r1"})
        eng.setup("s1")
        eng.execute("s1", [_make_call(call_id="c1")])
        eng.execute("s1", [_make_call(call_id="c2")])
        state = eng.get_session_state("s1")
        assert len(state["tool_executions"]) == 2

    def test_simulate_rolls_back_multi_call_batch(self):
        """simulate with multiple calls must leave history unchanged after returning."""
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "r"})
        eng.setup("s1")
        eng.simulate("s1", [_make_call(call_id="c1"), _make_call(call_id="c2")])
        state = eng.get_session_state("s1")
        assert state["tool_executions"] == []

    def test_execute_within_batch_sees_prior_results(self):
        """Within a single execute batch, call N+1 must see call N's result in history."""
        eng = _make_lm_engine()
        captured_histories = []

        original_execute_one = eng._execute_one

        def capturing_execute_one(tool_call, history):
            captured_histories.append(list(history))
            return original_execute_one(tool_call, history)

        eng._execute_one = capturing_execute_one
        _set_lm_response(eng, {"result": "r"})
        eng.setup("s1")
        eng.execute("s1", [_make_call(call_id="c1"), _make_call(call_id="c2")])

        assert len(captured_histories[0]) == 0  # c1 sees empty history
        assert len(captured_histories[1]) == 1  # c2 sees c1's result

    def test_simulate_within_batch_sees_prior_results(self):
        """Within a single simulate batch, call N+1 sees call N's result, but all rolled back."""
        eng = _make_lm_engine()
        captured_histories = []

        original_execute_one = eng._execute_one

        def capturing_execute_one(tool_call, history):
            captured_histories.append(list(history))
            return original_execute_one(tool_call, history)

        eng._execute_one = capturing_execute_one
        _set_lm_response(eng, {"result": "r"})
        eng.setup("s1")
        eng.simulate("s1", [_make_call(call_id="c1"), _make_call(call_id="c2")])

        assert len(captured_histories[0]) == 0  # c1 sees empty history
        assert len(captured_histories[1]) == 1  # c2 sees c1's result
        assert eng.get_session_state("s1")["tool_executions"] == []  # all rolled back

    def test_state_snapshot_is_independent(self):
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "r"})
        eng.setup("s1")
        eng.execute("s1", [_make_call(call_id="c1")])
        snapshot = eng.get_session_state("s1")
        eng.execute("s1", [_make_call(call_id="c2")])
        assert len(snapshot["tool_executions"]) == 1  # snapshot is a copy

    def test_fork_via_initial_state(self):
        eng = _make_lm_engine()
        _set_lm_response(eng, {"result": "r"})
        eng.setup("parent")
        eng.execute("parent", [_make_call(call_id="c1")])
        snapshot = eng.get_session_state("parent")

        eng.setup("branch_a", initial_state=snapshot)
        eng.setup("branch_b", initial_state=snapshot)
        eng.execute("branch_a", [_make_call(call_id="ca")])
        eng.execute("branch_b", [_make_call(call_id="cb")])

        history_a = eng.get_session_state("branch_a")["tool_executions"]
        history_b = eng.get_session_state("branch_b")["tool_executions"]
        assert len(history_a) == 2
        assert len(history_b) == 2
        assert history_a[1]["tool_call"]["id"] == "ca"
        assert history_b[1]["tool_call"]["id"] == "cb"


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestLMToolEngineUnknownTool:
    def test_unknown_tool_returns_error_result(self):
        eng = _make_lm_engine()
        eng.setup("s1")
        call = ToolCall(name="ns::nonexistent_tool", arguments={})
        results = eng.execute("s1", [call])
        assert results[0].is_error
        assert "Unknown tool" in results[0].error


# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------


class TestLMToolEngineErrorCategories:
    def test_network_error_fires(self):
        eng = _make_lm_engine(
            error_categories=[{"type": "network_error", "probability": 1.0, "message": "timeout"}]
        )
        eng.setup("s1")
        results = eng.execute("s1", [_make_call(call_id="c1")])
        assert results[0].is_error
        assert "timeout" in results[0].error

    def test_unparseable_result_fires(self):
        eng = _make_lm_engine(error_categories=[{"type": "unparseable_result", "probability": 1.0}])
        eng.setup("s1")
        results = eng.execute("s1", [_make_call(call_id="c1")])
        assert not results[0].is_error
        assert "garbled" in results[0].result

    def test_zero_probability_never_fires(self):
        eng = _make_lm_engine(
            error_categories=[{"type": "network_error", "probability": 0.0, "message": "timeout"}]
        )
        _set_lm_response(eng, {"result": "ok"})
        eng.setup("s1")
        results = eng.execute("s1", [_make_call(call_id="c1")])
        assert results[0].error != "timeout"

    def test_all_categories_sampled_independently(self):
        """High-probability category at index 0 must not starve others."""
        fired_types = set()
        # Run enough trials that a 0.5-probability second category fires at least once.
        eng = _make_lm_engine(
            error_categories=[
                {"type": "network_error", "probability": 1.0, "message": "net"},
                {"type": "unparseable_result", "probability": 0.5},
            ]
        )
        eng.setup("s1")
        for _ in range(20):
            results = eng.execute("s1", [_make_call()])
            if results[0].is_error:
                fired_types.add("network_error")
            else:
                fired_types.add("unparseable_result")
        # Both types must appear across 20 trials
        assert "unparseable_result" in fired_types


# ---------------------------------------------------------------------------
# Namespace scoping
# ---------------------------------------------------------------------------


class TestLMToolEngineNamespaceScoping:
    def test_engine_scoped_to_namespace_rejects_foreign_call(self):
        reg = _make_registry("weather_api")
        eng = _make_lm_engine(reg, namespaces=["hr_api"])
        eng.setup("s1")
        call = ToolCall(name="weather_api::search", arguments={})
        results = eng.execute("s1", [call])
        assert results[0].is_error
        assert "Unknown tool" in results[0].error

    def test_engine_with_no_namespace_restriction_sees_all(self):
        reg = _make_registry("weather_api")
        eng = _make_lm_engine(reg, namespaces=None)
        _set_lm_response(eng, {"result": "sunny"})
        eng.setup("s1")
        call = ToolCall(name="weather_api::search", arguments={"q": "hello"})
        results = eng.execute("s1", [call])
        assert not results[0].is_error

    def test_engine_namespaces_property(self):
        eng = _make_lm_engine(namespaces=["weather_api", "location_api"])
        assert set(eng.namespaces) == {"weather_api", "location_api"}


# ---------------------------------------------------------------------------
# Live integration test (skipped by default — run with pytest --live)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLMToolEngineLive:
    """Integration tests that spin up a real Ollama model.

    Skipped by default.  To run locally:

        pytest tests/core/tools/engines/test_lm.py --live

    Requires Ollama running locally with granite4:3b (or gpt-oss:20b) pulled.
    """

    _MODEL = "granite4:3b"
    _MODEL = "gpt-oss:20b"

    def _make_live_engine(self, yaml_file: str, namespace: str) -> LMToolEngine:
        lm_config = {
            "type": "ollama",
            "model_id_or_path": self._MODEL,
            "temperature": 0.0,
        }
        registry = ToolRegistry.from_file(str(TEST_DATA_DIR / yaml_file), namespace=namespace)
        return LMToolEngine(
            registry=registry,
            lm_config=lm_config,
            namespaces=[namespace],
        )

    def test_add_event_tool_call(self):
        """Load AddEvent from sgd.yaml and simulate a realistic tool call."""
        eng = self._make_live_engine("sgd.yaml", "sgd")
        eng.setup("live_s1")
        try:
            call = ToolCall(
                name="sgd::AddEvent",
                arguments={
                    "event_name": "Team standup",
                    "event_date": "2026-04-10",
                    "event_location": "Conference Room B",
                    "event_time": "09:00",
                },
            )
            results = eng.execute("live_s1", [call])
            assert len(results) == 1
            result = results[0]
            assert not result.is_error, f"Expected success but got error: {result.error}"
            assert isinstance(result.result, dict), "Result should be a dict"
            assert result.result, "Result dict should be non-empty"
        finally:
            eng.teardown("live_s1")

    def test_book_appointment_tool_call(self):
        """Load BookAppointment from sgd.yaml and simulate a tool call."""
        eng = self._make_live_engine("sgd.yaml", "sgd")
        eng.setup("live_s2")
        try:
            call = ToolCall(
                name="sgd::BookAppointment",
                arguments={
                    "doctor_name": "Dr. Smith",
                    "appointment_time": "14:00",
                    "appointment_date": "2026-04-15",
                },
            )
            results = eng.execute("live_s2", [call])
            assert len(results) == 1
            result = results[0]
            assert not result.is_error, f"Expected success but got error: {result.error}"
            assert isinstance(result.result, dict)
            assert result.result
        finally:
            eng.teardown("live_s2")

    def test_sequential_calls_build_history(self):
        """Two sequential tool calls — second call sees first result in prompt."""
        eng = self._make_live_engine("sgd.yaml", "sgd")
        eng.setup("live_s3")
        try:
            call1 = ToolCall(
                name="sgd::AddEvent",
                arguments={
                    "event_name": "Lunch",
                    "event_date": "2026-04-11",
                    "event_location": "Cafeteria",
                    "event_time": "12:00",
                },
            )
            call2 = ToolCall(
                name="sgd::BookAppointment",
                arguments={
                    "doctor_name": "Dr. Jones",
                    "appointment_time": "15:00",
                    "appointment_date": "2026-04-11",
                },
            )
            eng.execute("live_s3", [call1])
            results = eng.execute("live_s3", [call2])
            state = eng.get_session_state("live_s3")
            assert len(state["tool_executions"]) == 2
            assert not results[0].is_error
        finally:
            eng.teardown("live_s3")

    def test_hotel_booking_reflects_prior_search(self):
        """Book a hotel without specifying a name — the LM should carry the hotel
        name forward from the preceding find_hotel result.

        Scenario: find_hotel returns a specific hotel name for the given area and
        price range. book_hotel is then called for the same area and price range
        but with no hotel name. A coherent mock backend will echo the hotel name
        that was just found, because that is the only candidate in its simulated
        state.

        Assertion: the hotel name that appeared in the find result also appears
        somewhere in the booking result, confirming that history context
        influenced the LM's output.
        """
        eng = self._make_live_engine("multiwoz.yaml", "multiwoz")
        eng.setup("live_s4")
        try:
            find_call = ToolCall(
                name="multiwoz::find_hotel",
                arguments={
                    "hotel-area": "centre",
                    "hotel-pricerange": "cheap",
                    "hotel-type": "hotel",
                },
            )
            find_results = eng.execute("live_s4", [find_call])
            assert not find_results[0].is_error, f"find_hotel failed: {find_results[0].error}"
            assert isinstance(find_results[0].result, dict) and find_results[0].result

            # Extract whatever hotel name the LM produced for the find call.
            found_hotel_name = find_results[0].result.get("hotel-name")
            assert found_hotel_name, "find_hotel result must contain a hotel-name"

            # Book without specifying the hotel name — the LM must infer it from history.
            book_call = ToolCall(
                name="multiwoz::book_hotel",
                arguments={
                    "hotel-area": "centre",
                    "hotel-pricerange": "cheap",
                    "hotel-bookday": "monday",
                    "hotel-bookstay": 2,
                    "hotel-bookpeople": 2,
                },
            )
            book_results = eng.execute("live_s4", [book_call])
            assert not book_results[0].is_error, f"book_hotel failed: {book_results[0].error}"
            assert isinstance(book_results[0].result, dict) and book_results[0].result

            # The booking result should reference the same hotel the search returned.
            booked_hotel_name = book_results[0].result.get("hotel-name")
            assert booked_hotel_name == found_hotel_name, (
                f"Expected booking to reference '{found_hotel_name}' from prior search "
                f"but got '{booked_hotel_name}'"
            )
        finally:
            eng.teardown("live_s4")

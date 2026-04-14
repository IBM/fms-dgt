# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# Local
from fms_dgt.core.tools.engines import MultiServerToolEngine
from fms_dgt.core.tools.registry import ToolRegistry

# Local — shared helpers
from tests.core.tools.engines.helpers import (
    _make_call,
    _make_lm_engine,
    _make_registry,
    _set_lm_response,
)

# ---------------------------------------------------------------------------
# MultiServerToolEngine
# ---------------------------------------------------------------------------


class TestMultiServerToolEngine:
    def _make_multi(self):
        reg_a = _make_registry("server_a")
        reg_b = _make_registry("server_b")
        multi_reg = ToolRegistry(tools=reg_a.all_tools() + reg_b.all_tools())
        eng_a = _make_lm_engine(reg_a)
        eng_b = _make_lm_engine(reg_b)
        multi_eng = MultiServerToolEngine(
            registry=multi_reg,
            engines={"server_a": eng_a, "server_b": eng_b},
        )
        return multi_eng, eng_a, eng_b

    def test_setup_fans_out(self):
        multi, eng_a, eng_b = self._make_multi()
        multi.setup("s1")
        assert eng_a.get_session_state("s1") is not None
        assert eng_b.get_session_state("s1") is not None

    def test_teardown_fans_out(self):
        multi, eng_a, eng_b = self._make_multi()
        multi.setup("s1")
        multi.teardown("s1")
        assert eng_a.get_session_state("s1") is None
        assert eng_b.get_session_state("s1") is None

    def test_routes_by_namespace(self):
        multi, eng_a, eng_b = self._make_multi()
        _set_lm_response(eng_a, {"result": "a"})
        _set_lm_response(eng_b, {"result": "b"})
        multi.setup("s1")
        calls = [_make_call("server_a", "search", "c1"), _make_call("server_b", "search", "c2")]
        results = multi.execute("s1", calls)
        assert [r.call_id for r in results] == ["c1", "c2"]
        assert len(eng_a.get_session_state("s1")["tool_executions"]) == 1
        assert len(eng_b.get_session_state("s1")["tool_executions"]) == 1

    def test_simulate_does_not_mutate_sub_engines(self):
        multi, eng_a, eng_b = self._make_multi()
        _set_lm_response(eng_a, {"result": "a"})
        multi.setup("s1")
        multi.simulate("s1", [_make_call("server_a", "search", "c1")])
        assert eng_a.get_session_state("s1")["tool_executions"] == []

    def test_order_preserved_for_mixed_namespace_batch(self):
        multi, eng_a, eng_b = self._make_multi()
        multi.setup("s1")
        calls = [
            _make_call("server_b", "search", "b1"),
            _make_call("server_a", "search", "a1"),
            _make_call("server_b", "search", "b2"),
        ]
        results = multi.execute("s1", calls)
        assert [r.call_id for r in results] == ["b1", "a1", "b2"]

    def test_unknown_namespace_raises(self):
        multi, _, _ = self._make_multi()
        multi.setup("s1")
        with pytest.raises(ValueError, match="No engine registered for namespace"):
            multi.execute("s1", [_make_call("unknown_ns", "search")])

    def test_aggregate_session_state(self):
        multi, eng_a, _ = self._make_multi()
        _set_lm_response(eng_a, {"result": "a"})
        multi.setup("s1")
        multi.execute("s1", [_make_call("server_a", "search", "c1")])
        state = multi.get_session_state("s1")
        assert "server_a" in state
        assert len(state["server_a"]["tool_executions"]) == 1

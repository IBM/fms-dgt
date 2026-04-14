# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FanInToolSampler (tc/fan_in)."""

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint
from fms_dgt.core.tools.samplers.base import SamplingError
from fms_dgt.core.tools.samplers.fan_in import FanInToolSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, ns: str = "ns") -> Tool:
    return Tool(name=name, namespace=ns, description=f"Description of {name}.")


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _fp(tool: Tool) -> str:
    return schema_fingerprint(tool.parameters)


def _make_edge(src: Tool, score: float = 0.9) -> tuple:
    return (src.name, _fp(src), score, [])


def _fan_in_artifact(predecessors: list, sink: Tool, scores: list = None):
    """Build a dataflow_in artifact for a fan-in into sink."""
    ns = sink.namespace
    scores = scores or [0.9] * len(predecessors)
    edges = [_make_edge(t, s) for t, s in zip(predecessors, scores)]
    artifact = {sink.qualified_name: {_fp(sink): {ns: edges}}}
    for t in predecessors:
        artifact.setdefault(t.qualified_name, {_fp(t): {}})
    return artifact


# ===========================================================================
#                       Class attributes
# ===========================================================================


class TestFanInSamplerAttributes:
    def test_required_artifacts(self):
        assert FanInToolSampler.required_artifacts == ["dataflow"]

    def test_missing_artifact_raises(self):
        reg = _registry(_tool("t"))
        with pytest.raises(ValueError, match="dataflow"):
            FanInToolSampler(registry=reg, k=2)


# ===========================================================================
#                       Basic sampling
# ===========================================================================


class TestFanInSamplerBasic:
    def setup_method(self):
        self.sink = _tool("sink")
        self.preds = [_tool(f"p{i}") for i in range(4)]
        self.reg = _registry(*self.preds, self.sink)
        self.reg.artifacts["dataflow"] = {"out": {}, "in": _fan_in_artifact(self.preds, self.sink)}

    def test_returns_k_tools(self):
        sampler = FanInToolSampler(registry=self.reg, k=3)
        result = sampler.sample()
        assert len(result) == 3

    def test_sink_is_last(self):
        sampler = FanInToolSampler(registry=self.reg, k=3)
        result = sampler.sample()
        assert result[-1].name == "sink"

    def test_all_predecessors_connected_to_sink(self):
        """Every tool before sink must be a declared predecessor of sink."""
        sampler = FanInToolSampler(registry=self.reg, k=4)
        result = sampler.sample()
        rev = self.reg.artifacts["dataflow"]["in"]
        sink = result[-1]
        sink_fp = _fp(sink)
        edges = []
        for ns, elist in rev.get(sink.qualified_name, {}).get(sink_fp, {}).items():
            edges.extend(elist)
        pred_names = {e[0] for e in edges}
        for t in result[:-1]:
            assert t.name in pred_names

    def test_no_duplicates(self):
        sampler = FanInToolSampler(registry=self.reg, k=5)
        result = sampler.sample()
        names = [t.name for t in result]
        assert len(names) == len(set(names))

    def test_callsite_k_overrides_constructor(self):
        sampler = FanInToolSampler(registry=self.reg, k=2)
        result = sampler.sample(k=4)
        assert len(result) == 4

    def test_k_missing_raises(self):
        sampler = FanInToolSampler(registry=self.reg)
        with pytest.raises(ValueError, match="k"):
            sampler.sample()

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            FanInToolSampler(registry=self.reg, k=1).sample()


# ===========================================================================
#                       Insufficient predecessors
# ===========================================================================


class TestFanInSamplerInsufficientPredecessors:
    def test_raises_when_insufficient_predecessors(self):
        p1 = _tool("p1")
        sink = _tool("sink")
        reg = _registry(p1, sink)
        reg.artifacts["dataflow"] = {"out": {}, "in": _fan_in_artifact([p1], sink)}
        sampler = FanInToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        assert exc_info.value.requested == 3

    def test_sampling_error_partial_payload(self):
        sink = _tool("sink")
        reg = _registry(sink)
        reg.artifacts["dataflow"] = {"out": {}, "in": {sink.qualified_name: {_fp(sink): {}}}}
        sampler = FanInToolSampler(registry=reg, k=2)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        assert isinstance(exc_info.value.tools, list)

    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        reg.artifacts["dataflow"] = {"out": {}, "in": {}}
        sampler = FanInToolSampler(registry=reg, k=2)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       min_score filter
# ===========================================================================


class TestFanInSamplerMinScore:
    def test_min_score_filters_weak_predecessors(self):
        """With min_score=0.95, only the strong predecessor qualifies."""
        strong = _tool("strong")
        weak = _tool("weak")
        sink = _tool("sink")
        reg = _registry(strong, weak, sink)
        reg.artifacts["dataflow"] = {
            "out": {},
            "in": _fan_in_artifact([strong, weak], sink, scores=[0.99, 0.3]),
        }
        sampler = FanInToolSampler(registry=reg, k=2, min_score=0.95)
        for _ in range(20):
            result = sampler.sample()
            assert result[0].name == "strong"
            assert result[1].name == "sink"

    def test_min_score_too_high_raises(self):
        p1 = _tool("p1")
        sink = _tool("sink")
        reg = _registry(p1, sink)
        reg.artifacts["dataflow"] = {"out": {}, "in": _fan_in_artifact([p1], sink, scores=[0.5])}
        sampler = FanInToolSampler(registry=reg, k=2, min_score=0.9)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       Namespace filter
# ===========================================================================


class TestFanInSamplerNamespace:
    def test_namespace_filter(self):
        p_a = _tool("p_a", ns="api_a")
        p_b = _tool("p_b", ns="api_b")
        sink = _tool("sink", ns="api_a")
        reg = _registry(p_a, p_b, sink)
        reg.artifacts["dataflow"] = {
            "out": {},
            "in": {
                sink.qualified_name: {
                    _fp(sink): {
                        "api_a": [_make_edge(p_a)],
                        "api_b": [_make_edge(p_b)],
                    }
                },
            },
        }
        sampler = FanInToolSampler(registry=reg, k=2, namespace="api_a")
        for _ in range(20):
            result = sampler.sample()
            assert all(t.namespace == "api_a" for t in result)

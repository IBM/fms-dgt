# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FanOutToolSampler (tc/fan_out)."""

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint
from fms_dgt.core.tools.samplers.base import SamplingError
from fms_dgt.core.tools.samplers.fan_out import FanOutToolSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, ns: str = "ns") -> Tool:
    return Tool(name=name, namespace=ns, description=f"Description of {name}.")


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _fp(tool: Tool) -> str:
    return schema_fingerprint(tool.parameters)


def _make_edge(tgt: Tool, score: float = 0.9) -> tuple:
    return (tgt.name, _fp(tgt), score, [])


def _fan_out_artifact(seed: Tool, successors: list, scores: list = None):
    """Build a dataflow_out artifact for a fan-out from seed."""
    ns = seed.namespace
    scores = scores or [0.9] * len(successors)
    edges = [_make_edge(t, s) for t, s in zip(successors, scores)]
    artifact = {seed.qualified_name: {_fp(seed): {ns: edges}}}
    for t in successors:
        artifact[t.qualified_name] = {_fp(t): {}}
    return artifact


# ===========================================================================
#                       Class attributes
# ===========================================================================


class TestFanOutSamplerAttributes:
    def test_required_artifacts(self):
        assert FanOutToolSampler.required_artifacts == ["dataflow"]

    def test_missing_artifact_raises(self):
        reg = _registry(_tool("t"))
        with pytest.raises(ValueError, match="dataflow"):
            FanOutToolSampler(registry=reg, k=2)


# ===========================================================================
#                       Basic sampling
# ===========================================================================


class TestFanOutSamplerBasic:
    def setup_method(self):
        self.seed = _tool("seed")
        self.downstream = [_tool(f"d{i}") for i in range(4)]
        self.reg = _registry(self.seed, *self.downstream)
        self.reg.artifacts["dataflow"] = {
            "out": _fan_out_artifact(self.seed, self.downstream),
            "in": {},
        }

    def test_returns_k_tools(self):
        sampler = FanOutToolSampler(registry=self.reg, k=3)
        result = sampler.sample()
        assert len(result) == 3

    def test_seed_is_first(self):
        sampler = FanOutToolSampler(registry=self.reg, k=3)
        result = sampler.sample()
        assert result[0].name == "seed"

    def test_all_successors_connected_to_seed(self):
        """Every tool after seed must be a direct successor of seed."""
        sampler = FanOutToolSampler(registry=self.reg, k=4)
        result = sampler.sample()
        fwd = self.reg.artifacts["dataflow"]["out"]
        seed = result[0]
        seed_fp = _fp(seed)
        edges = []
        for ns, elist in fwd.get(seed.qualified_name, {}).get(seed_fp, {}).items():
            edges.extend(elist)
        successor_names = {e[0] for e in edges}
        for t in result[1:]:
            assert t.name in successor_names

    def test_no_duplicates(self):
        sampler = FanOutToolSampler(registry=self.reg, k=5)
        result = sampler.sample()
        names = [t.name for t in result]
        assert len(names) == len(set(names))

    def test_callsite_k_overrides_constructor(self):
        sampler = FanOutToolSampler(registry=self.reg, k=2)
        result = sampler.sample(k=4)
        assert len(result) == 4

    def test_k_missing_raises(self):
        sampler = FanOutToolSampler(registry=self.reg)
        with pytest.raises(ValueError, match="k"):
            sampler.sample()

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            FanOutToolSampler(registry=self.reg, k=1).sample()


# ===========================================================================
#                       Insufficient successors
# ===========================================================================


class TestFanOutSamplerInsufficientSuccessors:
    def test_raises_when_insufficient_successors(self):
        seed = _tool("seed")
        d1 = _tool("d1")
        reg = _registry(seed, d1)
        reg.artifacts["dataflow"] = {"out": _fan_out_artifact(seed, [d1]), "in": {}}
        sampler = FanOutToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        assert exc_info.value.requested == 3

    def test_sampling_error_partial_payload(self):
        seed = _tool("seed")
        reg = _registry(seed)
        reg.artifacts["dataflow"] = {"out": {seed.qualified_name: {_fp(seed): {}}}, "in": {}}
        sampler = FanOutToolSampler(registry=reg, k=2)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        # tools may be empty or partial — just check it's a list.
        assert isinstance(exc_info.value.tools, list)

    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        reg.artifacts["dataflow"] = {"out": {}, "in": {}}
        sampler = FanOutToolSampler(registry=reg, k=2)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       min_score filter
# ===========================================================================


class TestFanOutSamplerMinScore:
    def test_min_score_filters_weak_successors(self):
        """With min_score=0.95, weak successors are excluded."""
        seed = _tool("seed")
        strong = _tool("strong")
        weak = _tool("weak")
        reg = _registry(seed, strong, weak)
        reg.artifacts["dataflow"] = {
            "out": _fan_out_artifact(seed, [strong, weak], scores=[0.99, 0.3]),
            "in": {},
        }
        sampler = FanOutToolSampler(registry=reg, k=2, min_score=0.95)
        for _ in range(20):
            result = sampler.sample()
            assert result[1].name == "strong"

    def test_min_score_too_high_raises(self):
        seed = _tool("seed")
        d1 = _tool("d1")
        reg = _registry(seed, d1)
        reg.artifacts["dataflow"] = {"out": _fan_out_artifact(seed, [d1], scores=[0.5]), "in": {}}
        sampler = FanOutToolSampler(registry=reg, k=2, min_score=0.9)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       Namespace filter
# ===========================================================================


class TestFanOutSamplerNamespace:
    def test_namespace_filter(self):
        seed = _tool("seed", ns="api_a")
        d_a = _tool("d_a", ns="api_a")
        d_b = _tool("d_b", ns="api_b")
        reg = _registry(seed, d_a, d_b)
        reg.artifacts["dataflow"] = {
            "out": {
                seed.qualified_name: {
                    _fp(seed): {
                        "api_a": [_make_edge(d_a)],
                        "api_b": [_make_edge(d_b)],
                    }
                },
                d_a.qualified_name: {_fp(d_a): {}},
                d_b.qualified_name: {_fp(d_b): {}},
            },
            "in": {},
        }
        sampler = FanOutToolSampler(registry=reg, k=2, namespace="api_a")
        for _ in range(20):
            result = sampler.sample()
            assert all(t.namespace == "api_a" for t in result)

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChainToolSampler (tc/chain)."""

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint
from fms_dgt.core.tools.samplers.base import SamplingError
from fms_dgt.core.tools.samplers.chain import ChainToolSampler

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


def _chain_artifact(tools_chain):
    """Build a dataflow_out artifact for a linear chain A→B→C→…

    Each tool in the list can flow into the next.  The last tool loops back
    to the first so that the sampler never dead-ends regardless of which seed
    it picks — tests that need a true dead end build their own artifact.
    """
    artifact = {}
    ns = tools_chain[0].namespace
    n = len(tools_chain)
    for i, src in enumerate(tools_chain):
        tgt = tools_chain[(i + 1) % n]  # ring: last wraps to first
        artifact[src.qualified_name] = {_fp(src): {ns: [_make_edge(tgt)]}}
    return artifact


# ===========================================================================
#                       Class attributes
# ===========================================================================


class TestChainSamplerAttributes:
    def test_required_artifacts(self):
        assert ChainToolSampler.required_artifacts == ["dataflow"]

    def test_missing_artifact_raises(self):
        reg = _registry(_tool("t"))
        with pytest.raises(ValueError, match="dataflow"):
            ChainToolSampler(registry=reg, k=2)


# ===========================================================================
#                       Basic sampling
# ===========================================================================


class TestChainSamplerBasic:
    def setup_method(self):
        self.tools = [_tool(f"t{i}") for i in range(4)]
        self.reg = _registry(*self.tools)
        self.reg.artifacts["dataflow"] = {"out": _chain_artifact(self.tools), "in": {}}

    def test_returns_k_tools(self):
        sampler = ChainToolSampler(registry=self.reg, k=3)
        result = sampler.sample()
        assert len(result) == 3

    def test_returns_tool_objects(self):
        sampler = ChainToolSampler(registry=self.reg, k=2)
        result = sampler.sample()
        assert all(isinstance(t, Tool) for t in result)

    def test_chain_ordering_valid(self):
        """Each tool in the chain must be a valid successor of the previous."""
        sampler = ChainToolSampler(registry=self.reg, k=4)
        result = sampler.sample()
        fwd = self.reg.artifacts["dataflow"]["out"]
        for i in range(len(result) - 1):
            src = result[i]
            tgt = result[i + 1]
            src_fp = _fp(src)
            edges = []
            for ns, elist in fwd.get(src.qualified_name, {}).get(src_fp, {}).items():
                edges.extend(elist)
            successor_names = [e[0] for e in edges]
            assert tgt.name in successor_names, f"{tgt.name} is not a valid successor of {src.name}"

    def test_no_duplicates_in_chain(self):
        sampler = ChainToolSampler(registry=self.reg, k=4)
        result = sampler.sample()
        names = [t.name for t in result]
        assert len(names) == len(set(names))

    def test_k_missing_raises(self):
        sampler = ChainToolSampler(registry=self.reg)
        with pytest.raises(ValueError, match="k"):
            sampler.sample()

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ChainToolSampler(registry=self.reg, k=0).sample()

    def test_callsite_k_overrides_constructor(self):
        sampler = ChainToolSampler(registry=self.reg, k=2)
        result = sampler.sample(k=3)
        assert len(result) == 3


# ===========================================================================
#                       Dead-end / SamplingError
# ===========================================================================


class TestChainSamplerDeadEnd:
    def test_raises_sampling_error_when_dead_end(self):
        """Chain of 2 tools: requesting k=3 must dead-end."""
        t1, t2 = _tool("a"), _tool("b")
        reg = _registry(t1, t2)
        reg.artifacts["dataflow"] = {"out": _chain_artifact([t1, t2]), "in": {}}
        sampler = ChainToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        assert exc_info.value.requested == 3

    def test_sampling_error_carries_partial_chain(self):
        t1, t2 = _tool("a"), _tool("b")
        reg = _registry(t1, t2)
        reg.artifacts["dataflow"] = {"out": _chain_artifact([t1, t2]), "in": {}}
        sampler = ChainToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError) as exc_info:
            sampler.sample()
        # Partial chain should have at least 1 tool (the seed).
        assert len(exc_info.value.tools) >= 1

    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        reg.artifacts["dataflow"] = {"out": {}, "in": {}}
        sampler = ChainToolSampler(registry=reg, k=2)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       min_score filter
# ===========================================================================


class TestChainSamplerMinScore:
    def test_min_score_filters_weak_edges(self):
        """With min_score=0.95, only strong edges (>=0.95) qualify.

        Ring structure so no tool dead-ends regardless of seed:
          a→b (0.99), a→c (0.5)  — from a, only b qualifies
          b→c (0.99)
          c→a (0.99)
        After applying min_score=0.95, every tool has at least one outgoing
        qualifying edge, so the sampler never dead-ends.
        We verify that from a, only b is ever chosen (c is filtered out).
        """
        t1, t2, t3 = _tool("a"), _tool("b"), _tool("c")
        reg = _registry(t1, t2, t3)
        ns = t1.namespace
        reg.artifacts["dataflow"] = {
            "out": {
                t1.qualified_name: {_fp(t1): {ns: [_make_edge(t2, 0.99), _make_edge(t3, 0.5)]}},
                t2.qualified_name: {_fp(t2): {ns: [_make_edge(t3, 0.99)]}},
                t3.qualified_name: {_fp(t3): {ns: [_make_edge(t1, 0.99)]}},
            },
            "in": {},
        }
        sampler = ChainToolSampler(registry=reg, k=2, min_score=0.95)
        # Over many draws: whenever seed is a, successor must be b (not c).
        for _ in range(30):
            result = sampler.sample()
            if result[0].name == "a":
                assert result[1].name == "b", "weak edge a→c should be filtered"

    def test_min_score_causes_dead_end_raises(self):
        t1, t2 = _tool("a"), _tool("b")
        reg = _registry(t1, t2)
        ns = t1.namespace
        # Only weak edge.
        reg.artifacts["dataflow"] = {
            "out": {
                t1.qualified_name: {_fp(t1): {ns: [_make_edge(t2, 0.3)]}},
                t2.qualified_name: {_fp(t2): {}},
            },
            "in": {},
        }
        sampler = ChainToolSampler(registry=reg, k=2, min_score=0.9)
        with pytest.raises(SamplingError):
            sampler.sample()


# ===========================================================================
#                       Namespace filter
# ===========================================================================


class TestChainSamplerNamespace:
    def test_namespace_filter_restricts_seed_and_successors(self):
        t_a1 = _tool("a1", ns="api_a")
        t_a2 = _tool("a2", ns="api_a")
        t_b1 = _tool("b1", ns="api_b")
        reg = _registry(t_a1, t_a2, t_b1)
        # Ring within api_a (a1→a2→a1) plus cross-ns edge a1→b1.
        # With namespace="api_a" the cross-ns edge is ignored.
        reg.artifacts["dataflow"] = {
            "out": {
                t_a1.qualified_name: {
                    _fp(t_a1): {
                        "api_a": [_make_edge(t_a2)],
                        "api_b": [_make_edge(t_b1)],
                    }
                },
                t_a2.qualified_name: {_fp(t_a2): {"api_a": [_make_edge(t_a1)]}},
                t_b1.qualified_name: {_fp(t_b1): {}},
            },
            "in": {},
        }
        sampler = ChainToolSampler(registry=reg, k=2, namespace="api_a")
        for _ in range(20):
            result = sampler.sample()
            assert all(t.namespace == "api_a" for t in result)

    def test_callsite_namespace_overrides_constructor(self):
        t1 = _tool("a", ns="api_a")
        t2 = _tool("b", ns="api_a")
        reg = _registry(t1, t2)
        # Ring so both tools always have a successor.
        reg.artifacts["dataflow"] = {
            "out": {
                t1.qualified_name: {_fp(t1): {"api_a": [_make_edge(t2)]}},
                t2.qualified_name: {_fp(t2): {"api_a": [_make_edge(t1)]}},
            },
            "in": {},
        }
        sampler = ChainToolSampler(registry=reg, k=2, namespace="api_a")
        result = sampler.sample(namespace="api_a")
        assert all(t.namespace == "api_a" for t in result)

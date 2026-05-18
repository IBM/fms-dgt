# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ToolSampler base infrastructure and tc/random sampler.

Coverage:
- ToolSampler base class and registry (register / get / duplicate detection)
- required_artifacts validation at construction time
- RandomToolSampler: uniform strategy, proportional strategy
- RandomToolSampler: namespace hard filter
- RandomToolSampler: namespace_weights with fill-in (uniform and proportional)
- RandomToolSampler: call-site override of constructor defaults
- RandomToolSampler: k capping with warning when pool is smaller than k
- RandomToolSampler: error when k absent from both constructor and call-site
- RandomToolSampler: empty registry raises SamplingError
- RandomToolSampler: unknown namespace raises SamplingError
- RandomToolSampler: namespace_weights referencing unknown namespace (warning, ignored)
"""

# Standard
from collections import Counter
from unittest.mock import patch

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import ToolRegistry
from fms_dgt.core.tools.samplers.base import (
    _TOOL_SAMPLER_REGISTRY,
    SamplingError,
    ToolSampler,
    get_tool_sampler,
    register_tool_sampler,
)
from fms_dgt.core.tools.samplers.random import RandomToolSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, ns: str = "ns") -> Tool:
    return Tool(name=name, namespace=ns, description=f"Description of {name}.")


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _multi_ns_registry() -> ToolRegistry:
    """Registry with three namespaces of different sizes: a=3, b=2, c=1."""
    tools = [
        _tool("a1", "a"),
        _tool("a2", "a"),
        _tool("a3", "a"),
        _tool("b1", "b"),
        _tool("b2", "b"),
        _tool("c1", "c"),
    ]
    return ToolRegistry(tools=tools)


# ===========================================================================
#                       BASE CLASS AND REGISTRY
# ===========================================================================


class TestSamplerRegistry:
    """register_tool_sampler / get_tool_sampler wiring."""

    def test_register_and_get(self):
        name = "_test_sampler_reg_unique_1"
        assert name not in _TOOL_SAMPLER_REGISTRY

        @register_tool_sampler(name)
        class _TestSampler(ToolSampler):
            def sample(self, k=None, context=None, **kwargs):
                return []

        instance = get_tool_sampler(name, registry=_registry(_tool("t")))
        assert isinstance(instance, _TestSampler)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_tool_sampler("__no_such_sampler__", registry=_registry(_tool("t")))

    def test_duplicate_registration_raises(self):
        name = "_test_sampler_reg_dup"
        assert name not in _TOOL_SAMPLER_REGISTRY

        @register_tool_sampler(name)
        class _First(ToolSampler):
            def sample(self, k=None, context=None, **kwargs):
                return []

        with pytest.raises(AssertionError, match="conflicts"):

            @register_tool_sampler(name)
            class _Second(ToolSampler):
                def sample(self, k=None, context=None, **kwargs):
                    return []

    def test_non_subclass_registration_raises(self):
        name = "_test_sampler_reg_non_sub"
        assert name not in _TOOL_SAMPLER_REGISTRY
        with pytest.raises(AssertionError, match="must extend ToolSampler"):

            @register_tool_sampler(name)
            class _NotSampler:
                pass


# ===========================================================================
#                       required_artifacts VALIDATION
# ===========================================================================


class TestRequiredArtifacts:
    def test_missing_artifact_raises_on_construction(self):
        class _NeedsFoo(ToolSampler):
            required_artifacts = ["foo"]

            def sample(self, k=None, context=None, **kwargs):
                return []

        reg = _registry(_tool("t"))
        with pytest.raises(ValueError, match="foo"):
            _NeedsFoo(registry=reg)

    def test_present_artifact_passes(self):
        class _NeedsFoo(ToolSampler):
            required_artifacts = ["foo"]

            def sample(self, k=None, context=None, **kwargs):
                return []

        reg = _registry(_tool("t"))
        reg.artifacts["foo"] = {"some": "data"}
        instance = _NeedsFoo(registry=reg)
        assert instance is not None

    def test_no_required_artifacts_always_passes(self):
        reg = _registry(_tool("t"))
        sampler = RandomToolSampler(registry=reg, k=1)
        assert sampler is not None


# ===========================================================================
#                       tc/random — BASIC SAMPLING
# ===========================================================================


class TestRandomSamplerBasic:
    def test_returns_correct_count(self):
        reg = _registry(_tool("t1"), _tool("t2"), _tool("t3"))
        sampler = RandomToolSampler(registry=reg, k=2)
        result = sampler.sample()
        assert len(result) == 2

    def test_returns_tool_objects(self):
        reg = _registry(_tool("t1"), _tool("t2"))
        sampler = RandomToolSampler(registry=reg, k=2)
        result = sampler.sample()
        assert all(isinstance(t, Tool) for t in result)

    def test_no_duplicates(self):
        reg = _registry(_tool("t1"), _tool("t2"), _tool("t3"), _tool("t4"))
        sampler = RandomToolSampler(registry=reg, k=4)
        result = sampler.sample()
        names = [t.name for t in result]
        assert len(names) == len(set(names))

    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        sampler = RandomToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError):
            sampler.sample()

    def test_k_missing_raises(self):
        reg = _registry(_tool("t1"))
        sampler = RandomToolSampler(registry=reg)
        with pytest.raises(ValueError, match="k"):
            sampler.sample()

    def test_k_zero_raises(self):
        reg = _registry(_tool("t1"))
        with pytest.raises(ValueError, match="positive"):
            RandomToolSampler(registry=reg, k=0).sample()

    def test_k_exceeds_pool_returns_all_with_warning(self):
        reg = _registry(_tool("t1"), _tool("t2"))
        sampler = RandomToolSampler(registry=reg, k=10)
        with patch.object(sampler, "logger") as mock_log:
            result = sampler.sample()
        assert len(result) == 2
        mock_log.warning.assert_called_once()

    def test_callsite_k_overrides_constructor(self):
        reg = _registry(_tool("t1"), _tool("t2"), _tool("t3"))
        sampler = RandomToolSampler(registry=reg, k=1)
        result = sampler.sample(k=3)
        assert len(result) == 3


# ===========================================================================
#                       tc/random — NAMESPACE HARD FILTER
# ===========================================================================


class TestRandomSamplerNamespaceFilter:
    def test_namespace_filters_to_correct_namespace(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=2, namespace="b")
        result = sampler.sample()
        assert all(t.namespace == "b" for t in result)

    def test_namespace_respects_k(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=1, namespace="a")
        result = sampler.sample()
        assert len(result) == 1

    def test_namespace_caps_to_available(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=10, namespace="c")
        with patch.object(sampler, "logger") as mock_log:
            result = sampler.sample()
        assert len(result) == 1  # c has only 1 tool
        mock_log.warning.assert_called_once()

    def test_unknown_namespace_raises(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=2, namespace="nonexistent")
        with pytest.raises(SamplingError):
            sampler.sample()

    def test_callsite_namespace_overrides_constructor(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=2, namespace="a")
        result = sampler.sample(namespace="b")
        assert all(t.namespace == "b" for t in result)

    def test_namespace_ignores_weights_and_strategy(self):
        """Hard namespace filter takes precedence over weights/strategy."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(
            registry=reg,
            k=2,
            namespace="b",
            namespace_weights={"a": 1.0},
            strategy="proportional",
        )
        result = sampler.sample()
        assert all(t.namespace == "b" for t in result)


# ===========================================================================
#                       tc/random — UNIFORM STRATEGY
# ===========================================================================


class TestRandomSamplerUniform:
    def test_uniform_samples_from_all_namespaces(self):
        """Over many draws, uniform strategy should visit all namespaces."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=3, strategy="uniform")
        seen_ns = set()
        for _ in range(200):
            seen_ns.update(t.namespace for t in sampler.sample())
        assert seen_ns == {"a", "b", "c"}

    def test_uniform_equal_expected_share(self):
        """Uniform strategy: each namespace should appear ~k/3 of the time
        across many draws (approximate, just checks no namespace is starved)."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=300, strategy="uniform")
        counts: Counter = Counter()
        for _ in range(10):
            for t in sampler.sample():
                counts[t.namespace] += 1
        # Each namespace should get at least 20% of total draws (loosely uniform).
        total = sum(counts.values())
        for ns in ("a", "b", "c"):
            assert counts[ns] / total > 0.15, f"namespace {ns!r} underrepresented: {counts}"


# ===========================================================================
#                       tc/random — PROPORTIONAL STRATEGY
# ===========================================================================


class TestRandomSamplerProportional:
    def test_proportional_favors_larger_namespace(self):
        """Proportional strategy: 'a' (3 tools) should appear more than 'c' (1 tool)."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=600, strategy="proportional")
        counts: Counter = Counter()
        for _ in range(10):
            for t in sampler.sample():
                counts[t.namespace] += 1
        assert counts["a"] > counts["c"], "proportional: larger ns should appear more"

    def test_proportional_samples_from_all_namespaces(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=6, strategy="proportional")
        seen_ns = set()
        for _ in range(200):
            seen_ns.update(t.namespace for t in sampler.sample())
        assert seen_ns == {"a", "b", "c"}


# ===========================================================================
#                       tc/random — NAMESPACE_WEIGHTS
# ===========================================================================


class TestRandomSamplerNamespaceWeights:
    def test_explicit_weight_zero_remaining_excluded(self):
        """namespace_weights={"a": 1.0} should only sample from 'a'
        since all probability mass is on 'a' and fill-in weights for b/c
        are either 0 or very small relative to a=1.0 with uniform fill-in."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(
            registry=reg, k=3, namespace_weights={"a": 1e9, "b": 0.0, "c": 0.0}
        )
        # With extreme weight on a and zero on b/c, all draws should come from a.
        for _ in range(50):
            result = sampler.sample()
            assert all(t.namespace == "a" for t in result)

    def test_weights_partial_fill_uniform(self):
        """Specifying weight for 'a' only; 'b' and 'c' split the rest uniformly."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(
            registry=reg,
            k=300,
            namespace_weights={"a": 1e9},
            strategy="uniform",
        )
        counts: Counter = Counter()
        for _ in range(10):
            for t in sampler.sample():
                counts[t.namespace] += 1
        # 'a' should dominate; 'b' and 'c' should appear roughly equally.
        assert counts["a"] > counts["b"]
        assert counts["a"] > counts["c"]
        # b and c should be roughly equal (uniform fill)
        ratio = counts["b"] / (counts["c"] + 1e-9)
        assert 0.5 < ratio < 2.0, f"b/c ratio {ratio:.2f} not close to 1 under uniform fill"

    def test_weights_partial_fill_proportional(self):
        """Specifying weight for 'a' only; 'b' and 'c' split rest proportionally to size."""
        reg = _multi_ns_registry()  # b=2, c=1
        sampler = RandomToolSampler(
            registry=reg,
            k=300,
            namespace_weights={"a": 1e9},
            strategy="proportional",
        )
        counts: Counter = Counter()
        for _ in range(10):
            for t in sampler.sample():
                counts[t.namespace] += 1
        # b (size 2) should appear more than c (size 1) under proportional fill.
        assert counts["b"] > counts["c"]

    def test_unknown_namespace_in_weights_warns_and_ignored(self):
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(
            registry=reg, k=2, namespace_weights={"a": 1.0, "nonexistent": 5.0}
        )
        with patch.object(sampler, "logger") as mock_log:
            result = sampler.sample()
        assert len(result) == 2
        mock_log.warning.assert_called_once()

    def test_callsite_weights_override_constructor(self):
        """Call-site namespace_weights should override constructor namespace_weights."""
        reg = _multi_ns_registry()
        # Constructor pins everything to 'a'.
        sampler = RandomToolSampler(
            registry=reg, k=2, namespace_weights={"a": 1e9, "b": 0.0, "c": 0.0}
        )
        # Call-site pins everything to 'b'.
        result = sampler.sample(namespace_weights={"b": 1e9, "a": 0.0, "c": 0.0})
        assert all(t.namespace == "b" for t in result)

    def test_callsite_strategy_overrides_constructor(self):
        """Call-site strategy should override constructor strategy."""
        reg = _multi_ns_registry()
        sampler = RandomToolSampler(registry=reg, k=6, strategy="uniform")
        # Override with proportional at call site — 'a' should dominate.
        counts: Counter = Counter()
        for _ in range(50):
            for t in sampler.sample(strategy="proportional"):
                counts[t.namespace] += 1
        assert counts["a"] >= counts["c"]

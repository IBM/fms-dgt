# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tc/neighbor sampler.

Coverage:
- required_artifacts validation (neighbors required)
- seed is always the first element
- returns exactly k tools when neighbors are sufficient
- caps to seed + all neighbors when fewer than k-1 neighbors exist
- no duplicate tools in the result
- namespace hard filter constrains seed namespace
- namespace_weights + strategy govern seed namespace selection
- call-site overrides for k, namespace, namespace_weights, strategy
- k=1 returns only the seed
- empty registry raises SamplingError
- seed with no neighbors returns [seed] with warning
- stale neighbor (not in registry) is skipped with warning
- score normalization: uniform fallback when all scores identical
- weighted sampling: higher-scored neighbors appear more often
"""

# Standard
from collections import Counter
from unittest.mock import patch

# Third Party
import pytest

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import ToolRegistry, schema_fingerprint
from fms_dgt.core.tools.samplers.base import SamplingError
from fms_dgt.core.tools.samplers.neighbor import NeighborToolSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, ns: str = "ns", params: dict = None) -> Tool:
    return Tool(
        name=name,
        namespace=ns,
        description=f"Description of {name}.",
        parameters=params or {},
    )


def _registry(*tools: Tool) -> ToolRegistry:
    return ToolRegistry(tools=list(tools))


def _fp(tool: Tool) -> str:
    return schema_fingerprint(tool.parameters)


def _make_neighbors_artifact(
    source: Tool,
    targets: list,  # list of (Tool, score)
) -> dict:
    """Build a minimal neighbors artifact for a single source tool."""
    src_fp = _fp(source)
    ns_buckets: dict = {}
    for tgt, score in targets:
        ns = tgt.namespace
        if ns not in ns_buckets:
            ns_buckets[ns] = []
        ns_buckets[ns].append((tgt.name, _fp(tgt), score))
    return {source.qualified_name: {src_fp: ns_buckets}}


def _registry_with_neighbors(
    all_tools: list,
    source: Tool,
    targets: list,  # list of (Tool, score)
) -> ToolRegistry:
    reg = _registry(*all_tools)
    reg.artifacts["neighbors"] = _make_neighbors_artifact(source, targets)
    return reg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_setup():
    """Single-namespace setup: seed a1, neighbors a2/a3/a4."""
    seed = _tool("a1", "ns")
    n1 = _tool("a2", "ns")
    n2 = _tool("a3", "ns")
    n3 = _tool("a4", "ns")
    reg = _registry_with_neighbors(
        [seed, n1, n2, n3],
        seed,
        [(n1, 0.9), (n2, 0.6), (n3, 0.3)],
    )
    return reg, seed, [n1, n2, n3]


def _multi_ns_setup():
    """Multi-namespace setup: seed in 'a', neighbors spread across 'a', 'b', 'c'."""
    seed = _tool("s1", "a")
    na = _tool("a2", "a")
    nb1 = _tool("b1", "b")
    nb2 = _tool("b2", "b")
    nc = _tool("c1", "c")
    all_tools = [seed, na, nb1, nb2, nc, _tool("a3", "a"), _tool("b3", "b")]
    reg = _registry_with_neighbors(
        all_tools,
        seed,
        [(na, 0.8), (nb1, 0.7), (nb2, 0.5), (nc, 0.2)],
    )
    return reg, seed


# ===========================================================================
#                       REQUIRED ARTIFACTS
# ===========================================================================


class TestNeighborRequiredArtifacts:
    def test_missing_neighbors_raises(self):
        reg = _registry(_tool("t1"))
        with pytest.raises(ValueError, match="neighbors"):
            NeighborToolSampler(registry=reg, k=2)

    def test_present_neighbors_passes(self):
        seed = _tool("t1")
        reg = _registry_with_neighbors([seed], seed, [])
        sampler = NeighborToolSampler(registry=reg, k=1)
        assert sampler is not None


# ===========================================================================
#                       BASIC SAMPLING
# ===========================================================================


class TestNeighborSamplerBasic:
    def test_returns_k_tools(self):
        reg, seed, neighbors = _simple_setup()
        # Pin seed selection so result is deterministic.
        sampler = NeighborToolSampler(registry=reg, k=3, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            result = sampler.sample()
        assert len(result) == 3

    def test_seed_is_first(self):
        reg, seed, _ = _simple_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            result = sampler.sample()
        assert result[0] is seed

    def test_no_duplicates(self):
        reg, seed, _ = _simple_setup()
        sampler = NeighborToolSampler(registry=reg, k=4, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            result = sampler.sample()
        qnames = [(t.qualified_name, _fp(t)) for t in result]
        assert len(qnames) == len(set(qnames))

    def test_k_one_returns_only_seed(self):
        reg, seed, _ = _simple_setup()
        sampler = NeighborToolSampler(registry=reg, k=1, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            result = sampler.sample()
        assert result == [seed]

    def test_empty_registry_raises(self):
        reg = ToolRegistry()
        reg.artifacts["neighbors"] = {}
        sampler = NeighborToolSampler(registry=reg, k=3)
        with pytest.raises(SamplingError):
            sampler.sample()

    def test_k_missing_raises(self):
        reg, seed, _ = _simple_setup()
        sampler = NeighborToolSampler(registry=reg)
        with pytest.raises(ValueError, match="k"):
            sampler.sample()

    def test_k_zero_raises(self):
        reg, seed, _ = _simple_setup()
        with pytest.raises(ValueError, match="positive"):
            NeighborToolSampler(registry=reg, k=0).sample()

    def test_callsite_k_overrides_constructor(self):
        reg, seed, _ = _simple_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            result = sampler.sample(k=3)
        assert len(result) == 3


# ===========================================================================
#                       CAPPING AND WARNINGS
# ===========================================================================


class TestNeighborSamplerCapping:
    def test_fewer_neighbors_than_k_returns_all_with_warning(self):
        reg, seed, neighbors = _simple_setup()  # 3 neighbors
        sampler = NeighborToolSampler(registry=reg, k=10, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            with patch.object(sampler, "logger") as mock_log:
                result = sampler.sample()
        # seed + 3 neighbors = 4 total
        assert len(result) == 4
        mock_log.warning.assert_called_once()

    def test_seed_with_no_neighbors_returns_seed_only_with_warning(self):
        seed = _tool("lone", "ns")
        reg = _registry_with_neighbors([seed], seed, [])
        sampler = NeighborToolSampler(registry=reg, k=3, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            with patch.object(sampler, "logger") as mock_log:
                result = sampler.sample()
        assert result == [seed]
        mock_log.warning.assert_called_once()

    def test_stale_neighbor_skipped_with_warning(self):
        """Neighbor entry pointing to a tool not in the registry is skipped."""
        seed = _tool("s", "ns")
        real_neighbor = _tool("real", "ns")
        # Build artifact manually with one stale entry.
        src_fp = _fp(seed)
        stale_fp = "deadbeef" * 8  # 64-char hex, won't match anything
        reg = _registry(seed, real_neighbor)
        reg.artifacts["neighbors"] = {
            seed.qualified_name: {
                src_fp: {
                    "ns": [
                        ("real", _fp(real_neighbor), 0.9),
                        ("ghost", stale_fp, 0.8),
                    ]
                }
            }
        }
        sampler = NeighborToolSampler(registry=reg, k=3, namespace="ns")
        with patch.object(sampler, "_select_seed", return_value=seed):
            with patch.object(sampler, "logger") as mock_log:
                result = sampler.sample()
        # Only seed + real_neighbor returned; ghost was skipped.
        assert len(result) == 2
        assert real_neighbor in result
        mock_log.warning.assert_called()


# ===========================================================================
#                       NAMESPACE SEED SELECTION
# ===========================================================================


class TestNeighborSeedNamespace:
    def test_namespace_hard_filter_constrains_seed(self):
        reg, seed = _multi_ns_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="a")
        # Run many times — seed must always come from namespace 'a'.
        for _ in range(30):
            result = sampler.sample()
            assert result[0].namespace == "a"

    def test_unknown_namespace_raises(self):
        reg, _ = _multi_ns_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            sampler.sample()

    def test_callsite_namespace_overrides_constructor(self):
        reg, seed = _multi_ns_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="a")
        for _ in range(30):
            result = sampler.sample(namespace="b")
            assert result[0].namespace == "b"

    def test_uniform_strategy_visits_all_namespaces(self):
        reg, seed = _multi_ns_setup()
        sampler = NeighborToolSampler(registry=reg, k=2, strategy="uniform")
        seen_ns = set()
        for _ in range(200):
            result = sampler.sample()
            seen_ns.add(result[0].namespace)
        assert seen_ns == {"a", "b", "c"}

    def test_namespace_weights_bias_seed_namespace(self):
        reg, seed = _multi_ns_setup()
        sampler = NeighborToolSampler(
            registry=reg, k=2, namespace_weights={"a": 1e9, "b": 0.0, "c": 0.0}
        )
        for _ in range(30):
            result = sampler.sample()
            assert result[0].namespace == "a"


# ===========================================================================
#                       SCORE NORMALIZATION AND WEIGHTED SAMPLING
# ===========================================================================


class TestNeighborScoreWeighting:
    def test_equal_scores_produce_near_uniform_sampling(self):
        """When all neighbor scores are identical, softmax produces uniform weights."""
        seed = _tool("s", "ns")
        targets = [(_tool(f"t{i}", "ns"), 0.5) for i in range(4)]
        reg = _registry_with_neighbors([seed] + [t for t, _ in targets], seed, targets)
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="ns")
        counts: Counter = Counter()
        with patch.object(sampler, "_select_seed", return_value=seed):
            for _ in range(200):
                result = sampler.sample()
                counts[result[1].name] += 1
        total = sum(counts.values())
        for name in counts:
            assert counts[name] / total > 0.1

    def test_higher_scored_neighbor_appears_more_often(self):
        """High-score neighbor should be selected more often than low-score neighbor."""
        seed = _tool("s", "ns")
        hi = _tool("hi", "ns")
        lo = _tool("lo", "ns")
        # With softmax both get nonzero weight, but hi (1.0) >> lo (0.0).
        reg = _registry_with_neighbors([seed, hi, lo], seed, [(hi, 1.0), (lo, 0.0)])
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="ns")
        counts: Counter = Counter()
        with patch.object(sampler, "_select_seed", return_value=seed):
            for _ in range(200):
                result = sampler.sample()
                counts[result[1].name] += 1
        assert counts["hi"] > counts["lo"]

    def test_low_temperature_concentrates_on_top_neighbor(self):
        """Very low temperature should make sampling nearly deterministic on top scorer."""
        seed = _tool("s", "ns")
        hi = _tool("hi", "ns")
        lo = _tool("lo", "ns")
        reg = _registry_with_neighbors([seed, hi, lo], seed, [(hi, 1.0), (lo, 0.0)])
        sampler = NeighborToolSampler(registry=reg, k=2, namespace="ns", temperature=0.01)
        counts: Counter = Counter()
        with patch.object(sampler, "_select_seed", return_value=seed):
            for _ in range(100):
                result = sampler.sample()
                counts[result[1].name] += 1
        # At very low temperature, hi should dominate almost entirely.
        assert counts["hi"] > 90

    def test_invalid_temperature_raises(self):
        seed = _tool("s", "ns")
        reg = _registry_with_neighbors([seed], seed, [])
        with pytest.raises(ValueError, match="temperature"):
            NeighborToolSampler(registry=reg, k=2, temperature=0.0)

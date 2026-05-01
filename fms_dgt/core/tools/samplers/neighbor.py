# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import random as _random

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import schema_fingerprint
from fms_dgt.core.tools.samplers.base import (
    SamplingError,
    ToolSampler,
    register_tool_sampler,
)
from fms_dgt.core.tools.samplers.utils import (
    build_namespace_distribution,
    build_ns_pools,
)


@register_tool_sampler("tc/neighbor")
class NeighborToolSampler(ToolSampler):
    """Sample a tool subset using the neighbor graph.

    .. warning::
        **EXPERIMENTAL — training objective not fully established.**

        The intended signal was "topic-coherent tool sets" — tools that
        plausibly co-occur in the same workflow.  However, design review found
        that this objective is already covered by ``tc/random`` with namespace
        filtering: if tools are organized by namespace (weather API, HR API,
        etc.), random sampling within a namespace produces domain-coherent sets
        without requiring this sampler or its enrichment dependency.

        The residual case — role-coherent sets that span namespaces in
        non-obvious ways — has no clear signal in the tool definition itself
        and is not well served by the ``neighbors`` artifact's mixed
        domain-proximity / dataflow score.

        The sampler is retained as working code pending a validated comparison
        against ``tc/random`` on real training data.  For dataflow-driven
        sampling (chaining, fan-in, fan-out), use the planned ``tc/chain``,
        ``tc/fan_in``, and ``tc/fan_out`` samplers once the ``dataflow``
        enrichment is available.  See the tool subsystem design discussion.

    Selects one seed tool then fills the remaining k-1 slots from that seed's
    neighbors, weighted by their compatibility scores.

    **Seed selection:** a namespace is drawn according to the configured
    distribution (``namespace``, ``namespace_weights``, ``strategy``), then
    one tool is chosen uniformly at random within that namespace.

    **Neighbor sampling:** neighbors are drawn without replacement, weighted
    by softmax-normalized compatibility scores.  Neighbors span all namespaces.
    If fewer than k-1 neighbors exist, all available neighbors are returned
    alongside the seed and a warning is logged.

    Args:
        registry: The ``ToolRegistry`` to sample from.
        k: Default number of tools to sample (seed + k-1 neighbors).
            Required unless every ``sample()`` call provides ``k`` explicitly.
        namespace: Hard-filter seed selection to a single namespace.
        namespace_weights: Per-namespace relative weights for seed namespace
            selection.  Unspecified namespaces receive fill-in weight via
            ``strategy``.
        strategy: Fill-in rule for namespaces not listed in
            ``namespace_weights``.  ``"uniform"`` (default) or
            ``"proportional"`` (weight by namespace size).
        temperature: Softmax temperature for neighbor score weighting.
            Lower values concentrate sampling on the highest-scored neighbors;
            higher values flatten toward uniform.  Must be > 0.  Default 1.0.
    """

    required_artifacts: List[str] = ["neighbors"]

    def __init__(
        self,
        registry: Any,
        k: int | None = None,
        namespace: str | None = None,
        namespace_weights: Dict[str, float] | None = None,
        strategy: Literal["uniform", "proportional"] = "proportional",
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"tc/neighbor: 'temperature' must be > 0, got {temperature}.")
        super().__init__(registry=registry, **kwargs)
        self._default_k = k
        self._default_namespace = namespace
        self._default_namespace_weights = namespace_weights
        self._default_strategy = strategy
        self._default_temperature = temperature

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_k(self, k: int | None) -> int:
        resolved = k if k is not None else self._default_k
        if resolved is None:
            raise ValueError(
                "tc/neighbor: 'k' must be provided either in the sampler config or "
                "as an argument to sample()."
            )
        if resolved <= 0:
            raise ValueError(f"tc/neighbor: 'k' must be a positive integer, got {resolved}.")
        return resolved

    def _select_seed(
        self,
        ns_pools: Dict[str, List[Tool]],
        namespace_weights: Dict[str, float] | None,
        strategy: Literal["uniform", "proportional"],
        namespace: str | None,
    ) -> Tool:
        """Pick a seed tool: select namespace via distribution, then tool uniformly.

        Args:
            ns_pools: Tools grouped by namespace.
            namespace_weights: Per-namespace relative weights.
            strategy: Fill-in strategy for unspecified namespaces.
            namespace: Hard-filter to a single namespace when set.

        Returns:
            A single ``Tool`` selected as the seed.
        """
        if namespace is not None:
            pool = ns_pools.get(namespace, [])
            if not pool:
                raise ValueError(
                    f"tc/neighbor: namespace {namespace!r} has no tools in the registry."
                )
            return _random.choice(pool)

        ns_probs = build_namespace_distribution(
            ns_pools, namespace_weights, strategy, "tc/neighbor", self.logger
        )
        namespaces = list(ns_probs.keys())
        weights = [ns_probs[ns] for ns in namespaces]
        chosen_ns = _random.choices(namespaces, weights=weights, k=1)[0]
        return _random.choice(ns_pools[chosen_ns])

    def _collect_neighbors(
        self,
        seed: Tool,
        rounds: int = 2,
    ) -> Tuple[List[Tuple[Tool, float]], Dict[str, List[str]]]:
        """Collect all neighbors of the seed across all namespaces.

        Looks up the seed's entry in ``registry.artifacts["neighbors"]``,
        flattens all namespace buckets, and resolves each neighbor tuple to a
        ``Tool`` via ``registry.get_by_fingerprint``.  Neighbors that cannot
        be resolved (e.g. stale cache) are skipped with a warning.

        Args:
            seed: The seed tool whose neighbors to collect.
            rounds: Number of times to expand neighborhood.

        Returns:
            List of ``(Tool, raw_score)`` pairs, deduplicated by
            ``(qualified_name, schema_fp)``.
        """
        neighbors_artifact: Dict = self._registry.artifacts["neighbors"]

        result: Dict[str, Tuple[Tool, float]] = dict()
        duplicates: Dict[str, List[Tuple[Tool, float]]] = dict()

        to_explore = [(seed, 1.0)]
        for _ in range(rounds):
            #
            for tool, score in to_explore:
                if tool.name not in result:
                    result[tool.name] = (tool, score)
            #
            frontier: List[Tuple[Tool, float]] = []
            for exp, _ in to_explore:
                exp_fp = schema_fingerprint(exp.parameters)
                fp_map = neighbors_artifact.get(exp.qualified_name, {})
                ns_buckets = fp_map.get(exp_fp, {})
                duplicates[exp.qualified_name] = []
                for ns, (neighbors_entries, duplicates_entries) in ns_buckets.items():
                    for add_to, entries in [
                        (frontier, neighbors_entries),
                        (duplicates[exp.qualified_name], duplicates_entries),
                    ]:
                        for unqualified_name, tgt_fp, score in entries:
                            qualified_name = f"{ns}::{unqualified_name}"
                            try:
                                tool = self._registry.get_by_fingerprint(qualified_name, tgt_fp)
                            except KeyError:
                                self.logger.warning(
                                    "tc/neighbor: neighbor (%s, %s) not found in registry — skipping.",
                                    qualified_name,
                                    tgt_fp,
                                )
                                continue
                            add_to.append((tool, score))
            to_explore = frontier

        #
        ret_neighbors = list(result.values())
        ret_duplicates = {k: [t.qualified_name for t, _ in v] for k, v in duplicates.items()}
        return ret_neighbors, ret_duplicates

    @staticmethod
    def _softmax_scores(neighbors: List[Tuple[Tool, float]], temperature: float) -> List[float]:
        """Convert raw scores to sampling weights via softmax.

        Softmax guarantees every neighbor gets a strictly positive weight so
        no candidate is permanently excluded.  Temperature controls peakedness:
        lower values concentrate weight on the highest-scored neighbors;
        higher values flatten the distribution toward uniform.

        Numerically stable: subtract the max score before exponentiating.

        Args:
            neighbors: List of ``(Tool, raw_score)`` pairs.
            temperature: Softmax temperature (must be > 0).

        Returns:
            List of softmax weights aligned with ``neighbors``, summing to 1.0.
        """
        scores = [score / temperature for _, score in neighbors]
        max_s = max(scores)
        exps = [math.exp(score - max_s) for score in scores]
        total = sum(exps)
        return [e / total for e in exps]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        k: int | None = None,
        namespace: str | None = None,
        namespace_weights: Optional[Dict[str, float]] = None,
        strategy: Optional[Literal["uniform", "proportional"]] = None,
        temperature: Optional[float] = None,
        use_seed_namespace: Optional[bool] = True,
        **kwargs: Any,
    ) -> List[Tool]:
        """Sample up to ``k`` tools: one seed plus k-1 neighbors.

        Call-site arguments override constructor defaults for this call only.

        Args:
            k: Number of tools to sample.  Overrides constructor ``k``.
            namespace: Hard-filter seed selection to a single namespace.
                Overrides constructor ``namespace``.
            namespace_weights: Per-namespace weights for seed namespace
                selection.  Overrides constructor ``namespace_weights``.
            strategy: Fill-in strategy.  Overrides constructor ``strategy``.
            temperature: Softmax temperature for neighbor weighting.
                Overrides constructor ``temperature``.
            use_seed_namespace: Whether or not to keep all selected tools
                in same namespace as selected seed

        Returns:
            List of sampled ``Tool`` instances.  First element is the seed.
            Length is ``min(k, 1 + available_neighbors)``.
        """
        resolved_k = self._resolve_k(k)
        resolved_ns = namespace if namespace is not None else self._default_namespace
        resolved_weights = (
            namespace_weights if namespace_weights is not None else self._default_namespace_weights
        )
        resolved_strategy = strategy if strategy is not None else self._default_strategy
        resolved_temperature = temperature if temperature is not None else self._default_temperature

        all_tools = self._registry.all_tools()
        if not all_tools:
            raise SamplingError("tc/neighbor: registry is empty.", requested=resolved_k, tools=[])

        ns_pools = build_ns_pools(all_tools)

        seed = self._select_seed(ns_pools, resolved_weights, resolved_strategy, resolved_ns)

        if resolved_k == 1:
            return [seed]

        neighbors, duplicates = self._collect_neighbors(seed)

        # Filter
        allowed_namespaces = [seed.namespace] if use_seed_namespace else list(ns_pools.keys())
        neighbors = [(t, score) for t, score in neighbors if t.namespace in allowed_namespaces]

        assert neighbors, f"tc/neighbor: seed {seed.qualified_name} has no neighbors in the graph"

        # Initialize return lists
        selected_neighbors: List[Tool] = []
        nogoods = set()

        want = resolved_k - 1
        if want > len(neighbors):
            self.logger.warning(
                "tc/neighbor: requested k=%d but seed %r has only %d neighbor(s); "
                "returning seed + all available neighbors.",
                resolved_k,
                seed.qualified_name,
                len(neighbors),
            )
            for tool in [t for t, _ in neighbors]:
                if tool.qualified_name not in nogoods:
                    nogoods.update(duplicates[tool.qualified_name])
                    selected_neighbors.append(tool)
            return selected_neighbors

        # Sample without replacement using softmax-weighted draw.
        # Softmax guarantees all weights are strictly positive so random.choices
        # never raises.  Re-apply softmax over the remaining pool after each
        # draw to keep the relative score ordering correct.
        remaining = [(t, s) for t, s in list(neighbors) if t.qualified_name not in nogoods]
        for _ in range(want):
            weights = self._softmax_scores(remaining, resolved_temperature)
            tools_r = [t for t, _ in remaining]
            chosen = _random.choices(tools_r, weights=weights, k=1)[0]
            selected_neighbors.append(chosen)
            nogoods.update(duplicates[chosen.qualified_name])
            remaining = [
                (t, s) for t, s in remaining if t is not chosen and t.qualified_name not in nogoods
            ]

        return selected_neighbors

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Literal, Optional
import math
import random as _random

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.samplers.base import (
    SamplingError,
    ToolSampler,
    register_tool_sampler,
)
from fms_dgt.core.tools.samplers.utils import (
    build_namespace_distribution,
    build_ns_pools,
)


@register_tool_sampler("tc/random")
class RandomToolSampler(ToolSampler):
    """Sample a random subset of tools from the registry.

    This is the zero-dependency baseline sampler.  It requires no enrichment
    artifacts and provides three levels of namespace control:

    **Hard filter** (``namespace``): Sample only from a single namespace.
    ``namespace_weights`` and ``strategy`` are ignored when this is set.

    **Weighted distribution** (``namespace_weights``): Provide relative weights
    for specific namespaces.  Unspecified namespaces share the remaining
    probability mass according to ``strategy``.  For example, with two
    namespaces ``a`` and ``b`` and ``namespace_weights={"a": 0.8}``, namespace
    ``b`` receives the remaining 0.2.  With three namespaces ``a``, ``b``,
    ``c`` and ``namespace_weights={"a": 0.6}``, ``b`` and ``c`` split the
    remaining 0.4 equally (uniform) or proportionally to their sizes
    (proportional).

    **Strategy only** (``strategy``): No explicit weights.  ``"uniform"``
    gives each namespace an equal expected share of the ``k`` slots.
    ``"proportional"`` weights namespaces by the number of tools they contain.

    **Precedence (call-site overrides constructor):** Any argument passed to
    ``sample()`` wins over the value set in the constructor.  This lets the
    scenario stage adjust sampling per call without rebuilding the sampler.

    **Capping:** If the resolved pool contains fewer than ``k`` tools, all
    available tools are returned and a warning is logged.  No error is raised.

    Args:
        registry: The ``ToolRegistry`` to sample from.
        k: Default number of tools to sample.  Required unless every
            ``sample()`` call provides ``k`` explicitly.
        namespace: Hard-filter to a single namespace.  When set,
            ``namespace_weights`` and ``strategy`` are ignored.
        namespace_weights: Mapping of namespace name to relative weight.
            Values need not sum to 1 — they are normalized internally.
            Unspecified namespaces receive fill-in weight via ``strategy``.
        strategy: Fill-in rule for namespaces not listed in
            ``namespace_weights``.  ``"uniform"`` (default) gives each
            unspecified namespace an equal share.  ``"proportional"`` weights
            them by namespace size.
    """

    required_artifacts: List[str] = []

    def __init__(
        self,
        registry: Any,
        k: int | None = None,
        namespace: str | None = None,
        namespace_weights: Dict[str, float] | None = None,
        strategy: Literal["uniform", "proportional"] = "proportional",
        **kwargs: Any,
    ) -> None:
        super().__init__(registry=registry, **kwargs)
        self._default_k = k
        self._default_namespace = namespace
        self._default_namespace_weights = namespace_weights
        self._default_strategy = strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_k(self, k: int | None) -> int:
        resolved = k if k is not None else self._default_k
        if resolved is None:
            raise ValueError(
                "tc/random: 'k' must be provided either in the sampler config or "
                "as an argument to sample()."
            )
        if resolved <= 0:
            raise ValueError(f"tc/random: 'k' must be a positive integer, got {resolved}.")
        return resolved

    @staticmethod
    def _largest_remainder(probs: Dict[str, float], k: int) -> Dict[str, int]:
        """Allocate ``k`` integer slots proportionally using the largest remainder method.

        Guarantees ``sum(slots.values()) == k`` and each value is either
        ``floor`` or ``ceil`` of the fair share.

        Args:
            probs: Normalized probability per namespace (must sum to ~1.0).
            k: Total slots to allocate.

        Returns:
            Dict mapping namespace to integer slot count.
        """
        quotas = {ns: p * k for ns, p in probs.items()}
        floors = {ns: math.floor(q) for ns, q in quotas.items()}
        remainder = k - sum(floors.values())

        # Distribute leftover slots to namespaces with largest fractional parts.
        fractional = sorted(quotas.keys(), key=lambda ns: quotas[ns] - floors[ns], reverse=True)
        for ns in fractional[:remainder]:
            floors[ns] += 1

        return floors

    def _draw_slots(
        self,
        k: int,
        ns_probs: Dict[str, float],
        ns_pools: Dict[str, List[Tool]],
    ) -> List[Tool]:
        """Draw ``k`` tools by batch slot allocation with overflow redistribution.

        Algorithm:
        1. Allocate integer slots per namespace via largest remainder method.
        2. Cap each namespace's slots at its pool size; collect overflow.
        3. Redistribute overflow over non-exhausted namespaces using original
           probabilities renormalized over the eligible set.
        4. Repeat until no overflow or all namespaces exhausted.

        Args:
            k: Total number of tools to draw.
            ns_probs: Normalized probability per namespace.
            ns_pools: Available tools per namespace.

        Returns:
            List of sampled ``Tool`` instances (length <= k).
        """
        # Mutable copies so we can sample without affecting caller state.
        pools = {ns: list(pool) for ns, pool in ns_pools.items()}
        # slots[ns] tracks how many tools to take from each namespace.
        slots: Dict[str, int] = {ns: 0 for ns in ns_probs}
        remaining = k
        active_probs = dict(ns_probs)

        while remaining > 0 and active_probs:
            total_w = sum(active_probs.values())
            norm_probs = {ns: w / total_w for ns, w in active_probs.items()}
            allocated = self._largest_remainder(norm_probs, remaining)

            overflow = 0
            newly_exhausted = []
            for ns, alloc in allocated.items():
                capped = min(alloc, len(pools[ns]) - slots[ns])
                overflow += alloc - capped
                slots[ns] += capped
                if slots[ns] >= len(pools[ns]):
                    newly_exhausted.append(ns)

            for ns in newly_exhausted:
                del active_probs[ns]

            remaining = overflow
            if not newly_exhausted:
                # No namespace was exhausted this round — overflow must be zero.
                break

        selected: List[Tool] = []
        for ns, count in slots.items():
            for tool in _random.sample(pools[ns], count):
                if not any(t.name == tool.name for t in selected):
                    selected.append(tool)
        return selected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        k: int | None = None,
        namespace: str | None = None,
        namespace_weights: Optional[Dict[str, float]] = None,
        strategy: Optional[Literal["uniform", "proportional"]] = None,
        **kwargs: Any,
    ) -> List[Tool]:
        """Sample up to ``k`` tools from the registry.

        Call-site arguments override constructor defaults for this call only.

        Args:
            k: Number of tools to sample.  Overrides constructor ``k`` when
                provided.
            namespace: Hard-filter to a single namespace for this call.
                Overrides constructor ``namespace``.
            namespace_weights: Per-namespace relative weights for this call.
                Overrides constructor ``namespace_weights``.
            strategy: Fill-in strategy for this call.  Overrides constructor
                ``strategy``.

        Returns:
            List of sampled ``Tool`` instances.  Length is
            ``min(k, available_pool_size)``.
        """
        resolved_k = self._resolve_k(k)
        resolved_ns = namespace if namespace is not None else self._default_namespace
        resolved_weights = (
            namespace_weights if namespace_weights is not None else self._default_namespace_weights
        )
        resolved_strategy = strategy if strategy is not None else self._default_strategy

        all_tools = self._registry.all_tools()
        if not all_tools:
            raise SamplingError("tc/random: registry is empty.", requested=resolved_k, tools=[])

        # Hard namespace filter — ignore weights/strategy.
        if resolved_ns is not None:
            pool = [t for t in all_tools if t.namespace == resolved_ns]
            if not pool:
                raise SamplingError(
                    f"tc/random: namespace {resolved_ns!r} has no tools.",
                    requested=resolved_k,
                    tools=[],
                )
            if resolved_k > len(pool):
                self.logger.warning(
                    "tc/random: requested k=%d but namespace %r has only %d tool(s); "
                    "returning all available.",
                    resolved_k,
                    resolved_ns,
                    len(pool),
                )
            return _random.sample(pool, min(resolved_k, len(pool)))

        # Weighted / strategy-based sampling across namespaces.
        ns_pools = build_ns_pools(all_tools)

        total_available = sum(len(p) for p in ns_pools.values())
        if resolved_k > total_available:
            self.logger.warning(
                "tc/random: requested k=%d but only %d tool(s) available; returning all available.",
                resolved_k,
                total_available,
            )

        ns_probs = build_namespace_distribution(
            ns_pools, resolved_weights, resolved_strategy, "tc/random", self.logger
        )

        return self._draw_slots(resolved_k, ns_probs, ns_pools)

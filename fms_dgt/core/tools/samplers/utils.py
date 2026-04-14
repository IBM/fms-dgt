# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from typing import Any, Dict, List, Literal

# Local
from fms_dgt.core.tools.data_objects import Tool


def build_namespace_distribution(
    ns_pools: Dict[str, List[Tool]],
    namespace_weights: Dict[str, float] | None,
    strategy: Literal["uniform", "proportional"],
    sampler_name: str,
    logger: Any,
) -> Dict[str, float]:
    """Return a normalized probability map over namespaces.

    Only namespaces that actually have tools are included.

    Args:
        ns_pools: Tools grouped by namespace (keys are the active namespaces).
        namespace_weights: Caller-supplied relative weights for specific
            namespaces.  ``None`` means no explicit weights.
        strategy: Fill-in rule for unspecified namespaces.  ``"uniform"``
            gives each unspecified namespace an equal share.
            ``"proportional"`` weights them by namespace size.
        sampler_name: Registered sampler name used in warning messages.
        logger: Logger instance from the calling sampler.

    Returns:
        Dict mapping namespace name to probability (sums to 1.0).

    Raises:
        ValueError: If all explicit namespace_weights values are <= 0.
    """
    active_namespaces = set(ns_pools.keys())

    if not namespace_weights:
        if strategy == "proportional":
            total = sum(len(pool) for pool in ns_pools.values())
            return {ns: len(ns_pools[ns]) / total for ns in active_namespaces}
        else:  # uniform
            n = len(active_namespaces)
            return {ns: 1.0 / n for ns in active_namespaces}

    # Only keep entries that correspond to namespaces that exist in the registry.
    explicit: Dict[str, float] = {
        ns: w for ns, w in namespace_weights.items() if ns in active_namespaces
    }
    ignored = set(namespace_weights.keys()) - active_namespaces
    if ignored:
        logger.warning(
            "%s: namespace_weights contains unknown namespace(s) %s — ignoring.",
            sampler_name,
            sorted(ignored),
        )

    explicit_total = sum(explicit.values())
    if explicit_total <= 0:
        raise ValueError(
            f"{sampler_name}: namespace_weights values must be positive and sum to > 0."
        )

    unspecified = active_namespaces - set(explicit.keys())

    if not unspecified:
        return {ns: w / explicit_total for ns, w in explicit.items()}

    if strategy == "proportional":
        fill_weights = {ns: float(len(ns_pools[ns])) for ns in unspecified}
    else:  # uniform
        fill_weights = {ns: 1.0 for ns in unspecified}

    combined = {**explicit, **fill_weights}
    total = sum(combined.values())
    return {ns: w / total for ns, w in combined.items()}


def build_ns_pools(tools: List[Tool]) -> Dict[str, List[Tool]]:
    """Group a flat tool list into a namespace-keyed dict.

    Args:
        tools: Flat list of ``Tool`` instances.

    Returns:
        Dict mapping namespace name to list of tools in that namespace.
    """
    pools: Dict[str, List[Tool]] = defaultdict(list)
    for tool in tools:
        pools[tool.namespace].append(tool)
    return dict(pools)

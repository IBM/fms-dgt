# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List, Optional
import random as _random

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.samplers.base import (
    SamplingError,
    ToolSampler,
    register_tool_sampler,
)
from fms_dgt.core.tools.samplers.chain import _collect_successors, _fp, _weighted_choice


@register_tool_sampler("tc/fan_out")
class FanOutToolSampler(ToolSampler):
    """Sample a 1→N fan-out topology: one seed feeds k-1 independent successors.

    The seed tool is selected at random (namespace-filtered if set).  Then
    ``k-1`` distinct successors are sampled from the seed's outgoing edges in
    ``dataflow_out``, with selection probability proportional to edge score.

    **Return order:** ``[seed, B, C, …]`` — seed first.

    **Failure:** If fewer than ``k-1`` qualifying successors exist after
    applying ``min_score``, a ``SamplingError`` is raised.  The ``tools``
    attribute carries ``[seed] + collected_successors``.

    Args:
        registry: ``ToolRegistry`` instance with ``dataflow`` artifact.
        k: Total number of tools (seed + k-1 successors).  Must be ≥ 2.
        min_score: Minimum edge score for a successor to qualify (default 0.0).
        namespace: Restrict seed and successors to this namespace.
    """

    required_artifacts: List[str] = ["dataflow"]

    def __init__(
        self,
        registry: Any,
        k: int | None = None,
        min_score: float = 0.0,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry=registry, **kwargs)
        self._default_k = k
        self._default_min_score = min_score
        self._default_namespace = namespace

    def _resolve_k(self, k: int | None) -> int:
        resolved = k if k is not None else self._default_k
        if resolved is None:
            raise ValueError(
                "tc/fan_out: 'k' must be provided either in the sampler config or "
                "as an argument to sample()."
            )
        if resolved < 2:
            raise ValueError(
                f"tc/fan_out: 'k' must be at least 2 (seed + 1 successor), got {resolved}."
            )
        return resolved

    def sample(
        self,
        k: int | None = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tool]:
        """Sample a seed and k-1 successors from the dataflow graph.

        Args:
            k: Total tools (seed + successors).  Overrides constructor value.
            min_score: Edge score threshold for this call.
            namespace: Namespace filter for this call.

        Returns:
            ``List[Tool]`` of length ``k``, seed first.

        Raises:
            SamplingError: If fewer than ``k-1`` qualifying successors exist.
        """
        resolved_k = self._resolve_k(k)
        resolved_min = min_score if min_score is not None else self._default_min_score
        resolved_ns = namespace if namespace is not None else self._default_namespace

        dataflow_out = self._registry.artifacts["dataflow"]["out"]
        all_tools = self._registry.all_tools()

        if not all_tools:
            raise SamplingError("tc/fan_out: registry is empty.", requested=resolved_k, tools=[])

        pool = [t for t in all_tools if resolved_ns is None or t.namespace == resolved_ns]
        if not pool:
            raise SamplingError(
                f"tc/fan_out: no tools in namespace {resolved_ns!r}.",
                requested=resolved_k,
                tools=[],
            )

        # Try seeds in random order until one has enough successors.
        _random.shuffle(pool)
        for seed in pool:
            seed_fp = _fp(seed)
            successors = _collect_successors(
                seed, seed_fp, dataflow_out, resolved_ns, resolved_min, self._registry
            )
            # Remove seed itself from candidates.
            successors = [(t, s) for t, s in successors if t.qualified_name != seed.qualified_name]
            if len(successors) < resolved_k - 1:
                continue

            # Sample k-1 successors without replacement, weighted by score.
            chosen: List[Tool] = []
            remaining = list(successors)
            for _ in range(resolved_k - 1):
                pick = _weighted_choice(remaining)
                chosen.append(pick)
                remaining = [
                    (t, s) for t, s in remaining if t.qualified_name != pick.qualified_name
                ]

            return [seed] + chosen

        # No seed had enough successors.
        raise SamplingError(
            f"tc/fan_out: no seed tool has at least {resolved_k - 1} qualifying "
            f"successor(s) with min_score={resolved_min}.",
            requested=resolved_k,
            tools=[],
        )

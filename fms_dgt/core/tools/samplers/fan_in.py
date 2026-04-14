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
from fms_dgt.core.tools.samplers.chain import (
    _collect_predecessors,
    _fp,
    _weighted_choice,
)


@register_tool_sampler("tc/fan_in")
class FanInToolSampler(ToolSampler):
    """Sample an N→1 fan-in topology: k-1 predecessors feed one sink tool.

    A sink tool is selected at random from tools that have at least ``k-1``
    qualifying predecessors in ``dataflow_in`` (after applying ``min_score``).
    Then ``k-1`` predecessors are sampled with probability proportional to
    edge score.

    **Return order:** ``[pred_1, pred_2, …, sink]`` — sink last.

    **Failure:** If no tool has enough qualifying predecessors, a
    ``SamplingError`` is raised with an empty ``tools`` list.

    Args:
        registry: ``ToolRegistry`` instance with ``dataflow`` artifact.
        k: Total tools (k-1 predecessors + sink).  Must be ≥ 2.
        min_score: Minimum edge score for a predecessor to qualify (default 0.0).
        namespace: Restrict sink and predecessors to this namespace.
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
                "tc/fan_in: 'k' must be provided either in the sampler config or "
                "as an argument to sample()."
            )
        if resolved < 2:
            raise ValueError(
                f"tc/fan_in: 'k' must be at least 2 (1 predecessor + sink), got {resolved}."
            )
        return resolved

    def sample(
        self,
        k: int | None = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tool]:
        """Sample k-1 predecessors and a sink from the dataflow graph.

        Args:
            k: Total tools (predecessors + sink).  Overrides constructor value.
            min_score: Edge score threshold for this call.
            namespace: Namespace filter for this call.

        Returns:
            ``List[Tool]`` of length ``k``, sink last.

        Raises:
            SamplingError: If no tool has at least ``k-1`` qualifying predecessors.
        """
        resolved_k = self._resolve_k(k)
        resolved_min = min_score if min_score is not None else self._default_min_score
        resolved_ns = namespace if namespace is not None else self._default_namespace

        dataflow_in = self._registry.artifacts["dataflow"]["in"]
        all_tools = self._registry.all_tools()

        if not all_tools:
            raise SamplingError("tc/fan_in: registry is empty.", requested=resolved_k, tools=[])

        pool = [t for t in all_tools if resolved_ns is None or t.namespace == resolved_ns]
        if not pool:
            raise SamplingError(
                f"tc/fan_in: no tools in namespace {resolved_ns!r}.",
                requested=resolved_k,
                tools=[],
            )

        # Try sinks in random order until one has enough predecessors.
        _random.shuffle(pool)
        for sink in pool:
            sink_fp = _fp(sink)
            predecessors = _collect_predecessors(
                sink, sink_fp, dataflow_in, resolved_ns, resolved_min, self._registry
            )
            # Remove sink itself from candidates (self-loops are not valid predecessors).
            predecessors = [
                (t, s) for t, s in predecessors if t.qualified_name != sink.qualified_name
            ]
            if len(predecessors) < resolved_k - 1:
                continue

            # Sample k-1 predecessors without replacement, weighted by score.
            chosen: List[Tool] = []
            remaining = list(predecessors)
            for _ in range(resolved_k - 1):
                pick = _weighted_choice(remaining)
                chosen.append(pick)
                remaining = [
                    (t, s) for t, s in remaining if t.qualified_name != pick.qualified_name
                ]

            return chosen + [sink]

        raise SamplingError(
            f"tc/fan_in: no sink tool has at least {resolved_k - 1} qualifying "
            f"predecessor(s) with min_score={resolved_min}.",
            requested=resolved_k,
            tools=[],
        )

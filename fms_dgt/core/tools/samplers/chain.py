# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional, Tuple
import random as _random

# Local
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.registry import schema_fingerprint
from fms_dgt.core.tools.samplers.base import (
    SamplingError,
    ToolSampler,
    register_tool_sampler,
)


@register_tool_sampler("tc/chain")
class ChainToolSampler(ToolSampler):
    """Sample a linear dataflow chain: A→B→C→…

    Each tool in the returned list is a valid successor of the previous tool
    according to the ``dataflow_out`` artifact.  The chain is built by a
    greedy random walk: at each step, all qualifying successors for the
    current tool are collected, then one is sampled at random with probability
    proportional to its edge score.

    **Return order:** ``[A, B, C, …]`` — position communicates execution
    dependency; the scenario generator uses this order to construct prompts.

    **Failure:** If the walk dead-ends before reaching ``k`` tools (no
    qualifying successor with score ≥ ``min_score``), a ``SamplingError`` is
    raised.  The ``tools`` attribute of the exception carries the partial chain
    collected so far.

    **Namespace filter:** When ``namespace`` is set, both the seed and all
    successors are restricted to that namespace.

    Args:
        registry: ``ToolRegistry`` instance with ``dataflow_out`` artifact.
        k: Chain length.  Required unless every ``sample()`` call provides it.
        min_score: Minimum edge score to consider a successor qualifying
            (default 0.0 — all edges qualify).
        namespace: Restrict seed and all successors to this namespace.
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
                "tc/chain: 'k' must be provided either in the sampler config or "
                "as an argument to sample()."
            )
        if resolved <= 0:
            raise ValueError(f"tc/chain: 'k' must be a positive integer, got {resolved}.")
        return resolved

    def sample(
        self,
        k: int | None = None,
        min_score: Optional[float] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tool]:
        """Build and return a linear dataflow chain of length ``k``.

        Args:
            k: Chain length.  Overrides constructor ``k`` when provided.
            min_score: Minimum edge score threshold for this call.
            namespace: Namespace filter for this call.

        Returns:
            ``List[Tool]`` of length ``k`` in A→B→C order.

        Raises:
            SamplingError: If the chain dead-ends before reaching length ``k``.
        """
        resolved_k = self._resolve_k(k)
        resolved_min = min_score if min_score is not None else self._default_min_score
        resolved_ns = namespace if namespace is not None else self._default_namespace

        dataflow_out = self._registry.artifacts["dataflow"]["out"]
        all_tools = self._registry.all_tools()

        if not all_tools:
            raise SamplingError("tc/chain: registry is empty.", requested=resolved_k, tools=[])

        # Pick a seed tool (namespace-filtered if set).
        pool = [t for t in all_tools if resolved_ns is None or t.namespace == resolved_ns]
        if not pool:
            raise SamplingError(
                f"tc/chain: no tools in namespace {resolved_ns!r}.",
                requested=resolved_k,
                tools=[],
            )

        seed = _random.choice(pool)
        chain: List[Tool] = [seed]

        while len(chain) < resolved_k:
            current = chain[-1]
            current_fp = _fp(current)
            successors = _collect_successors(
                current, current_fp, dataflow_out, resolved_ns, resolved_min, self._registry
            )
            if not successors:
                raise SamplingError(
                    f"tc/chain: dead end at tool {current.qualified_name!r} after "
                    f"{len(chain)} step(s); cannot reach k={resolved_k}.",
                    requested=resolved_k,
                    tools=list(chain),
                )
            # Avoid revisiting tools already in chain.
            visited = {t.qualified_name for t in chain}
            candidates = [(t, s) for t, s in successors if t.qualified_name not in visited]
            if not candidates:
                raise SamplingError(
                    f"tc/chain: all successors of {current.qualified_name!r} already in chain.",
                    requested=resolved_k,
                    tools=list(chain),
                )
            chosen = _weighted_choice(candidates)
            chain.append(chosen)

        return chain


# ---------------------------------------------------------------------------
# Shared graph helpers (used by chain, fan_out, fan_in)
# ---------------------------------------------------------------------------


def _fp(tool: Tool) -> str:
    """Return the schema fingerprint for a tool's input parameters."""
    return schema_fingerprint(tool.parameters)


def _collect_successors(
    src: Tool,
    src_fp: str,
    dataflow_out: Dict,
    namespace: Optional[str],
    min_score: float,
    registry: Any,
) -> List[Tuple[Tool, float]]:
    """Return ``(tool, edge_score)`` pairs for all qualifying successors of ``src``.

    Args:
        src: Source tool.
        src_fp: Schema fingerprint of ``src``.
        dataflow_out: ``registry.artifacts["dataflow_out"]``.
        namespace: If set, restrict successors to this namespace.
        min_score: Minimum edge score.
        registry: ``ToolRegistry`` for resolving tools by fingerprint.

    Returns:
        List of ``(Tool, score)`` tuples, all with score >= min_score.
    """
    fp_map = dataflow_out.get(src.qualified_name, {})
    ns_map = fp_map.get(src_fp, {})
    result: List[Tuple[Tool, float]] = []
    for ns, edges in ns_map.items():
        if namespace is not None and ns != namespace:
            continue
        for entry in edges:
            tgt_name, tgt_fp, score, _pairs = entry
            if score < min_score:
                continue
            try:
                tool = registry.get_by_fingerprint(f"{ns}::{tgt_name}", tgt_fp)
                result.append((tool, score))
            except KeyError:
                continue
    return result


def _collect_predecessors(
    sink: Tool,
    sink_fp: str,
    dataflow_in: Dict,
    namespace: Optional[str],
    min_score: float,
    registry: Any,
) -> List[Tuple[Tool, float]]:
    """Return ``(tool, edge_score)`` pairs for all qualifying predecessors of ``sink``."""
    fp_map = dataflow_in.get(sink.qualified_name, {})
    ns_map = fp_map.get(sink_fp, {})
    result: List[Tuple[Tool, float]] = []
    for ns, edges in ns_map.items():
        if namespace is not None and ns != namespace:
            continue
        for entry in edges:
            src_name, src_fp, score, _pairs = entry
            if score < min_score:
                continue
            try:
                tool = registry.get_by_fingerprint(f"{ns}::{src_name}", src_fp)
                result.append((tool, score))
            except KeyError:
                continue
    return result


def _weighted_choice(candidates: List[Tuple[Tool, float]]) -> Tool:
    """Pick one tool from ``candidates`` with probability proportional to score."""
    tools, scores = zip(*candidates)
    total = sum(scores)
    r = _random.random() * total
    cumulative = 0.0
    for tool, score in zip(tools, scores):
        cumulative += score
        if r <= cumulative:
            return tool
    return tools[-1]  # fallback for floating-point edge cases

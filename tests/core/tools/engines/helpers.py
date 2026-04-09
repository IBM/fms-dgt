# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for engine test modules."""

# Standard
from unittest.mock import MagicMock, patch
import json

# Local
from fms_dgt.core.tools.constants import TOOL_NAMESPACE_SEP
from fms_dgt.core.tools.data_objects import Tool, ToolCall
from fms_dgt.core.tools.engines import LMToolEngine
from fms_dgt.core.tools.registry import ToolRegistry


def _make_registry(namespace: str = "ns") -> ToolRegistry:
    return ToolRegistry(
        tools=[
            Tool(
                name="search",
                namespace=namespace,
                description="Search",
                output_parameters={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            )
        ]
    )


def _make_call(
    namespace: str = "ns",
    tool: str = "search",
    call_id: str | None = None,
) -> ToolCall:
    return ToolCall(
        name=f"{namespace}{TOOL_NAMESPACE_SEP}{tool}",
        arguments={"q": "hello"},
        call_id=call_id,
    )


def _make_lm_engine(registry=None, **kwargs) -> LMToolEngine:
    """Build an LMToolEngine with a mocked LM provider."""
    if registry is None:
        registry = _make_registry()
    lm_config = {"type": "mock_lm_for_test"}
    with patch("fms_dgt.core.tools.engines.lm.get_block") as mock_get_block:
        mock_lm = MagicMock()
        mock_get_block.return_value = mock_lm
        engine = LMToolEngine(registry, lm_config=lm_config, **kwargs)
    engine._lm = mock_lm
    return engine


def _set_lm_response(engine: LMToolEngine, tool_output: dict) -> None:
    """Configure the mock LM to return a well-formed tool output response."""
    engine._lm.return_value = [
        {
            "result": {"content": json.dumps(tool_output)},
            "addtl": {"token_logprobs": [[{"a": -0.1}]]},
        }
    ]

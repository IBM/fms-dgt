# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import asyncio
import logging

# Third Party
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import ToolEngine, register_tool_engine
from fms_dgt.core.tools.loaders.mcp import MCP_BASE_URL
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ===========================================================================
#                       MCP TOOL ENGINE
# ===========================================================================


@register_tool_engine("mcp")
class MCPToolEngine(ToolEngine):
    """Execute tool calls against a live MCP server over SSE transport.

    Each ``execute`` / ``simulate`` call opens a fresh SSE session, dispatches
    all tool calls via ``tools/call``, then closes the session.  Multiple MCP
    servers can coexist in the same ``ToolRegistry`` as long as their
    namespaces are distinct — each tool carries its server URL in metadata.

    Requires the ``mcp`` package (``pip install 'fms_dgt[mcp]'``).

    Auth options (mutually exclusive in precedence order):

    * ``bearer_token`` — sets ``Authorization: Bearer <token>``
    * ``basic_auth`` — ``(username, password)`` tuple
    * ``headers`` — arbitrary extra headers (applied in addition to auth)

    TLS:

    * ``verify`` — ``True`` (default), ``False``, or path to CA bundle.

    Args:
        registry: Shared ``ToolRegistry`` for tool lookup.
        bearer_token: Optional Bearer token.
        basic_auth: Optional ``(username, password)`` tuple.
        headers: Optional dict of extra HTTP headers.
        verify: TLS verification setting.
        timeout: Per-call timeout in seconds (default 30).
        namespaces: Optional namespace filter (see ``ToolEngine``).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        bearer_token: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
        headers: Optional[Dict[str, str]] = None,
        verify: bool | str = True,
        timeout: float = 30.0,
        namespaces: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, namespaces=namespaces)
        self._bearer_token = bearer_token
        self._basic_auth = tuple(basic_auth) if basic_auth else None
        self._extra_headers = dict(headers or {})
        self._verify = verify
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        hdrs = dict(self._extra_headers)
        if self._bearer_token:
            hdrs["Authorization"] = f"Bearer {self._bearer_token}"
        return hdrs

    def _build_httpx_auth(self) -> Any:
        if self._basic_auth:

            return httpx.BasicAuth(*self._basic_auth)
        return None

    async def _call_tools_async(
        self,
        base_url: str,
        calls: List[tuple[ToolCall, str]],
    ) -> List[ToolResult]:
        """Open one SSE session to ``base_url`` and dispatch all ``calls``.

        Args:
            base_url: MCP server base URL.
            calls: List of ``(tool_call, unqualified_tool_name)`` pairs.

        Returns:
            One ``ToolResult`` per call, in order.
        """
        results: List[ToolResult] = []
        sse_url = f"{base_url.rstrip('/')}/sse"

        async with sse_client(
            url=sse_url,
            headers=self._build_headers() or None,
            auth=self._build_httpx_auth(),
            timeout=self._timeout,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                for tool_call, tool_name in calls:
                    try:
                        mcp_result = await session.call_tool(
                            name=tool_name,
                            arguments=tool_call.arguments or {},
                        )
                        if mcp_result.isError:
                            error_text = _extract_text_content(mcp_result.content)
                            results.append(
                                ToolResult(
                                    call_id=tool_call.call_id,
                                    name=tool_call.name,
                                    error=error_text or "Tool reported isError=true",
                                )
                            )
                        else:
                            # Serialize content blocks to a plain dict/list.
                            content = [block.model_dump() for block in mcp_result.content]
                            results.append(
                                ToolResult(
                                    call_id=tool_call.call_id,
                                    name=tool_call.name,
                                    result=content,
                                )
                            )
                    except Exception as exc:
                        results.append(
                            ToolResult(
                                call_id=tool_call.call_id,
                                name=tool_call.name,
                                error=str(exc),
                            )
                        )

        return results

    def _dispatch(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Resolve tools, group by server URL, dispatch, and reassemble in order.

        Tool calls whose tool is unknown are returned as error results immediately.
        Remaining calls are grouped by ``mcp_base_url`` so each server gets one
        SSE session.  Results are reassembled in the original call order.
        """
        results: Dict[int, ToolResult] = {}
        # Group by base_url; preserve original index for ordering.
        by_server: Dict[str, List[tuple[int, ToolCall, str]]] = {}

        for i, tool_call in enumerate(tool_calls):
            tool = self._catalog.match(tool_call, namespaces=self._namespaces)
            if tool is None:
                results[i] = ToolResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    error=f"Unknown tool '{tool_call.name}'",
                )
                continue
            base_url = tool.metadata.get(MCP_BASE_URL)
            if not base_url:
                results[i] = ToolResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    error=(
                        f"Tool '{tool_call.name}' metadata missing '{MCP_BASE_URL}'. "
                        f"Was it loaded via MCPToolLoader?"
                    ),
                )
                continue
            by_server.setdefault(base_url, []).append((i, tool_call, tool.name))

        for base_url, entries in by_server.items():
            indices = [e[0] for e in entries]
            calls = [(e[1], e[2]) for e in entries]
            try:
                server_results = asyncio.run(self._call_tools_async(base_url, calls))
            except Exception as exc:
                server_results = [
                    ToolResult(call_id=tc.call_id, name=tc.name, error=str(exc)) for tc, _ in calls
                ]
            for idx, result in zip(indices, server_results):
                results[idx] = result

        return [results[i] for i in range(len(tool_calls))]

    # ------------------------------------------------------------------
    # ToolEngine interface
    # ------------------------------------------------------------------

    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls against the MCP server and record results.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to execute, in order.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.
        """
        with self._session_transaction(session_id):
            return self._dispatch(tool_calls)

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls without updating session state.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to probe.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.
        """
        with self._session_transaction(session_id, rollback=True):
            return self._dispatch(tool_calls)


# ===========================================================================
#                       HELPERS
# ===========================================================================


def _extract_text_content(content: Any) -> str:
    """Concatenate text from MCP content blocks."""
    if not isinstance(content, list):
        return str(content) if content else ""
    parts = []
    for block in content:
        text = getattr(block, "text", None) or (
            block.get("text") if isinstance(block, dict) else None
        )
        if text:
            parts.append(text)
    return " ".join(parts).strip()

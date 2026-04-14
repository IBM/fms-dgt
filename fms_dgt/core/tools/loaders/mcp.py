# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Optional
import asyncio
import logging

# Third Party
from mcp import ClientSession
from mcp.client.sse import sse_client
import httpx

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE
from fms_dgt.core.tools.data_objects import Tool, ToolList
from fms_dgt.core.tools.loaders.base import ToolLoader, register_tool_loader

logger = logging.getLogger(__name__)

# ===========================================================================
#                       METADATA KEYS
# ===========================================================================
# Written by MCPToolLoader, read back by MCPToolEngine.

MCP_BASE_URL = "mcp_base_url"
"""SSE endpoint base URL (e.g. ``http://localhost:8000``)."""

MCP_TRANSPORT = "mcp_transport"
"""Always ``"sse"`` for this loader."""


# ===========================================================================
#                       MCP TOOL LOADER
# ===========================================================================


@register_tool_loader("mcp")
class MCPToolLoader(ToolLoader):
    """Discover tools from a running MCP server over SSE transport.

    Uses the ``mcp`` client library (``pip install mcp``) to connect to the
    server's SSE endpoint and call ``tools/list``.  The server's base URL and
    auth configuration are stored in ``Tool.metadata`` so that
    ``MCPToolEngine`` can route calls back to the same server.

    Auth options (mutually exclusive in precedence order):

    * ``bearer_token`` — sets ``Authorization: Bearer <token>``
    * ``basic_auth`` — ``(username, password)`` tuple (HTTP Basic)
    * ``headers`` — arbitrary extra headers (applied in addition to auth)

    TLS:

    * ``verify`` — ``True`` (default), ``False``, or path to CA bundle.
      Passed to ``httpx`` via the ``mcp`` client.

    Args:
        base_url: Base URL of the MCP server (e.g. ``http://localhost:8000``).
            The SSE endpoint is resolved as ``<base_url>/sse``.
        namespace: Namespace to assign to all loaded tools.
        bearer_token: Optional Bearer token.
        basic_auth: Optional ``(username, password)`` tuple.
        headers: Optional dict of extra HTTP headers.
        verify: TLS verification setting.
        timeout: Connection timeout in seconds (default 10).
    """

    def __init__(
        self,
        base_url: str,
        namespace: str = TOOL_DEFAULT_NAMESPACE,
        bearer_token: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
        headers: Optional[Dict[str, str]] = None,
        verify: bool | str = True,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._namespace = namespace
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
        """Return an ``httpx.BasicAuth`` instance or ``None``."""
        if self._basic_auth:

            return httpx.BasicAuth(*self._basic_auth)
        return None

    async def _fetch_tools_async(self) -> ToolList:
        """Open an SSE session, call ``tools/list``, and close the session."""

        sse_url = f"{self._base_url}/sse"
        async with sse_client(
            url=sse_url,
            headers=self._build_headers() or None,
            auth=self._build_httpx_auth(),
            timeout=self._timeout,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

        tools: ToolList = []
        for mcp_tool in result.tools:
            tool = Tool(
                name=mcp_tool.name,
                namespace=self._namespace,
                description=mcp_tool.description or "",
                parameters=mcp_tool.inputSchema or {},
                output_parameters=(
                    mcp_tool.outputSchema.model_dump()
                    if getattr(mcp_tool, "outputSchema", None)
                    else {}
                ),
                metadata={
                    MCP_BASE_URL: self._base_url,
                    MCP_TRANSPORT: "sse",
                },
            )
            tools.append(tool)

        return tools

    # ------------------------------------------------------------------
    # ToolLoader interface
    # ------------------------------------------------------------------

    def load(self) -> ToolList:
        """Connect to the MCP server, call ``tools/list``, return ``Tool`` objects.

        Runs the async MCP client in a new event loop (safe to call from any
        synchronous context including the main thread).

        Returns:
            List of ``Tool`` instances with ``namespace`` and routing metadata set.

        Raises:
            Exception: On connection failure or JSON-RPC error.
        """
        tools = asyncio.run(self._fetch_tools_async())
        logger.debug(
            "MCPToolLoader loaded %d tools from %s (namespace=%r)",
            len(tools),
            self._base_url,
            self._namespace,
        )
        return tools

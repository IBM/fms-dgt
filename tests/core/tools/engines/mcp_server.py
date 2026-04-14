# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal MCP SSE server for integration testing MCPToolLoader / MCPToolEngine.

Exposes three tools:

* ``get_weather`` — returns a canned weather dict for a given city.
* ``calculator`` — evaluates simple arithmetic (add, subtract, multiply, divide).
* ``echo`` — returns its input unchanged (useful for smoke-testing round-trips).

Usage (from the repo root)::

    source .venv/bin/activate
    python tests/core/tools/engines/mcp_server.py          # port 8765 (default)
    python tests/core/tools/engines/mcp_server.py --port 9000

The server can also be launched programmatically for in-process tests via
``run_server_in_thread`` / ``stop_server``.
"""

# Standard
import argparse
import threading

# Third Party
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
import uvicorn

# ===========================================================================
#                       TOOL DEFINITIONS
# ===========================================================================

_TOOLS = [
    Tool(
        name="get_weather",
        description="Return current weather conditions for a city.",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="calculator",
        description="Evaluate simple arithmetic.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    ),
    Tool(
        name="echo",
        description="Return the input message unchanged.",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    ),
]

# ===========================================================================
#                       SERVER IMPLEMENTATION
# ===========================================================================


def build_app(require_bearer: str | None = None) -> Starlette:
    """Build and return the Starlette ASGI app.

    Args:
        require_bearer: When set, all requests must carry
            ``Authorization: Bearer <token>`` matching this value.
            Requests with wrong or missing tokens receive 401.
    """
    mcp_server = Server("dgt-test-server")

    @mcp_server.list_tools()
    async def list_tools():
        return _TOOLS

    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "get_weather":
            city = arguments.get("city", "unknown")
            units = arguments.get("units", "celsius")
            temp = 22 if units == "celsius" else 72
            text = f'{{"city": "{city}", "temperature": {temp}, "units": "{units}", "condition": "sunny"}}'
            return [TextContent(type="text", text=text)]

        if name == "calculator":
            a = arguments["a"]
            b = arguments["b"]
            op = arguments["operation"]
            if op == "add":
                val = a + b
            elif op == "subtract":
                val = a - b
            elif op == "multiply":
                val = a * b
            elif op == "divide":
                if b == 0:
                    return [TextContent(type="text", text='{"error": "division by zero"}')]
                val = a / b
            else:
                val = 0
            return [TextContent(type="text", text=f'{{"result": {val}}}')]

        if name == "echo":
            return [TextContent(type="text", text=arguments.get("message", ""))]

        return [TextContent(type="text", text=f'{{"error": "unknown tool: {name}"}}')]

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        if require_bearer:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {require_bearer}":
                return Response("Unauthorized", status_code=401)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
        return Response()

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]
    )


# ===========================================================================
#                       IN-PROCESS LAUNCH HELPERS
# ===========================================================================


class _ServerHandle:
    """Holds a running uvicorn server and the thread it runs in."""

    def __init__(self, server: uvicorn.Server, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


def run_server_in_thread(
    port: int = 8765,
    require_bearer: str | None = None,
) -> _ServerHandle:
    """Start the MCP server on a background thread and return a handle.

    The caller is responsible for calling ``handle.stop()`` when done.

    Args:
        port: TCP port to listen on.
        require_bearer: Optional bearer token to enforce on all SSE connections.

    Returns:
        A ``_ServerHandle`` with a ``stop()`` method.
    """
    app = build_app(require_bearer=require_bearer)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    started = threading.Event()
    _original_startup = server.startup

    async def _patched_startup(sockets=None):
        await _original_startup(sockets)
        started.set()

    server.startup = _patched_startup

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started.wait(timeout=5)
    return _ServerHandle(server, thread)


# ===========================================================================
#                       CLI ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGT test MCP SSE server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bearer", type=str, default=None, help="Required bearer token")
    args = parser.parse_args()

    app = build_app(require_bearer=args.bearer)
    uvicorn.run(app, host="127.0.0.1", port=args.port)

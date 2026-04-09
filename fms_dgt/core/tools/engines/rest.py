# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging

# Third Party
import requests

# Local
from fms_dgt.core.tools.data_objects import ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import ToolEngine, register_tool_engine
from fms_dgt.core.tools.loaders.rest import (
    REST_BASE_URL,
    REST_METHOD,
    REST_PARAM_LOCATIONS,
    REST_PATH,
)
from fms_dgt.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ===========================================================================
#                       REST TOOL ENGINE
# ===========================================================================


@register_tool_engine("rest")
class RESTToolEngine(ToolEngine):
    """Execute tool calls as real HTTP requests against a REST API.

    Routing metadata (base URL, HTTP method, path template, parameter
    locations) is read from ``Tool.metadata`` as written by ``RESTToolLoader``.
    Parameters are routed to path segments, query string, or JSON body
    depending on their declared location.

    Auth options (applied to every request, mutually exclusive in precedence):

    * ``bearer_token`` — sets ``Authorization: Bearer <token>``
    * ``basic_auth`` — ``(username, password)`` tuple for HTTP Basic auth
    * ``api_key`` — dict with keys ``name``, ``value``, and optionally
      ``location`` (``"header"`` or ``"query"``, default ``"header"``)

    These can be combined: e.g. ``bearer_token`` and extra ``headers`` together.

    TLS options:

    * ``verify`` — passed to ``requests`` (``True``, ``False``, or CA path).

    Args:
        registry: Shared ``ToolRegistry`` for tool lookup.
        bearer_token: Optional Bearer token.
        basic_auth: Optional ``(username, password)`` tuple.
        api_key: Optional dict ``{name, value, location}``.
        headers: Optional dict of extra HTTP headers.
        verify: TLS verification — ``True`` (default), ``False``, or CA path.
        timeout: Request timeout in seconds (default 30).
        namespaces: Optional namespace filter (see ``ToolEngine``).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        bearer_token: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
        api_key: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify: bool | str = True,
        timeout: float = 30.0,
        namespaces: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, namespaces=namespaces)
        self._bearer_token = bearer_token
        self._basic_auth = tuple(basic_auth) if basic_auth else None
        self._api_key = api_key or {}
        self._extra_headers = headers or {}
        self._verify = verify
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        hdrs: Dict[str, str] = {**self._extra_headers}
        if self._bearer_token:
            hdrs["Authorization"] = f"Bearer {self._bearer_token}"
        if self._api_key:
            loc = self._api_key.get("location", "header")
            if loc == "header":
                hdrs[self._api_key["name"]] = self._api_key["value"]
        return hdrs

    def _build_request_kwargs(
        self,
        method: str,
        url: str,
        arguments: Dict[str, Any],
        locations: Dict[str, str],
    ) -> Dict[str, Any]:
        """Partition arguments into query, path, and body buckets.

        Args:
            method: Uppercase HTTP method.
            url: Full URL (path placeholders already substituted).
            arguments: Tool call argument dict.
            locations: Param-to-location map from tool metadata.

        Returns:
            kwargs dict ready to unpack into ``requests.request``.
        """
        query_params: Dict[str, Any] = {}
        body_params: Dict[str, Any] = {}

        # API-key as query param (if configured).
        if self._api_key and self._api_key.get("location") == "query":
            query_params[self._api_key["name"]] = self._api_key["value"]

        for name, value in arguments.items():
            loc = locations.get(name, "query")
            if loc == "query":
                query_params[name] = value
            elif loc == "body":
                body_params[name] = value
            # "path" params are substituted directly into the URL before this call.

        kwargs: Dict[str, Any] = {
            "headers": self._build_headers(),
            "auth": self._basic_auth,
            "verify": self._verify,
            "timeout": self._timeout,
            "params": query_params or None,
        }
        if body_params:
            kwargs["json"] = body_params

        return kwargs

    def _execute_one(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a single tool call as an HTTP request."""
        tool = self._catalog.match(tool_call, namespaces=self._namespaces)
        if tool is None:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=f"Unknown tool '{tool_call.name}'",
            )

        base_url = tool.metadata.get(REST_BASE_URL)
        method = tool.metadata.get(REST_METHOD, "GET")
        path_template = tool.metadata.get(REST_PATH, "")
        locations: Dict[str, str] = tool.metadata.get(REST_PARAM_LOCATIONS, {})

        if not base_url:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=(
                    f"Tool '{tool_call.name}' metadata missing '{REST_BASE_URL}'. "
                    f"Was it loaded via RESTToolLoader?"
                ),
            )

        # Substitute path parameters.
        try:
            path = path_template.format(
                **{k: v for k, v in tool_call.arguments.items() if locations.get(k) == "path"}
            )
        except KeyError as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=f"Missing required path parameter: {exc}",
            )

        url = f"{base_url}{path}"
        request_kwargs = self._build_request_kwargs(method, url, tool_call.arguments, locations)

        try:
            response = requests.request(method, url, **request_kwargs)
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                result = response.text
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                result=result,
            )
        except requests.HTTPError as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except requests.RequestException as exc:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # ToolEngine interface
    # ------------------------------------------------------------------

    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> ToolResult:
        """Execute tool calls as real HTTP requests and record results.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to execute, in order.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.
        """
        with self._session_transaction(session_id):
            return [self._execute_one(tc) for tc in tool_calls]

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> ToolResult:
        """Execute tool calls without updating session state.

        REST calls always hit the live API — simulate and execute differ only
        in whether results are persisted to session history (they are not here).

        Args:
            session_id: Active session.
            tool_calls: Tool calls to probe.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.
        """
        with self._session_transaction(session_id, rollback=True):
            return [self._execute_one(tc) for tc in tool_calls]

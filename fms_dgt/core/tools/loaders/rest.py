# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import logging
import os

# Third Party
import requests

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE
from fms_dgt.core.tools.data_objects import Tool, ToolList
from fms_dgt.core.tools.loaders.base import ToolLoader, register_tool_loader
from fms_dgt.utils import read_json, read_yaml

logger = logging.getLogger(__name__)

# ===========================================================================
#                       METADATA KEYS
# ===========================================================================
# Written by RESTToolLoader and read back by RESTToolEngine.

REST_BASE_URL = "rest_base_url"
"""Base URL of the REST API (e.g. ``https://api.open-meteo.com``)."""

REST_METHOD = "rest_method"
"""HTTP method for this operation (e.g. ``"GET"``, ``"POST"``)."""

REST_PATH = "rest_path"
"""Path template for this operation (e.g. ``"/v1/forecast"``)."""

REST_PARAM_LOCATIONS = "rest_param_locations"
"""Dict mapping parameter name to location: ``"query"``, ``"path"``, or ``"body"``."""

# ===========================================================================
#                       REST TOOL LOADER
# ===========================================================================


@register_tool_loader("rest")
class RESTToolLoader(ToolLoader):
    """Discover tools from an OpenAPI/Swagger spec.

    Parses an OpenAPI 3.x (or Swagger 2.0) spec — from a local file or a
    remote URL — and converts each operation into a ``Tool`` object.  Routing
    metadata (HTTP method, path template, parameter locations) is stored in
    ``Tool.metadata`` for use by ``RESTToolEngine``.

    **Which operations are included:**
    Only operations that declare at least one parameter or a request body are
    included.  Operations without a declared ``operationId`` use
    ``<METHOD>_<path_slug>`` as the tool name.

    **Parameter location mapping:**
    OpenAPI ``in: query``, ``in: path``, and ``in: body`` / ``requestBody``
    are mapped to the ``rest_param_locations`` metadata dict.

    Auth options (for fetching a remote spec):

    * ``bearer_token`` — sets ``Authorization: Bearer <token>``
    * ``basic_auth`` — ``(username, password)`` tuple

    Args:
        spec: Path to a local ``.yaml``/``.json`` file **or** a remote URL
            starting with ``http://`` or ``https://``.
        namespace: Namespace to assign to all loaded tools.
        base_url: Base URL override.  When omitted the loader reads
            ``servers[0].url`` from the spec (OpenAPI 3.x) or constructs it
            from ``host`` + ``basePath`` (Swagger 2.0).
        bearer_token: Optional Bearer token (used only when fetching remote spec).
        basic_auth: Optional ``(username, password)`` for fetching remote spec.
        verify: TLS verification for remote spec fetch.
        timeout: Timeout in seconds for remote spec fetch (default 10).
        operation_ids: Optional allowlist of ``operationId`` values.  When
            provided, only matching operations are loaded.
    """

    def __init__(
        self,
        spec: str,
        namespace: str = TOOL_DEFAULT_NAMESPACE,
        base_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
        verify: bool | str = True,
        timeout: float = 10.0,
        operation_ids: Optional[List[str]] = None,
    ) -> None:
        self._spec = spec
        self._namespace = namespace
        self._base_url_override = base_url
        self._bearer_token = bearer_token
        self._basic_auth = tuple(basic_auth) if basic_auth else None
        self._verify = verify
        self._timeout = timeout
        self._operation_ids = set(operation_ids) if operation_ids else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_spec(self) -> Dict[str, Any]:
        """Load the OpenAPI spec from a local file or remote URL."""
        spec = self._spec
        if spec.startswith("http://") or spec.startswith("https://"):
            hdrs: Dict[str, str] = {}
            if self._bearer_token:
                hdrs["Authorization"] = f"Bearer {self._bearer_token}"
            resp = requests.get(
                spec,
                headers=hdrs,
                auth=self._basic_auth,
                verify=self._verify,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            # Detect format from Content-Type or URL extension.
            ct = resp.headers.get("Content-Type", "")
            if "yaml" in ct or spec.endswith((".yaml", ".yml")):
                # Third Party
                import yaml

                return yaml.safe_load(resp.text)
            return resp.json()

        path = spec
        ext = os.path.splitext(path)[-1].lower()
        if ext in (".yaml", ".yml"):
            return read_yaml(path)
        if ext == ".json":
            return read_json(path)
        raise ValueError(f"Unsupported spec file format '{ext}'. Use .yaml, .yml, or .json.")

    @staticmethod
    def _resolve_base_url(raw_spec: Dict[str, Any]) -> str:
        """Extract the base URL from the spec (OpenAPI 3.x or Swagger 2.0)."""
        # OpenAPI 3.x
        servers = raw_spec.get("servers")
        if servers and isinstance(servers, list):
            return servers[0].get("url", "").rstrip("/")
        # Swagger 2.0
        host = raw_spec.get("host", "")
        base_path = raw_spec.get("basePath", "")
        schemes = raw_spec.get("schemes", ["https"])
        scheme = schemes[0] if schemes else "https"
        if host:
            return f"{scheme}://{host}{base_path}".rstrip("/")
        return ""

    @staticmethod
    def _collect_parameters(
        operation: Dict[str, Any],
        path_item: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Build an input JSON Schema and a param-location map from an operation.

        Parameters defined at the path-item level are merged with
        operation-level parameters (operation wins on conflicts).

        Returns:
            Tuple of (json_schema_dict, param_locations_dict).
        """
        # Merge path-item params (lower priority) with operation params.
        merged: Dict[str, Dict[str, Any]] = {}
        for param in path_item.get("parameters", []):
            if "$ref" in param:
                continue
            merged[param["name"]] = param
        for param in operation.get("parameters", []):
            if "$ref" in param:
                continue
            merged[param["name"]] = param

        properties: Dict[str, Any] = {}
        required: List[str] = []
        locations: Dict[str, str] = {}

        for name, param in merged.items():
            loc = param.get("in", "query")
            locations[name] = loc
            schema = param.get("schema", {"type": "string"})
            properties[name] = {
                "description": param.get("description", ""),
                **schema,
            }
            if param.get("required", loc == "path"):
                required.append(name)

        # Handle requestBody (OpenAPI 3.x).
        request_body = operation.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            body_schema = json_content.get("schema", {})
            for prop_name, prop_schema in body_schema.get("properties", {}).items():
                properties[prop_name] = prop_schema
                locations[prop_name] = "body"
            for req_prop in body_schema.get("required", []):
                if req_prop not in required:
                    required.append(req_prop)

        input_schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required

        return input_schema, locations

    # ------------------------------------------------------------------
    # ToolLoader interface
    # ------------------------------------------------------------------

    def load(self) -> ToolList:
        """Parse the OpenAPI spec and return ``Tool`` objects.

        Each tool's ``metadata`` is populated with:

        * ``rest_base_url``: API base URL
        * ``rest_method``: HTTP method (uppercase)
        * ``rest_path``: path template
        * ``rest_param_locations``: ``{param_name: "query"|"path"|"body"}``

        Returns:
            List of ``Tool`` instances with ``namespace`` and routing metadata set.

        Raises:
            ValueError: If the spec cannot be parsed or has an unsupported shape.
            requests.HTTPError: If a remote spec URL returns a non-2xx response.
        """
        raw = self._fetch_spec()

        base_url = (
            self._base_url_override.rstrip("/")
            if self._base_url_override
            else self._resolve_base_url(raw)
        )

        paths: Dict[str, Any] = raw.get("paths", {})
        tools: ToolList = []

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ("get", "post", "put", "patch", "delete"):
                operation = path_item.get(method)
                if not isinstance(operation, dict):
                    continue

                op_id = operation.get("operationId")
                if self._operation_ids is not None and op_id not in self._operation_ids:
                    continue

                # Derive tool name from operationId or path+method slug.
                if op_id:
                    tool_name = op_id
                else:
                    slug = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
                    tool_name = f"{method}_{slug}" if slug else method

                description = operation.get("summary") or operation.get("description") or ""

                input_schema, locations = self._collect_parameters(operation, path_item)

                # Skip operations with no usable parameters.
                if not input_schema.get("properties"):
                    continue

                tool = Tool(
                    name=tool_name,
                    namespace=self._namespace,
                    description=description,
                    parameters=input_schema,
                    metadata={
                        REST_BASE_URL: base_url,
                        REST_METHOD: method.upper(),
                        REST_PATH: path,
                        REST_PARAM_LOCATIONS: locations,
                    },
                )
                tools.append(tool)

        logger.debug(
            "RESTToolLoader loaded %d tools from %s (namespace=%r)",
            len(tools),
            self._spec,
            self._namespace,
        )
        return tools

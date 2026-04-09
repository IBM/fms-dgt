# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Iterator, List, Optional, Set
import hashlib
import json
import logging
import os
import warnings

# Third Party
import jsonschema

# Local
from fms_dgt.core.tools.constants import (
    TOOL_DEFAULT_NAMESPACE,
    TOOL_NAMESPACE_SEP,
)
from fms_dgt.core.tools.data_objects import Tool, ToolCall, ToolList
from fms_dgt.utils import read_json, read_yaml

logger = logging.getLogger("fms_dgt.tools.registry")


# ===========================================================================
#                       SCHEMA FINGERPRINTING
# ===========================================================================


def _schema_fingerprint(schema: Dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint of a normalized JSON schema dict.

    Normalization: serialize with sorted keys, no extra whitespace.  This is
    deterministic across Python versions for pure-JSON schemas.

    Args:
        schema: A JSON-serializable dict (the tool's ``parameters`` field).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    normalized = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


# ===========================================================================
#                       TOOL REGISTRY
# ===========================================================================


class ToolRegistry:
    """Registry of Tool definitions spanning one or more namespaces.

    ``ToolRegistry`` is a first-class, databuilder-agnostic object.  A single
    instance holds tools from N namespaces; namespace is a property of each
    ``Tool``, not of the registry container.

    **Construction paths:**

    Normal path — file loader assigns namespace::

        registry = ToolRegistry.from_file("tools.yaml", namespace="weather_api")

    Multi-source path — each loader assigns its own namespace::

        registry = ToolRegistry.from_loaders([
            FileToolLoader("weather.yaml", namespace="weather_api"),
            FileToolLoader("hr.yaml", namespace="hr_api"),
        ])

    Advanced path — caller owns namespace on each Tool::

        registry = ToolRegistry([
            Tool(name="search", namespace="weather_api", ...),
            Tool(name="lookup", namespace="hr_api", ...),
        ])

    **Tool.namespace is required.** Every ``Tool`` passed to the constructor
    must have ``namespace`` set.  The registry raises immediately if any tool
    has an empty namespace.

    **Invariants enforced at construction (fail-fast, fail-loud):**

    * ``qualified_name(schema_fingerprint)`` must be globally unique.
      Same name + same schema + same namespace = duplicate symbol.  Hard error.
    * Same name + different schemas in the same namespace = valid overload.
      Allowed; resolved by schema-matching at dispatch time.
    * Same name across different namespaces = always allowed.

    **Refresh:**  When constructed via ``from_loaders``, calling ``refresh()``
    re-invokes ``load()`` on each retained loader and rebuilds the index.
    Useful for mid-run catalog updates (e.g. MCP server hot-reload).

    Args:
        tools: Flat list of ``Tool`` instances.  Each must have ``namespace``
            set.  Pass ``None`` or an empty list for an empty registry.

    Raises:
        ValueError: If any tool has an empty namespace, or on duplicate
            ``(namespace, name, schema)`` triplet.
    """

    def __init__(self, tools: ToolList | None = None) -> None:
        # Primary index: qualified_name -> list[Tool]  (list supports overloads)
        self._by_qualified_name: Dict[str, List[Tool]] = {}
        # Fingerprint guard: set of "qualified_name(schema_fp)" strings
        self._fingerprints: Set[str] = set()
        # Loaders retained for refresh(); empty when constructed directly
        self._loaders: List[Any] = []

        for tool in tools or []:
            self._register(tool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, tool: Tool) -> None:
        if not tool.namespace:
            raise ValueError(
                f"Tool {tool.name!r} has no namespace. Set tool.namespace before "
                f"registering, or use ToolRegistry.from_file() / from_loaders() "
                f"which assign namespace automatically."
            )
        key = f"{tool.qualified_name}({_schema_fingerprint(tool.parameters)})"
        if key in self._fingerprints:
            raise ValueError(
                f"Duplicate tool: {tool.qualified_name!r} with identical input schema "
                f"registered twice. Remove the duplicate definition."
            )
        self._fingerprints.add(key)
        qname = tool.qualified_name
        if qname not in self._by_qualified_name:
            self._by_qualified_name[qname] = []
        self._by_qualified_name[qname].append(tool)

    def _rebuild(self, tools: ToolList) -> None:
        """Clear all registered tools and re-register from ``tools``."""
        self._by_qualified_name.clear()
        self._fingerprints.clear()
        for tool in tools:
            self._register(tool)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(
        cls,
        tool_dicts: List[Dict[str, Any]],
        namespace: str = TOOL_DEFAULT_NAMESPACE,
    ) -> "ToolRegistry":
        """Construct a registry from a list of plain dicts.

        Each dict must have at least a ``name`` key.  The ``namespace``
        argument overrides any per-dict ``namespace`` field.

        Args:
            tool_dicts: List of tool definition dicts.
            namespace: Namespace to assign to every tool.

        Returns:
            Fully validated ``ToolRegistry``.
        """
        tools = [Tool.from_dict(d, namespace=namespace) for d in tool_dicts]
        return cls(tools=tools)

    @classmethod
    def from_file(cls, path: str, namespace: str | None = None) -> "ToolRegistry":
        """Load a registry from a YAML or JSON file.

        Two file shapes are supported:

        **Shape 1** — dict with a top-level ``namespace`` key and a ``tools``
        list.  The top-level ``namespace`` value is used unless overridden by
        the ``namespace`` argument::

            namespace: weather_api
            tools:
              - name: get_weather
                ...

        **Shape 2** — bare list of tool dicts.  All tools receive the
        ``namespace`` argument if provided, otherwise ``"default"``::

            - name: get_weather
              ...

        Args:
            path: Absolute or relative path to a ``.yaml``, ``.yml``, or
                ``.json`` file.
            namespace: Explicit namespace override.  When supplied it takes
                precedence over any ``namespace`` field in the file.

        Returns:
            Fully validated ``ToolRegistry``.

        Raises:
            ValueError: If the file format is unsupported or the content does
                not match either accepted shape.
        """

        ext = os.path.splitext(path)[-1].lower()
        if ext in (".yaml", ".yml"):
            raw = read_yaml(path)
        elif ext == ".json":
            raw = read_json(path)
        else:
            raise ValueError(f"Unsupported tool file format '{ext}'. Use .yaml, .yml, or .json.")

        if isinstance(raw, list):
            # Shape 2: bare list of tool dicts
            tool_dicts = raw
            file_namespace = TOOL_DEFAULT_NAMESPACE
        elif isinstance(raw, dict):
            first_value = next(iter(raw.values()), None) if raw else None
            if isinstance(first_value, list):
                # Shape 1: {<namespace>: [...tools...]}
                file_namespace, tool_dicts = next(iter(raw.items()))
            else:
                # Shape 3: {ToolName: {name, description, parameters, ...}, ...}
                tool_dicts = list(raw.values())
                file_namespace = TOOL_DEFAULT_NAMESPACE
        else:
            raise ValueError(
                f"Tool file must be a list of tool dicts, a dict mapping a namespace "
                f"to a list of tools, or a dict mapping tool names to tool defs. "
                f"Got: {type(raw).__name__}"
            )

        resolved_namespace = namespace if namespace is not None else file_namespace
        return cls.from_dicts(tool_dicts, namespace=resolved_namespace)

    @classmethod
    def from_loaders(cls, loaders: List[Any]) -> "ToolRegistry":
        """Construct a registry from one or more ``ToolLoader`` instances.

        Each loader is called once via ``load()`` to produce its tool list.
        All loaders are retained so that ``refresh()`` can reload them.

        Args:
            loaders: List of ``ToolLoader`` instances.  Each loader must have
                its namespace set before ``load()`` is called.

        Returns:
            Fully validated ``ToolRegistry``.
        """
        all_tools: ToolList = []
        for loader in loaders:
            all_tools.extend(loader.load())
        registry = cls(tools=all_tools)
        registry._loaders = list(loaders)
        return registry

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload all tools from the retained loaders and rebuild the index.

        Only meaningful when the registry was constructed via ``from_loaders``.
        If no loaders are retained (direct construction or ``from_file``), this
        is a no-op and a warning is logged.
        """
        if not self._loaders:
            warnings.warn(
                "ToolRegistry.refresh() called but no loaders are retained. "
                "Construct via from_loaders() to enable refresh.",
                UserWarning,
                stacklevel=2,
            )
            return
        all_tools: ToolList = []
        for loader in self._loaders:
            all_tools.extend(loader.load())
        self._rebuild(all_tools)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def all_tools(self) -> ToolList:
        """Return a flat list of all registered tools (including overloads)."""
        result: ToolList = []
        for group in self._by_qualified_name.values():
            result.extend(group)
        return result

    def get(self, qualified_name: str) -> List[Tool]:
        """Return all tools matching a qualified name (may be >1 for overloads).

        Args:
            qualified_name: ``namespace::tool_name`` string.

        Returns:
            List of matching ``Tool`` objects (empty if not found).
        """
        return list(self._by_qualified_name.get(qualified_name, []))

    def get_by_name(self, tool_name: str, namespace: str) -> List[Tool]:
        """Look up tools by unqualified name within a namespace.

        Args:
            tool_name: Unqualified tool name.
            namespace: Namespace to search within.

        Returns:
            List of matching ``Tool`` objects.
        """
        return self.get(f"{namespace}{TOOL_NAMESPACE_SEP}{tool_name}")

    def match(
        self,
        tool_call: ToolCall,
        namespaces: Optional[List[str]] = None,
    ) -> Optional[Tool]:
        """Find the tool definition that best matches a live tool call.

        When only one overload exists for the qualified name, it is returned
        immediately.  When multiple overloads exist (same name, different
        parameter schemas), jsonschema validation against each overload's
        ``parameters`` schema is used to narrow the candidates.  If more than
        one overload passes validation, the one with the highest argument key
        overlap is returned as the most specific match.

        Args:
            tool_call: A ``ToolCall`` carrying a qualified name and arguments.
            namespaces: Optional list of namespaces to restrict the search to.
                When ``None``, all namespaces are considered.

        Returns:
            The best-matching ``Tool``, or ``None`` if no overload is registered
            for the qualified name (or within the given namespaces).
        """
        candidates = self.get(tool_call.name)
        if namespaces is not None:
            candidates = [t for t in candidates if t.namespace in namespaces]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Multiple overloads: validate arguments against each parameters schema.
        valid = []
        for tool in candidates:
            try:
                jsonschema.validate(
                    instance=tool_call.arguments,
                    schema=tool.parameters,
                )
                valid.append(tool)
            except jsonschema.ValidationError:
                pass

        if not valid:
            # No overload strictly validates — fall back to highest key overlap.
            valid = candidates

        if len(valid) == 1:
            return valid[0]

        # Break ties by argument key overlap (most specific match wins).
        call_keys = set(tool_call.arguments)
        return max(
            valid,
            key=lambda t: len(call_keys & set(t.parameters.get("properties", {}))),
        )

    def namespaces(self) -> List[str]:
        """Return the distinct set of namespaces in this registry."""
        return list({t.namespace for t in self.all_tools()})

    def tool_names(
        self,
        namespace: str | None = None,
        namespaces: List[str] | None = None,
    ) -> List[str]:
        """Return unqualified tool names, optionally filtered by namespace(s).

        ``namespace`` (singular) and ``namespaces`` (plural) are mutually
        exclusive conveniences — pass one or the other.  ``namespaces`` takes
        precedence if both are supplied.

        Args:
            namespace: Restrict to a single namespace.
            namespaces: Restrict to any of the listed namespaces.

        Returns:
            Sorted list of unqualified names (duplicates removed).
        """
        if namespaces is not None:
            tools = [t for t in self.all_tools() if t.namespace in namespaces]
        elif namespace is not None:
            tools = [t for t in self.all_tools() if t.namespace == namespace]
        else:
            tools = self.all_tools()
        return sorted({t.name for t in tools})

    def to_dicts(self, qualified: bool = False) -> List[Dict[str, Any]]:
        """Serialize all tools to a list of plain dicts.

        Args:
            qualified: If True, use ``qualified_name`` in the ``name`` field.

        Returns:
            List of serialized tool dicts.
        """
        return [
            tool.to_qualified_dict() if qualified else tool.to_dict() for tool in self.all_tools()
        ]

    def __len__(self) -> int:
        return sum(len(g) for g in self._by_qualified_name.values())

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.all_tools())

    def __contains__(self, qualified_name: str) -> bool:
        return qualified_name in self._by_qualified_name

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self)}, namespaces={self.namespaces()})"

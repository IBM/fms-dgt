# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Local
from fms_dgt.core.tools.constants import (
    TOOL_CALL_ARGS,
    TOOL_CALL_ID,
    TOOL_DEFAULT_NAMESPACE,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_NAMESPACE,
    TOOL_NAMESPACE_SEP,
    TOOL_OUTPUT_PARAMETERS,
    TOOL_PARAMETERS,
    TOOL_RESULT,
    TOOL_RESULT_ERROR,
)


# ===========================================================================
#                       TOOL
# ===========================================================================
@dataclass
class Tool:
    """A single tool definition held inside a ToolCatalog.

    Tools are first-class objects independent of any particular databuilder.
    A Tool can be loaded from YAML/JSON, served by an MCP server, or
    constructed programmatically.  The same Tool instance can be referenced
    by multiple ToolEngines and multiple databuilders simultaneously.

    Attributes:
        name: Unqualified tool name, unique within its namespace.
        namespace: Catalog or server name.  Combined with ``name`` gives the
            always-qualified identifier ``namespace::tool_name``.
        description: Human-readable description passed to the LLM.
        parameters: JSON-Schema dict describing the input arguments.
        output_parameters: Optional JSON-Schema dict describing return values.
            Required for nested-call resolution in ToolCallValidator.
        metadata: Arbitrary extra fields (version, tags, source URL, etc.).
    """

    name: str
    namespace: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    output_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Always-qualified ``namespace::name``."""
        return f"{self.namespace}{TOOL_NAMESPACE_SEP}{self.name}"

    def to_dict(self, keep_keys: List[str] | None = None) -> Dict[str, Any]:
        """Serialize to a plain dict using the OpenAI function-call schema keys.

        The ``namespace`` field is included as a top-level extension key so
        that round-tripping through YAML/JSON preserves namespace information
        without encoding it inside ``name``.
        """
        d: Dict[str, Any] = {
            TOOL_NAME: self.name,
            TOOL_NAMESPACE: self.namespace,
            TOOL_DESCRIPTION: self.description,
            TOOL_PARAMETERS: self.parameters,
        }
        if self.output_parameters:
            d[TOOL_OUTPUT_PARAMETERS] = self.output_parameters
        if self.metadata:
            d["metadata"] = self.metadata
        if keep_keys is not None:
            d = {k: v for k, v in d.items() if k in keep_keys}
        return d

    def to_qualified_dict(self) -> Dict[str, Any]:
        """Like ``to_dict`` but uses the qualified name in the ``name`` field.

        Used when serializing tool definitions into a prompt or wire format
        where the consumer needs to emit a fully-qualified ``ToolCall.name``.
        """
        d = self.to_dict()
        d[TOOL_NAME] = self.qualified_name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], namespace: str | None = None) -> "Tool":
        """Construct a Tool from a plain dict.

        Namespace resolution order:
        1. ``namespace`` key inside the dict (per-tool authority).
        2. Explicit ``namespace`` argument (catalog-level fallback).
        3. ``TOOL_DEFAULT_NAMESPACE`` ("default").

        Args:
            d: Dict with at least a ``name`` key.
            namespace: Catalog-level namespace fallback.  Used when the dict
                does not carry its own ``namespace`` key.

        Returns:
            A fully initialized Tool instance.
        """
        resolved_ns = d.get(TOOL_NAMESPACE) or namespace or TOOL_DEFAULT_NAMESPACE
        return cls(
            name=d[TOOL_NAME],
            namespace=resolved_ns,
            description=d.get(TOOL_DESCRIPTION, ""),
            parameters=d.get(TOOL_PARAMETERS, {}),
            output_parameters=d.get(TOOL_OUTPUT_PARAMETERS, {}),
            metadata=d.get("metadata", {}),
        )


# ===========================================================================
#                       TOOL CALL
# ===========================================================================
@dataclass
class ToolCall:
    """A single tool invocation request, always carrying a qualified name.

    ``name`` is always of the form ``namespace::tool_name``.  This mirrors the
    OpenAI function-calling schema: the qualified name travels in the existing
    ``name`` field — no structural change to the wire format.

    Attributes:
        name: Always-qualified ``namespace::tool_name``.
        arguments: Argument dict matching the tool's ``parameters`` schema.
        call_id: Optional opaque ID for nested-call reference (``$id.output``).
    """

    name: str
    namespace: str
    qualified_name: str = field(default_factory=str)
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def __post_init__(self):
        if not self.qualified_name:
            self.qualified_name = TOOL_NAMESPACE_SEP.join([self.namespace, self.name])

    def to_dict(self, keep_keys: List[str] | None = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {TOOL_NAME: self.name, TOOL_CALL_ARGS: self.arguments}
        if self.call_id is not None:
            d[TOOL_CALL_ID] = self.call_id
        if keep_keys is not None:
            d = {k: v for k, v in d.items() if k in keep_keys}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(
            name=d[TOOL_NAME],
            namespace=d[TOOL_NAMESPACE],
            arguments=d.get(TOOL_CALL_ARGS, {}),
            call_id=d.get(TOOL_CALL_ID),
        )


# ===========================================================================
#                       TOOL RESULT
# ===========================================================================
@dataclass
class ToolResult:
    """The result of executing a single ToolCall.

    Attributes:
        call_id: Matches ``ToolCall.call_id`` when set; otherwise matched by
            position in the batch.
        name: Qualified tool name, copied from the originating ToolCall.
        result: The return value of the tool execution.  ``None`` on error.
        error: Error message string when execution failed; ``None`` on success.
        metadata: Arbitrary engine-specific extras (reward, done-signal, env
            info for RL engines).
    """

    call_id: str | None
    name: str
    result: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            TOOL_CALL_ID: self.call_id,
            TOOL_NAME: self.name,
            TOOL_RESULT: self.result,
        }
        if self.error is not None:
            d[TOOL_RESULT_ERROR] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolResult":
        return cls(
            call_id=d.get(TOOL_CALL_ID),
            name=d[TOOL_NAME],
            result=d.get(TOOL_RESULT),
            error=d.get(TOOL_RESULT_ERROR),
            metadata=d.get("metadata", {}),
        )


# ===========================================================================
#                       TYPE ALIAS
# ===========================================================================
ToolList = List[Tool]

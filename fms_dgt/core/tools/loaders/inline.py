# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE
from fms_dgt.core.tools.data_objects import Tool, ToolList
from fms_dgt.core.tools.loaders.base import ToolLoader, register_tool_loader


@register_tool_loader("inline")
class InlineToolLoader(ToolLoader):
    """Load tool definitions declared directly in the task YAML.

    Eliminates the need for a separate tool definition file when the schema
    is recipe-specific and simple enough to inline — for example, a single
    retrieval tool with a ``query: string`` parameter.

    ``tools`` is a required argument: a list of tool definition dicts following
    the same shape as Shape 2 of ``FileToolLoader`` (bare list of tool dicts).
    ``namespace`` defaults to ``"default"`` when omitted.

    Example task YAML::

        tools:
          registry:
            - type: inline
              namespace: retrieval
              engine: file_retriever
              tools:
                - name: search_documents
                  description: Search for relevant document chunks.
                  parameters:
                    type: object
                    properties:
                      query: {type: string}
                    required: [query]

    Args:
        tools: Required list of tool definition dicts.
        namespace: Namespace assigned to all loaded tools. Defaults to
            ``"default"``. A ``namespace`` key on an individual tool dict
            takes precedence over this argument.
    """

    def __init__(
        self,
        tools: List[dict],
        namespace: str | None = None,
    ) -> None:
        if not isinstance(tools, list):
            raise TypeError(
                f"InlineToolLoader requires 'tools' to be a list of tool dicts, "
                f"got {type(tools).__name__}."
            )
        self._tool_dicts = tools
        self._namespace = namespace or TOOL_DEFAULT_NAMESPACE

    def load(self) -> ToolList:
        """Return ``Tool`` objects constructed from the inline tool dicts.

        Returns:
            List of ``Tool`` instances with ``namespace`` set.
        """
        return [Tool.from_dict(d, namespace=self._namespace) for d in self._tool_dicts]

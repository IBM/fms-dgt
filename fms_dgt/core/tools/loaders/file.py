# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Local
from fms_dgt.core.tools.constants import TOOL_DEFAULT_NAMESPACE
from fms_dgt.core.tools.data_objects import Tool, ToolList
from fms_dgt.core.tools.loaders.base import ToolLoader, register_tool_loader
from fms_dgt.utils import read_json, read_yaml


@register_tool_loader("file")
class FileToolLoader(ToolLoader):
    """Load tool definitions from a YAML or JSON file.

    Supports two file shapes:

    **Shape 1** — dict with top-level ``namespace`` and ``tools`` keys::

        namespace: weather_api
        tools:
          - name: get_weather
            ...

    **Shape 2** — bare list of tool dicts; namespace defaults to ``"default"``
    unless overridden by the ``namespace`` constructor argument::

        - name: get_weather
          ...

    The ``namespace`` constructor argument always takes precedence over the
    file-level ``namespace`` key.

    Environment variables in ``path`` are expanded at ``load()`` time, so
    paths like ``${DGT_DATA_DIR}/tools.yaml`` work correctly.

    Args:
        path: Path to the ``.yaml``, ``.yml``, or ``.json`` file.
        namespace: Namespace override.  When supplied it takes precedence over
            any ``namespace`` field in the file.  When omitted, the file-level
            namespace is used (Shape 1) or ``"default"`` (Shape 2).
    """

    def __init__(self, path: str, namespace: str | None = None) -> None:
        self._path = path
        self._namespace = namespace

    @property
    def path(self) -> str:
        return self._path

    @property
    def namespace(self) -> str | None:
        return self._namespace

    def load(self) -> ToolList:
        """Read the file and return a list of ``Tool`` objects.

        Environment variables in the path are expanded at call time.

        Supported file shapes:

        **Shape 1** — single-key dict mapping namespace name to a list of tool
        dicts.  The key becomes the namespace::

            sgd:
              - name: AddEvent
                description: ...

        **Shape 2** — bare list of tool dicts; namespace defaults to the
        ``namespace`` constructor argument or ``"default"``::

            - name: AddEvent
              description: ...

        **Shape 3** — dict mapping tool name to tool def (typical format used
        by public benchmark datasets).  Namespace defaults to the ``namespace``
        constructor argument or ``"default"``::

            AddEvent:
              name: AddEvent
              description: ...
              parameters: ...

        In all shapes, a ``namespace`` key present on an individual tool dict
        takes precedence over any file-level or constructor-level namespace.

        Returns:
            List of ``Tool`` instances with ``namespace`` set.

        Raises:
            ValueError: If the file format is unsupported or the content shape
                is not recognized.
            FileNotFoundError: If the expanded path does not exist.
        """
        path = self._path

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

        resolved_namespace = self._namespace if self._namespace is not None else file_namespace
        return [Tool.from_dict(d, namespace=resolved_namespace) for d in tool_dicts]

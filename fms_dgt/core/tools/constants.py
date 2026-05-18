# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# ===========================================================================
#                       QUALIFIED NAME
# ===========================================================================

TOOL_NAMESPACE_SEP = "::"
"""Separator between namespace and tool name in a qualified tool name.

Qualified names always take the form ``namespace::tool_name``.  The separator
``::`` is chosen because it is atypical in tool and namespace identifiers while
being widely readable (C++, Rust, Go module paths).  Single ``:`` and ``/``
are excluded because they appear in real tool names and URLs.
"""

TOOL_DEFAULT_NAMESPACE = "default"
"""Namespace assigned when no explicit namespace is provided."""

# ===========================================================================
#                       TOOL DEFINITION KEYS
# ===========================================================================
# Keys used in the tool-definition dict (OpenAI function-calling schema +
# DiGiT extensions).  Prefixed TOOL_* to avoid collisions with other
# framework-level constants (e.g. NAME_KEY in fms_dgt/constants.py).

TOOL_NAME = "name"
TOOL_NAMESPACE = "namespace"
TOOL_DESCRIPTION = "description"
TOOL_PARAMETERS = "parameters"
TOOL_OUTPUT_PARAMETERS = "output_parameters"
TOOL_PROPERTIES = "properties"
TOOL_REQUIRED = "required"
TOOL_TYPE = "type"

# ===========================================================================
#                       TOOL CALL / RESULT KEYS
# ===========================================================================

TOOL_CALL_ARGS = "arguments"
TOOL_CALL_ID = "id"
TOOL_RESULT = "result"
TOOL_RESULT_ERROR = "error"

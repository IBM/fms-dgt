# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Optional
import json

# Local
from fms_dgt.base.registry import get_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.core.tools.enrichments.base import ToolEnrichment, register_tool_enrichment
from fms_dgt.core.tools.enrichments.cache import (
    compute_fingerprint,
    enrichment_cache_path,
    load_cache,
    save_cache,
)
from fms_dgt.utils import try_parse_json_string

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a tool schema analyst.

You will be given a tool specification in OpenAI function-calling format.
Your task is to infer a plausible JSON Schema for the tool's OUTPUT — the value
it would return upon successful execution.

Requirements:
- Return a JSON Schema object (type: "object") with a "properties" key.
- Include only the most important return fields suggested by the tool's name and description.
- Use standard JSON Schema types (string, integer, number, boolean, array, object).
- Add a short "description" and a "type" for each property.
- Do not include "required" — treat all fields as optional.
- Do not include fields that represent error states.
- Return ONLY the JSON object, no explanation.

Example output (types vary — use the most appropriate type for each field):
{
  "type": "object",
  "properties": {
    "id": {"type": "string", "description": "Unique identifier of the created resource"},
    "status": {"type": "string", "description": "Outcome of the operation"},
    "count": {"type": "integer", "description": "Number of items returned"},
    "available": {"type": "boolean", "description": "Whether the resource is available"},
    "items": {"type": "array", "description": "List of matching results"}
  }
}\
"""

_USER_TEMPLATE = """\
Tool specification:
{tool_spec}

Infer the output_parameters JSON Schema for this tool.\
"""


def _tool_to_openai_spec(tool: Tool) -> str:
    """Render a tool as an OpenAI function-calling spec dict (JSON string).

    This format is familiar to instruction-tuned models and unambiguously
    describes both name, description, and input parameters.
    """
    spec: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {},
        },
    }
    return json.dumps(spec, indent=2)


# ===========================================================================
#                       ENRICHMENT
# ===========================================================================


@register_tool_enrichment("output_parameters")
class OutputParametersEnrichment(ToolEnrichment):
    """Infer ``output_parameters`` for tools that lack them using an LLM.

    Calls the LM once per tool that has an empty ``output_parameters`` dict.
    Tools that already have ``output_parameters`` are skipped entirely —
    no LLM tokens are spent on them.

    Results are cached under ``{DGT_CACHE_DIR}/enrichments/output_parameters/
    {fingerprint}.json`` and delta-merged on subsequent runs so that only tools
    missing from the cache incur LLM calls.

    Args:
        lm_config: Provider config dict passed to ``get_block``.  Must include
            a ``type:`` key, e.g. ``{"type": "ollama", "model_id_or_path": ...}``.
        force: If ``True``, bypass the cache and recompute all output_parameters
            from scratch.

    Raises:
        AssertionError: If ``lm_config`` is missing or has no ``type`` key.
    """

    def __init__(
        self,
        lm_config: Optional[Dict] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert (
            lm_config and TYPE_KEY in lm_config
        ), "OutputParametersEnrichment requires lm_config with a 'type' key"
        self._lm_config = lm_config
        self._force = force
        self._lm: LMProvider = get_block(lm_config[TYPE_KEY], **lm_config)

    def _cache_fingerprint(self, tools: List[Tool]) -> str:
        """Fingerprint: sorted (qualified_name, parameters) pairs + model ID."""
        tool_ids = sorted((t.qualified_name, t.parameters or {}) for t in tools)
        model_id = self._lm_config.get("model_id_or_path", "")
        return compute_fingerprint(tool_ids, model_id)

    def enrich(self, registry: Any) -> None:
        """Fill in ``output_parameters`` for tools that lack them.

        Args:
            registry: ``ToolRegistry`` instance.  Tools are mutated in-place.
        """
        all_tools = registry.all_tools()
        tools_missing = [t for t in all_tools if not t.output_parameters]
        if not tools_missing:
            self.logger.debug(
                "OutputParametersEnrichment: all tools already have output_parameters"
            )
            return

        # --- Cache lookup ---------------------------------------------------
        fingerprint = self._cache_fingerprint(all_tools)
        cache_path = enrichment_cache_path("output_parameters", fingerprint)

        cached: Dict[str, Any] = {}
        if not self._force:
            cached = load_cache(cache_path)
            if cached:
                self.logger.debug(
                    "OutputParametersEnrichment: loaded cache from %s (%d entries)",
                    cache_path,
                    len(cached),
                )

        # Apply cache hits immediately.
        for tool in list(tools_missing):
            if tool.qualified_name in cached:
                tool.output_parameters = cached[tool.qualified_name]

        tools_to_enrich = [t for t in tools_missing if not t.output_parameters]
        if not tools_to_enrich:
            self.logger.info(
                "OutputParametersEnrichment: all %d tool(s) satisfied from cache",
                len(tools_missing),
            )
            return

        self.logger.info(
            "OutputParametersEnrichment: inferring output_parameters for %d tool(s) "
            "(%d cache hits)",
            len(tools_to_enrich),
            len(tools_missing) - len(tools_to_enrich),
        )

        lm_inputs = [
            {
                "input": self._make_messages(tool),
                "gen_kwargs": {"response_format": {"type": "json_object"}},
                "tool": tool,
                "task_name": getattr(self.logger, "extra", {}).get("task_name", ""),
            }
            for tool in tools_to_enrich
        ]
        lm_outputs = self._lm(lm_inputs, method=LMProvider.CHAT_COMPLETION)

        new_entries: Dict[str, Any] = {}
        for lm_output in lm_outputs:
            tool = lm_output["tool"]
            schema = self._parse_schema(lm_output, tool)
            if schema:
                tool.output_parameters = schema
                new_entries[tool.qualified_name] = schema
                self.logger.debug(
                    "OutputParametersEnrichment: inferred schema for '%s'", tool.qualified_name
                )
            else:
                self.logger.warning(
                    "OutputParametersEnrichment: failed to infer schema for '%s'; skipping",
                    tool.qualified_name,
                )

        if new_entries:
            save_cache(cache_path, new_entries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_messages(self, tool: Tool) -> List[Dict[str, str]]:
        user_content = _USER_TEMPLATE.format(tool_spec=_tool_to_openai_spec(tool))
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_schema(self, lm_output: Any, tool: Tool) -> Optional[Dict]:
        """Extract a valid JSON Schema object from a single LM output entry."""
        prediction = lm_output if isinstance(lm_output, dict) else {}
        result = prediction.get("result")
        candidates = result if isinstance(result, list) else [result]

        for res in candidates:
            text = ((res or {}).get("content") or "").strip()
            schema = try_parse_json_string(text)
            if not isinstance(schema, dict):
                continue
            # Accept if it looks like a JSON Schema object
            if "properties" in schema or schema.get("type") == "object":
                schema.setdefault("type", "object")
                return schema
            # Unwrap a wrapping layer like {"output_parameters": {...}}
            for val in schema.values():
                if isinstance(val, dict) and ("properties" in val or val.get("type") == "object"):
                    val.setdefault("type", "object")
                    return val
            # Flat properties map: every value is a dict (model skipped the wrapper)
            if schema and all(isinstance(v, dict) for v in schema.values()):
                return {"type": "object", "properties": schema}

        self.logger.debug(
            "OutputParametersEnrichment: unparseable LM output for '%s': %r",
            tool.name,
            lm_output,
        )
        return None

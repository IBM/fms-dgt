# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import logging
import random

# Third Party
import jsonschema

# Local
from fms_dgt.base.registry import get_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.tools.constants import (
    TOOL_PROPERTIES,
)
from fms_dgt.core.tools.data_objects import Tool, ToolCall, ToolResult
from fms_dgt.core.tools.engines.base import ToolEngine, register_tool_engine
from fms_dgt.core.tools.registry import ToolRegistry
from fms_dgt.utils import try_parse_json_string

logger = logging.getLogger(__name__)

# ===========================================================================
#                       CONSTANTS
# ===========================================================================
_HISTORY_KEY = "tool_executions"

_SYSTEM_PROMPT = """\
You are a tool call simulator.

You will be given a tool specification and a tool call. Your task is to simulate \
a realistic, successful result for that tool call.

The result you generate must meet the following criteria:
- Return a JSON object that conforms to the tool's output_parameters schema.
- Include at least all required fields from the output_parameters schema.
- Simulate realistic values based on the tool description and the arguments supplied.
- Assume the operation succeeded — do not include error messages, "not found", \
"unsuccessful", or similar failure indicators.
- Use only valid UTF-8 characters.\
"""


# ===========================================================================
#                       ERROR CATEGORIES
# ===========================================================================


@dataclass
class ErrorCategory:
    """Probabilistic error injection descriptor.

    Attributes:
        type: One of ``"network_error"``, ``"unparseable_result"``,
            ``"schema_violation"``.
        probability: Float in [0, 1] — chance this category fires on any
            given ``simulate()`` call.
        message: Optional human-readable error string.  Used for
            ``"network_error"`` results.
    """

    type: str
    probability: float
    message: str = "Tool execution failed"

    def should_fire(self) -> bool:
        return random.random() < self.probability


# ===========================================================================
#                       LM TOOL ENGINE
# ===========================================================================


@register_tool_engine("lm")
class LMToolEngine(ToolEngine):
    """LM-simulated tool executor.

    An LM generates realistic tool outputs given the tool spec and call history.
    Structured decoding (``response_format``) is used when the tool exposes an
    ``output_parameters`` schema; otherwise JSON-object mode is requested.

    Constructor mirrors ``LMJudgeValidator`` — the ``lm_config`` kwarg holds
    the provider config dict (must include ``type:``).

    Session state is a list of ``{"tool_call": ..., "tool_output": ...}`` dicts
    (serializable plain Python, safe for DPO session forking).

    Args:
        registry: Shared ``ToolRegistry`` for name lookup at dispatch time.
        lm_config: Provider config dict.  Must include ``type:``.
        error_categories: Optional list of error-injection descriptors (dicts
            or ``ErrorCategory`` instances).  Sampled before each ``simulate``
            call.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        lm_config: Optional[Dict] = None,
        error_categories: Optional[List[Dict]] = None,
        namespaces: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(registry, namespaces=namespaces)
        assert (
            lm_config and TYPE_KEY in lm_config
        ), "LMToolEngine requires lm_config with a 'type' key"
        self._lm: LMProvider = get_block(lm_config[TYPE_KEY], **lm_config)
        self._error_categories: List[ErrorCategory] = [
            ec if isinstance(ec, ErrorCategory) else ErrorCategory(**ec)
            for ec in (error_categories or [])
        ]

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def _init_session_state(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {_HISTORY_KEY: []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Generate tool results without permanently updating session state.

        Within the batch, earlier results are visible to later calls.
        All mutations are rolled back before returning.

        Args:
            session_id: Active session (must have been set up).
            tool_calls: Tool calls to simulate, processed in order.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.

        Raises:
            KeyError: If ``session_id`` has not been set up.
        """
        with self._session_transaction(session_id, rollback=True) as state:
            return self._run_sequence(tool_calls, state[_HISTORY_KEY])

    def execute(self, session_id: str, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls and permanently append results to session history.

        Within the batch, earlier results are visible to later calls.
        Failed results are appended too — the assistant must handle errors.

        Args:
            session_id: Active session.
            tool_calls: Tool calls to execute, processed in order.

        Returns:
            One ``ToolResult`` per ``ToolCall``, in the same order.

        Raises:
            KeyError: If ``session_id`` has not been set up.
        """
        with self._session_transaction(session_id) as state:
            return self._run_sequence(tool_calls, state[_HISTORY_KEY])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_sequence(self, tool_calls: List[ToolCall], history: List[Dict]) -> List[ToolResult]:
        """Process tool calls in order, appending each result to history.

        The caller is responsible for rolling back history if needed
        (simulate) or leaving it intact (execute).
        """
        results = []
        for tool_call in tool_calls:
            result = self._execute_one(tool_call, history)
            history.append(
                {
                    "tool_call": tool_call.to_dict(),
                    "tool_output": result.to_dict(),
                }
            )
            results.append(result)
        return results

    def _execute_one(self, tool_call: ToolCall, history: List[Dict]) -> ToolResult:
        """Generate one tool result given the current history snapshot."""
        # 1. Check error injection before touching the LM.
        # Sample all categories independently so low-probability ones are not
        # starved by high-probability ones appearing earlier in the list.
        fired = [ec for ec in self._error_categories if ec.should_fire()]
        if fired:
            return self._make_error_result(tool_call, random.choice(fired))

        # 2. Look up the tool definition within this engine's namespace scope.
        tool = self._catalog.match(tool_call, namespaces=self._namespaces)
        if tool is None:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=f"Unknown tool '{tool_call.name}'",
            )

        # 3. Build response_format from output_parameters.
        if tool.output_parameters:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": tool.name,
                    "schema": tool.output_parameters,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        # 4. Build prompt and call the LM.
        messages = self._make_prompt(tool_call, tool, history)
        lm_input = [{"input": messages, "gen_kwargs": {"response_format": response_format}}]
        lm_outputs = self._lm(lm_input, method=LMProvider.CHAT_COMPLETION)

        # 5. Parse and validate output.
        output, error_message = self._parse_output(lm_outputs, tool)

        if error_message is not None:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error=error_message,
            )
        if output is None:
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                error="failed to generate valid tool result",
            )
        return ToolResult(
            call_id=tool_call.call_id,
            name=tool_call.name,
            result=output,
        )

    def _make_error_result(self, tc: ToolCall, ec: ErrorCategory) -> ToolResult:
        if ec.type == "network_error":
            return ToolResult(
                call_id=tc.call_id,
                name=tc.name,
                error=ec.message,
            )
        if ec.type == "unparseable_result":
            return ToolResult(
                call_id=tc.call_id,
                name=tc.name,
                result="<garbled: simulated unparseable result>",
            )
        # schema_violation: treat as generic error at this layer.
        return ToolResult(
            call_id=tc.call_id,
            name=tc.name,
            error=ec.message,
        )

    def _make_prompt(self, tool_call: ToolCall, tool: Tool, history: List[Dict]) -> List[Dict]:
        """Build the message list for the LM.

        Structure:
          - system: static instructions
          - For each history entry: user turn (tool spec + call), assistant turn (tool result)
          - Final user turn: tool spec + current call
        """
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        for entry in history:
            tc_dict = entry["tool_call"]
            # Reconstruct the tool spec for the historical call from the full
            # catalog — no namespace filter here since history may include calls
            # handled by other engines in a multi-engine session.
            hist_tool = self._catalog.match(ToolCall.from_dict(tc_dict))
            hist_tool_spec = (
                json.dumps([hist_tool.to_dict()], indent=1) if hist_tool is not None else "{}"
            )
            messages.append(
                {
                    "role": "user",
                    "content": _user_turn(hist_tool_spec, json.dumps(tc_dict, indent=1)),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(entry["tool_output"].get("result", {}), indent=1),
                }
            )

        tool_spec = json.dumps([tool.to_dict()], indent=1)
        messages.append(
            {
                "role": "user",
                "content": _user_turn(tool_spec, json.dumps(tool_call.to_dict(), indent=1)),
            }
        )

        return messages

    def _parse_output(
        self,
        lm_outputs: Any,
        tool: Tool,
    ):
        """Extract and validate tool output from the LM response.

        Parsing strategy differs based on whether the tool declares
        ``output_parameters``:

        **With ``output_parameters``:**
        1. Try ``json_obj`` as-is against the schema.
        2. If that fails and ``"result"`` is present, is not a declared schema
           property, and its value is a dict, try the unwrapped value.  This
           handles models that wrap their output in a ``{"result": {...}}``
           envelope despite structured-output mode.
        3. Both fail → return ``(None, None)``.

        **Without ``output_parameters``:**
        Return ``json_obj`` as-is with no validation or unwrapping.  The LM
        output shape is undefined without a schema, so the framework makes no
        assumptions.  Override ``_parse_output`` in a subclass to add custom
        normalization for specific models or tools.

        Returns:
            Tuple of ``(output_dict | None, error_message | None)``.
        """
        # We always send a single request, so only the first element is relevant.
        prediction = lm_outputs[0] if lm_outputs else None
        result = (prediction or {}).get("result")
        candidates = result if isinstance(result, list) else [result]

        for res in candidates:
            text = ((res or {}).get("content") or "").strip()
            json_obj = try_parse_json_string(text)
            if not (json_obj and isinstance(json_obj, dict)):
                continue

            error_msg = json_obj.get("error")
            if error_msg:
                return None, str(error_msg)

            # No output_parameters: return raw parsed dict, no assumptions.
            if not tool.output_parameters:
                logger.debug(
                    "Tool '%s' has no output_parameters — returning raw LM output",
                    tool.name,
                )
                return json_obj, None

            output_props = tool.output_parameters.get(TOOL_PROPERTIES) or {}

            # Try the full object first.
            filtered = _filter_output_vals(json_obj, output_props)
            if self._is_valid_tool_result(filtered, tool):
                return filtered, None

            # Fallback: unwrap a "result" envelope if the LM added one.
            # Only attempt when "result" is not a declared schema property and
            # its value is a dict — scalars and legitimate "result" properties
            # are never candidates for unwrapping.
            unwrapped = json_obj.get("result")
            if "result" not in output_props and isinstance(unwrapped, dict):
                filtered_unwrapped = _filter_output_vals(unwrapped, output_props)
                if self._is_valid_tool_result(filtered_unwrapped, tool):
                    return filtered_unwrapped, None

        return None, None

    def _is_valid_tool_result(self, output: Dict, tool: Tool) -> bool:
        """Validate parsed output against the tool's output_parameters schema.

        When ``output_parameters`` is present, validates with ``jsonschema``.
        When absent, a non-empty dict is sufficient (json_object mode).
        """
        if not tool.output_parameters:
            return bool(output)
        try:
            jsonschema.validate(instance=output, schema=tool.output_parameters)
            return True
        except jsonschema.ValidationError:
            return False


# ===========================================================================
#                       HELPERS
# ===========================================================================


def _user_turn(tool_spec: str, tool_call: str) -> str:
    """Format a user turn containing a tool spec and tool call."""
    return f"Tool specification:\n{tool_spec}\n\nTool call:\n{tool_call}"


def _filter_output_vals(inp: Dict, props: Dict) -> Dict:
    """Strip empty values and, when a schema is present, keep only known keys.

    When ``props`` is empty (no ``output_parameters`` schema), all keys are
    kept and only empty values are stripped.
    """

    def _filter(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _filter(v) for k, v in d.items() if v not in ([], "", None, {})}
        if isinstance(d, (list, tuple)):
            return type(d)(_filter(v) for v in d)
        return d

    projected = {k: v for k, v in inp.items() if k in props} if props else inp
    return _filter(projected)

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json

# Local
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)


@register_formatter("formatters/conversation/messages")
class ConversationMessagesFormatter(Formatter):
    """Formats a ConversationDataPoint into a standard SFT messages dict.

    Output shape:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

    Handles all conversational step types in OpenAI-compatible format:
    - UserStep → {"role": "user", "content": "..."}
    - AssistantStep → {"role": "assistant", "content": "..."}
    - ToolCallStep → {"role": "assistant", "tool_calls": [...]} with arguments as JSON string
    - ToolResultStep → {"role": "tool", "tool_call_id": "...", "content": "..."}

    Pipeline-internal steps (scenario, persona, flow_controller) are stripped.
    This format is directly compatible with HuggingFace TRL, OpenAI fine-tuning,
    and most other SFT frameworks.
    """

    def apply(
        self, data: ConversationDataPoint, *args: Any, **kwargs: Any
    ) -> Dict[str, List[Dict[str, Any]]]:
        messages = []
        for step in data.steps:
            if isinstance(step, UserStep):
                messages.append({"role": "user", "content": step.content})
            elif isinstance(step, AssistantStep):
                messages.append({"role": "assistant", "content": step.content})
            elif isinstance(step, ToolCallStep):
                tc = step.content
                arguments = tc.get("arguments", {})
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": (
                                        json.dumps(arguments)
                                        if isinstance(arguments, dict)
                                        else arguments
                                    ),
                                },
                            }
                        ],
                    }
                )
            elif isinstance(step, ToolResultStep):
                tr = step.content
                result = tr.get("result")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("call_id"),
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    }
                )
        return {"conversation_id": data.conversation_id, "messages": messages}

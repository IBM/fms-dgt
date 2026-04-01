# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.formatter import Formatter
from fms_dgt.base.registry import register_formatter
from fms_dgt.core.databuilders.conversation.data_objects import ConversationDataPoint


@register_formatter("formatters/conversation/messages")
class ConversationMessagesFormatter(Formatter):
    """Formats a ConversationDataPoint into a standard SFT messages dict.

    Output shape:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

    Only user and assistant steps are kept. All pipeline-internal steps
    (scenario, persona, flow_controller) are stripped. This format is
    directly compatible with HuggingFace TRL, OpenAI fine-tuning, and
    most other SFT frameworks.
    """

    def apply(
        self, data: ConversationDataPoint, *args: Any, **kwargs: Any
    ) -> Dict[str, List[Dict[str, str]]]:
        messages = [
            {"role": step.role, "content": step.content}
            for step in data.steps
            if step.role in ("user", "assistant")
        ]
        return {"conversation_id": data.conversation_id, "messages": messages}

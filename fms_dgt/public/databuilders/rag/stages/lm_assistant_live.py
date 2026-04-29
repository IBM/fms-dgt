# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, List
import dataclasses
import json
import logging
import uuid

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    ConversationDataPoint,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.tools.data_objects import ToolCall
from fms_dgt.core.tools.engines.composite import CompositeToolEngine

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = (
    "You have just retrieved information using a tool. "
    "Answer the user's question directly using the tool results above. "
    "Do not call any more tools."
)


def _build_history_messages(steps: list) -> list:
    """Serialize ctx.steps to OpenAI-compatible chat messages.

    Handles UserStep, AssistantStep, ToolCallStep, and ToolResultStep.
    Pipeline-internal steps (scenario, persona, flow_controller) are skipped.
    """
    messages = []
    for step in steps:
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
                    "content": None,
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
    return messages


@register_stage("lm/assistant/rag")
class LiveRetrievalAssistantStage(Stage):
    """Pattern 2 assistant stage: model-driven tool selection with live retrieval.

    For each conversation turn the model decides whether to call a tool based
    solely on the conversation history. All tools registered in the composite
    engine's catalog are made available; the model picks the right one.

    When the model issues tool calls:
      1. ToolCallStep(s) are appended to ctx.steps.
      2. The composite engine executes the calls (routing by namespace).
      3. ToolResultStep(s) are appended to ctx.steps.
      4. A second LM call synthesizes a final answer from the tool results.
      5. An AssistantStep is appended.

    When the model responds with content only (e.g. clarification questions,
    ambiguous turns where retrieval is not needed):
      1. An AssistantStep is appended directly.

    Both outcome types are valid training examples. The scenario stage controls
    the distribution of turn types; this stage is policy-neutral.
    """

    def __init__(
        self,
        *,
        name: str,
        generator: LMProvider,
        composite_tool_engine: CompositeToolEngine,
        **kwargs: Any,
    ) -> None:
        """Initialize LiveRetrievalAssistantStage.

        Args:
            name: Stage registry name.
            generator: LM provider for tool-selection and synthesis calls.
            composite_tool_engine: Engine wrapping all registered retrieval tools.
                Its catalog is used to build the tool list passed to the model.
        """
        super().__init__(name=name)
        self._generator = generator
        self._engine = composite_tool_engine

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs: Any,
    ) -> List[ConversationDataPoint]:
        tools = self._engine.catalog.to_dicts(qualified=True)

        # LM call #1: present full conversation history + all tools, let model decide.
        call1_inputs = [
            {
                "input": _build_history_messages(data_point.steps),
                "tools": tools,
                "tool_choice": "auto",
                "gen_kwargs": {"max_new_tokens": 512},
                "reference": data_point,
                "task_name": data_point.task_name,
            }
            for data_point in data_points
        ]

        if not call1_inputs:
            return []

        call1_outputs = self._generator(
            call1_inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True
        )

        # Partition by whether model called a tool or responded with content.
        tool_call_batch: List[tuple] = []  # (ctx, tool_calls list)
        direct_results: List[ConversationDataPoint] = []

        for out in call1_outputs:
            result = out.get("result")
            ctx: ConversationDataPoint = out["reference"]

            if isinstance(result, dict) and result.get("tool_calls"):
                tool_call_batch.append((ctx, result["tool_calls"]))
            else:
                # Content-only response: clarification, ambiguous turn, closing, etc.
                text = ""
                if isinstance(result, dict):
                    text = result.get("content") or ""
                elif isinstance(result, str):
                    text = result
                text = text.strip()
                if text:
                    ctx.steps.append(AssistantStep(content=text, stage_name=self.name))
                    direct_results.append(ctx)

        if not tool_call_batch:
            return direct_results

        # Execute tool calls and collect retrieved results for synthesis.
        synthesis_batch: List[tuple] = []  # (ctx, list of tool results)

        for ctx, raw_tool_calls in tool_call_batch:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", {}),
                        call_id=tc.get("id") or str(uuid.uuid4()),
                    )
                )
                ctx.steps.append(
                    ToolCallStep(
                        content=dataclasses.asdict(tool_calls[-1]),
                        stage_name=self.name,
                    )
                )

            try:
                tool_results = self._engine.execute(
                    session_id=ctx.conversation_id,
                    tool_calls=tool_calls,
                )
            except Exception as exc:
                logger.warning("Tool execution failed for session %s: %s", ctx.conversation_id, exc)
                continue

            for tool_result in tool_results:
                ctx.steps.append(
                    ToolResultStep(
                        content=dataclasses.asdict(tool_result),
                        stage_name=self.name,
                    )
                )

            synthesis_batch.append((ctx, tool_results))

        if not synthesis_batch:
            return direct_results

        # LM call #2: synthesize answer from tool results. No tools param so
        # the model cannot call again (enforces single-round tool use).
        call2_inputs = [
            {
                "input": [{"role": "system", "content": _SYNTHESIS_SYSTEM}]
                + _build_history_messages(ctx.steps),
                "gen_kwargs": {"max_new_tokens": 1024},
                "reference": ctx,
                "task_name": ctx.task_name,
            }
            for ctx, _ in synthesis_batch
        ]

        call2_outputs = self._generator(
            call2_inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True
        )

        synthesis_results: List[ConversationDataPoint] = []
        for out in call2_outputs:
            result = out.get("result")
            ctx: ConversationDataPoint = out["reference"]
            text = ""
            if isinstance(result, dict):
                text = result.get("content") or ""
            elif isinstance(result, str):
                text = result
            text = text.strip()
            if not text:
                continue
            ctx.steps.append(AssistantStep(content=text, stage_name=self.name))
            synthesis_results.append(ctx)

        return direct_results + synthesis_results

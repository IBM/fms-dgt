# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Any, Dict

# Local
from fms_dgt.core.databuilders.conversation.data_objects import Step, UserStep
from fms_dgt.core.tools.data_objects import Tool, ToolCall


# ===========================================================================
#                       DATACLASSES
# ===========================================================================
@dataclass(kw_only=True)
class ToolInfoStep(Step):
    """Scenario step data for tool calling scenarios.

    Attributes:
        type: Step type identifier ("tool_calling/scenario").
        namespace: Namespace of the tools being used.
        tools: List of tool dictionaries available in this scenario.
        details: Additional scenario details.
        date: Date/time context for the scenario.
    """

    role: str = field(default="tc/sampler", init=False)
    tools: list[Tool]
    date: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolInfoStep":
        return cls(
            tools=[Tool.from_dict(t) for t in d["tools"]],
            date=d["date"],
        )


@dataclass(kw_only=True)
class ToolPlanStep(Step):
    """Plan step data for tool calling plans.

    Attributes:
        type: Step type identifier ("tool_calling/plan").
        plan: List of planned tool calls or steps.
    """

    role: str = field(default="tc/plan", init=False)
    plan: list[ToolCall] | None = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolPlanStep":
        return cls(
            plan=[ToolCall.from_dict(tc) for tc in d["plan"]] if d["plan"] is not None else None,
        )


@dataclass(kw_only=True)
class ToolUserStep(UserStep):
    """User step data for tool calling interactions.

    Attributes:
        type: Step type identifier ("tool_calling/user").
        required_terms: Terms that must be present in the user query.
        hidden_set: Set of tools or information hidden from the user.
    """

    role: str = field(default="tc/user", init=False)
    required_terms: list | None = None
    hidden_set: list[dict] | None = None

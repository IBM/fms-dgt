# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from copy import deepcopy
from typing import Dict, List
import datetime
import random

# Local
from fms_dgt.core.blocks.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import ConversationDataPoint
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.tools.registry import ToolRegistry
from fms_dgt.core.tools.samplers.base import ToolSampler, get_tool_sampler
from fms_dgt.public.databuilders.tool_calling.multi_turn.stateless.data_objects import (
    ToolInfoStep,
)


@register_stage("tool_calling/multi_turn/stages/scenario_generator")
class ToolCallingSelectionScenarioStage(Stage):
    """Scenario generator for tool calling conversations.

    Generates contextual scenarios for tool calling tasks, including tool
    selection, persona assignment, and scenario details using an LLM.
    """

    def __init__(
        self,
        *args,
        generator: LMProvider,
        sampler_mix: List[Dict],
        tool_registry: ToolRegistry,
        **kwargs,
    ):
        """Initialize the ToolCallingSelectionScenarioGenerator.

        Args:
            *args: Positional arguments passed to parent.
            lm: Language model provider for scenario generation.
            tool_handler: Handler for managing available tools.
            personas_path: Optional path to personas file.
            **kwargs: Additional keyword arguments passed to parent.
        """
        super().__init__(*args, **kwargs)

        # set tool handler and lm
        self._lm = generator
        self._tool_registry = tool_registry

        self._samplers: List[ToolSampler] = []
        self._sampler_weights: List[float] = []
        for sampler_info in sampler_mix:
            sampler_kwargs: dict = sampler_info.get("sampler")
            sampler = get_tool_sampler(
                name=sampler_kwargs.get("type"),
                registry=self._tool_registry,
                **sampler_kwargs,
            )
            self._samplers.append(sampler)
            self._sampler_weights.append(sampler_info.get("weight", 1))

    # ===========================================================================
    #                       MAIN FUNCTIONS
    # ===========================================================================
    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs,
    ) -> List[ConversationDataPoint]:
        """Generate scenarios for tool calling conversations.

        Args:
            data_points: List of data points to generate scenarios for.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List of data points with generated scenarios added.
        """
        assert len(data_points) == 1
        data_point = data_points[0]

        # Create deep copy
        updated_data_point = deepcopy(data_point)

        sampler = random.choices(self._samplers, weights=self._sampler_weights)[0]
        tools = sampler.sample()
        date = _get_random_date()
        updated_data_point.steps.append(
            ToolInfoStep(
                tools=tools,
                date=date,
            )
        )

        return [updated_data_point]


def _get_random_date():
    """ """
    start_date, end_date = datetime.date(1990, 1, 1), datetime.date(2025, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return str(random_date)

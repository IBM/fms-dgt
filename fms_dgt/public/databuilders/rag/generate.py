# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict

# Local
from fms_dgt.base.registry import register_data_builder
from fms_dgt.core.databuilders.conversation.generate import ConversationDataBuilder
from fms_dgt.core.databuilders.conversation.registry import get_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.task import ConversationTask
from fms_dgt.public.databuilders.rag.task import RAGConversationTask

# Ensure RAG step types and stage classes are registered when this module loads.
import fms_dgt.public.databuilders.rag.data_objects  # noqa: F401
import fms_dgt.public.databuilders.rag.stages  # noqa: F401


@register_data_builder("public/rag/conversation")
class RAGDataBuilder(ConversationDataBuilder):
    """Databuilder for RAG data generation supporting Pattern 1 and Pattern 2.

    Extends ConversationDataBuilder by injecting the task's named engine dict
    into every stage constructor. Stages that need an engine (scenario stage,
    live assistant stage) resolve engine names from ``engines`` directly.
    Stages that do not need engines absorb the kwarg via ``**kwargs``.

    Every RAG task must define a ``tools:`` block. ``Task.__init__`` builds
    ``_engines`` (dict mapping engine name → ToolEngine instance)
    automatically; this databuilder passes that dict through to stages.

    Pattern 1 (static context):
        Use ``lm/scenario/rag`` as the initialization stage with a
        ``document_samplers`` list. The stage resolves engine names and
        constructs DocumentSampler instances at stage init time.
        Use ``lm/assistant/rag/static`` as the assistant iteration stage.

    Pattern 2 (live retrieval):
        Use ``lm/scenario/rag/live`` as the initialization stage (no samplers).
        Use ``lm/assistant/rag`` with ``retriever_engine: <engine_name>`` as the
        assistant iteration stage. Each turn appends ToolCallStep + ToolResultStep.
    """

    TASK_TYPE = RAGConversationTask

    def _build_stage(
        self,
        config: Dict,
        block_kwargs: Dict[str, Any],
        **extra: Any,
    ) -> Stage:
        """Extend base _build_stage to forward engine collections to every stage.

        Keyword arguments in ``extra`` are forwarded to the stage constructor
        alongside the standard block kwargs. Stages that need them declare the
        params by name; stages that do not absorb them via ``**kwargs``.
        """
        cfg = dict(config)
        name = cfg.get("name", "?")

        stage_blocks: Dict[str, Any] = {}
        for arg_name, block_name in cfg.pop("blocks", {}).items():
            if block_name not in block_kwargs:
                raise ValueError(
                    f"Stage '{name}' references block '{block_name}' "
                    f"(as '{arg_name}') which is not defined in the databuilder. "
                    f"Available blocks: {sorted(block_kwargs)}"
                )
            stage_blocks[arg_name] = block_kwargs[block_name]

        for key in list(cfg.keys()):
            if isinstance(cfg[key], str) and cfg[key] in block_kwargs:
                stage_blocks[key] = block_kwargs[cfg.pop(key)]

        remaining_blocks = {k: v for k, v in block_kwargs.items() if k not in stage_blocks}
        cfg.pop("name")
        stage_cls = get_stage(name)
        return stage_cls(name=name, **remaining_blocks, **stage_blocks, **cfg, **extra)

    def _init_stages(self, task: ConversationTask) -> None:
        """Initialize stages, forwarding both engine collections to each stage.

        Passes ``component_tool_engines`` (dict of engine name → ToolEngine,
        used by scenario stages for document samplers) and ``composite_tool_engine``
        (the single CompositeToolEngine wrapping all sub-engines, used by the
        assistant stage for registry-driven tool calling). Stages take what they
        need and absorb the rest via ``**kwargs``.
        """
        block_kwargs: Dict[str, Any] = {b.name: b for b in self._blocks}
        extra: Dict[str, Any] = {}
        if task.component_tool_engines:
            extra["component_tool_engines"] = task.component_tool_engines
        if task.tool_engine is not None:
            extra["composite_tool_engine"] = task.tool_engine

        task.initialization_stages = [
            self._build_stage(cfg, block_kwargs, **extra)
            for cfg in task.initialization_stage_configs
        ]
        task.iteration_stages = [
            self._build_stage(cfg, block_kwargs, **extra) for cfg in task.iteration_stage_configs
        ]
        self._stages_initialized.add(task.name)

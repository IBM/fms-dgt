# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RAG databuilder.

Coverage:
- RAGScenarioStep: registration, serialization round-trip
- RAGScenarioStage: documents stored on step, scenario text appended
- StaticContextAssistantStage (static): documents injected from scenario step,
  AssistantStep appended, no ToolCallStep/ToolResultStep in context
- LiveRetrievalAssistantStage (live): ToolCallStep + ToolResultStep appended,
  AssistantStep appended, query text passed to engine

All tests are fully mocked — no LM calls, no file I/O, no network.
"""

# Standard
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Third Party
import pytest

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    ScenarioStep,
    Step,
)
from fms_dgt.core.databuilders.conversation.registry import get_step
from fms_dgt.core.tools.data_objects import Tool, ToolResult
from fms_dgt.core.tools.engines.composite import CompositeToolEngine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.engines.search.samplers.base import DocumentSampler
from fms_dgt.core.tools.registry import ToolRegistry
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep
from fms_dgt.public.databuilders.rag.stages.lm_assistant_live import (
    LiveRetrievalAssistantStage,
)
from fms_dgt.public.databuilders.rag.stages.lm_assistant_static import (
    StaticContextAssistantStage,
)
from fms_dgt.public.databuilders.rag.stages.lm_scenario_rag import RAGScenarioStage

# ===========================================================================
#                       HELPERS
# ===========================================================================


def _blank_context(task_name: str = "test") -> ConversationDataPoint:
    return ConversationDataPoint(task_name=task_name)


def _docs(*texts: str) -> List[Document]:
    return [Document(id=str(i), text=t) for i, t in enumerate(texts)]


def _mock_generator(response: str) -> MagicMock:
    """Return a mock LMProvider that always replies with ``response``."""
    gen = MagicMock()
    gen.side_effect = lambda inputs, **kw: [
        {"result": {"content": response}, "reference": inp["reference"]} for inp in inputs
    ]
    return gen


def _mock_sampler(documents: List[Document]) -> MagicMock:
    sampler = MagicMock(spec=DocumentSampler)
    sampler.sample = MagicMock(return_value=documents)
    return sampler


def _mock_search_engine(documents: List[Document]) -> MagicMock:
    """Mock SearchToolEngine that returns ``documents`` for any tool call."""
    engine = MagicMock(spec=SearchToolEngine)
    engine.execute = MagicMock(
        side_effect=lambda session_id, tool_calls: [
            ToolResult(
                call_id=tc.call_id,
                name=tc.name,
                result=[d.to_dict() for d in documents],
            )
            for tc in tool_calls
        ]
    )
    return engine


def _mock_search_engine_empty() -> MagicMock:
    """Mock SearchToolEngine that returns an empty result set."""
    engine = MagicMock(spec=SearchToolEngine)
    engine.execute = MagicMock(
        side_effect=lambda session_id, tool_calls: [
            ToolResult(call_id=tc.call_id, name=tc.name, result=[]) for tc in tool_calls
        ]
    )
    return engine


def _engines(engine_map: Dict[str, Any]) -> Dict[str, Any]:
    """Return a plain dict of named engines (as RAGDataBuilder passes to stages)."""
    return engine_map


def _mock_composite_engine(
    search_engine: MagicMock,
    tool_name: str = "search_documents",
    namespace: str = "rag",
) -> MagicMock:
    """Build a mock CompositeToolEngine with one tool registered in its catalog."""
    tool = Tool(
        name=tool_name,
        namespace=namespace,
        description="Search for relevant documents.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    registry = ToolRegistry([tool])
    engine = MagicMock(spec=CompositeToolEngine)
    engine.catalog = registry
    engine.execute = search_engine.execute
    return engine


def _tool_call_result(tool_name: str, query: str, call_id: str = "call_1") -> Dict[str, Any]:
    """Build an LM result dict that looks like a tool-calling response."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": {"query": query}},
            }
        ],
    }


# ===========================================================================
#                       RAGScenarioStep
# ===========================================================================


class TestRAGScenarioStep:
    def test_role_is_rag_scenario(self):
        step = RAGScenarioStep(content="scenario text")
        assert step.role == "rag/scenario"

    def test_documents_default_empty(self):
        step = RAGScenarioStep(content="scenario")
        assert step.documents == []

    def test_documents_stored(self):
        docs = [{"id": "0", "text": "hello"}]
        step = RAGScenarioStep(content="scenario", documents=docs)
        assert step.documents == docs

    def test_reasoning_optional(self):
        step = RAGScenarioStep(content="scenario")
        assert step.reasoning is None

    def test_registered_in_step_registry(self):
        cls = get_step("rag/scenario")
        assert cls is RAGScenarioStep

    def test_serialization_round_trip(self):
        docs = [{"id": "1", "text": "doc text", "score": 0.9}]
        step = RAGScenarioStep(
            content="test scenario",
            documents=docs,
            reasoning="relevant documents",
            scenario_family_id="fam-001",
            stage_name="lm/scenario/rag",
        )
        d = step.to_dict()
        assert d["role"] == "rag/scenario"
        assert d["documents"] == docs

        restored = Step.from_dict(d)
        assert isinstance(restored, RAGScenarioStep)
        assert restored.documents == docs
        assert restored.reasoning == "relevant documents"
        assert restored.scenario_family_id == "fam-001"


# ===========================================================================
#                       RAGScenarioStage (Pattern 1)
# ===========================================================================


class TestRAGScenarioStage:
    def _stage(self, generator, sampler, engine_name="file_retriever"):
        search_engine = MagicMock(spec=SearchToolEngine)
        engines = _engines({engine_name: search_engine})

        # Local
        import fms_dgt.public.databuilders.rag.stages.lm_scenario_rag as stage_mod

        original_get = stage_mod.get_document_sampler
        stage_mod.get_document_sampler = lambda name, engine, **kwargs: sampler
        try:
            stage = RAGScenarioStage(
                name="lm/scenario/rag",
                generator=generator,
                component_tool_engines=engines,
                document_samplers=[{"type": "search/random", "engine": engine_name, "weight": 1.0}],
                k=3,
            )
        finally:
            stage_mod.get_document_sampler = original_get

        return stage

    def test_appends_rag_scenario_step(self):
        docs = _docs("doc A", "doc B", "doc C")
        sampler = _mock_sampler(docs)
        gen = _mock_generator("A user wants to learn about Python.")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        results = stage([ctx])

        assert len(results) == 1
        rag_steps = [step for step in results[0].steps if step.role == "rag/scenario"]
        assert len(rag_steps) == 1

    def test_documents_stored_on_step(self):
        docs = _docs("doc A", "doc B")
        sampler = _mock_sampler(docs)
        gen = _mock_generator("Scenario text.")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        results = stage([ctx])

        step = [step for step in results[0].steps if step.role == "rag/scenario"][-1]
        assert len(step.documents) == 2
        assert step.documents[0]["text"] == "doc A"

    def test_scenario_text_stored_on_step(self):
        docs = _docs("doc A")
        sampler = _mock_sampler(docs)
        gen = _mock_generator("  A user wants info on climate.  ")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        results = stage([ctx])

        step = [step for step in results[0].steps if step.role == "rag/scenario"][-1]
        assert step.content == "A user wants info on climate."

    def test_sampler_called_with_session_id(self):
        docs = _docs("doc A")
        sampler = _mock_sampler(docs)
        gen = _mock_generator("Scenario.")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        stage([ctx])

        sampler.sample.assert_called_once_with(session_id=ctx.conversation_id, k=3)

    def test_drops_context_on_empty_lm_output(self):
        sampler = _mock_sampler(_docs("doc A"))
        gen = _mock_generator("")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        results = stage([ctx])

        assert results == []

    def test_processes_multiple_contexts(self):
        sampler = _mock_sampler(_docs("doc A"))
        gen = _mock_generator("Scenario.")
        stage = self._stage(gen, sampler)

        contexts = [_blank_context() for _ in range(3)]
        results = stage(contexts)

        assert len(results) == 3

    def test_scenario_family_id_set(self):
        sampler = _mock_sampler(_docs("doc A"))
        gen = _mock_generator("Scenario.")
        stage = self._stage(gen, sampler)

        ctx = _blank_context()
        results = stage([ctx])

        step = [step for step in results[0].steps if step.role == "rag/scenario"][-1]
        assert step.scenario_family_id is not None

    def test_seed_data_passed_to_icl_sampling(self):
        """Stage should not crash when seed_data is provided."""
        sampler = _mock_sampler(_docs("doc A"))
        gen = _mock_generator("Scenario.")
        stage = self._stage(gen, sampler)

        seed_ctx = ConversationDataPoint(task_name="test")
        seed_ctx.steps.append(RAGScenarioStep(content="seed scenario"))

        ctx = _blank_context()
        results = stage([ctx], seed_data=[seed_ctx])
        assert len(results) == 1

    def test_unknown_engine_raises_at_construction(self):
        """Stage construction must fail if engine name is not in engines."""
        with pytest.raises(ValueError, match="engine 'missing_engine'"):
            RAGScenarioStage(
                name="lm/scenario/rag",
                generator=_mock_generator("Scenario."),
                component_tool_engines={},
                document_samplers=[
                    {"type": "search/random", "engine": "missing_engine", "weight": 1.0}
                ],
            )

    def test_weights_not_summing_to_one_raises(self):
        """Stage construction must fail if sampler weights do not sum to 1.0."""
        search_engine = MagicMock(spec=SearchToolEngine)
        engines = _engines({"eng": search_engine})

        # Local
        import fms_dgt.public.databuilders.rag.stages.lm_scenario_rag as stage_mod

        original_get = stage_mod.get_document_sampler
        stage_mod.get_document_sampler = lambda name, engine, **kw: _mock_sampler([])
        try:
            with pytest.raises(ValueError, match="weights must sum to 1.0"):
                RAGScenarioStage(
                    name="lm/scenario/rag",
                    generator=_mock_generator("Scenario."),
                    component_tool_engines=engines,
                    document_samplers=[
                        {"type": "search/random", "engine": "eng", "weight": 0.4},
                        {"type": "search/random", "engine": "eng", "weight": 0.4},
                    ],
                )
        finally:
            stage_mod.get_document_sampler = original_get


# ===========================================================================
#                       StaticContextAssistantStage (Pattern 1)
# ===========================================================================


class TestStaticContextAssistantStage:
    def _stage(self, generator):
        return StaticContextAssistantStage(
            name="lm/assistant/rag/static",
            generator=generator,
        )

    def _ctx_with_rag_scenario(self, docs: List[Document]) -> ConversationDataPoint:
        ctx = _blank_context()
        ctx.steps.append(
            RAGScenarioStep(
                content="Scenario: user asks about Python.",
                documents=[d.to_dict() for d in docs],
            )
        )
        ctx.steps.append(Step(role="user", content="What is Python?"))
        return ctx

    def test_appends_assistant_step(self):
        gen = _mock_generator("Python is a programming language.")
        stage = self._stage(gen)

        ctx = self._ctx_with_rag_scenario(_docs("Python is popular."))
        results = stage([ctx])

        assert len(results) == 1
        assistant_steps = [step for step in results[0].steps if step.role == "assistant"]
        assert len(assistant_steps) == 1

    def test_no_tool_steps_appended(self):
        """Pattern 1 must produce no ToolCallStep or ToolResultStep."""
        gen = _mock_generator("Python is a language.")
        stage = self._stage(gen)

        ctx = self._ctx_with_rag_scenario(_docs("Python is popular."))
        results = stage([ctx])

        tool_steps = [
            step for step in results[0].steps if step.role in ("tool_call", "tool_result")
        ]
        assert tool_steps == []

    def test_assistant_text_stored(self):
        gen = _mock_generator("  Python is great.  ")
        stage = self._stage(gen)

        ctx = self._ctx_with_rag_scenario(_docs("doc text"))
        results = stage([ctx])

        assistant = [step for step in results[0].steps if step.role == "assistant"][-1]
        assert assistant.content == "Python is great."

    def test_falls_back_to_scenario_step_when_no_rag_step(self):
        """Stage should work with a plain ScenarioStep if no RAGScenarioStep exists."""
        gen = _mock_generator("Response text.")
        stage = self._stage(gen)

        ctx = _blank_context()
        ctx.steps.append(ScenarioStep(content="Plain scenario."))
        ctx.steps.append(Step(role="user", content="Tell me more."))
        results = stage([ctx])

        assert len(results) == 1

    def test_drops_on_empty_lm_output(self):
        gen = _mock_generator("")
        stage = self._stage(gen)

        ctx = self._ctx_with_rag_scenario(_docs("doc"))
        results = stage([ctx])

        assert results == []

    def test_empty_inputs_returns_empty(self):
        gen = _mock_generator("answer")
        stage = self._stage(gen)
        assert stage([]) == []


# ===========================================================================
#                       LiveRetrievalAssistantStage (Pattern 2)
# ===========================================================================


class TestLiveRetrievalAssistantStage:
    _TOOL_NAME = "rag::search_documents"

    def _stage(self, generator, search_engine, tool_name="search_documents", namespace="rag"):
        composite = _mock_composite_engine(search_engine, tool_name=tool_name, namespace=namespace)
        return LiveRetrievalAssistantStage(
            name="lm/assistant/rag",
            generator=generator,
            composite_tool_engine=composite,
        )

    def _ctx_with_user_turn(self) -> ConversationDataPoint:
        ctx = _blank_context()
        ctx.steps.append(RAGScenarioStep(content="User queries a knowledge base."))
        ctx.steps.append(Step(role="user", content="What is quantum computing?"))
        return ctx

    def _gen_tool_then_answer(
        self,
        search_engine,
        query="quantum computing",
        answer="Quantum computing uses qubits.",
        tool_name="search_documents",
        namespace="rag",
    ):
        """Build a generator mock that returns a tool call on call #1, plain text on call #2."""
        call_count = [0]
        qualified = f"{namespace}::{tool_name}"

        def side_effect(inputs, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"result": _tool_call_result(qualified, query), "reference": inp["reference"]}
                    for inp in inputs
                ]
            return [
                {"result": {"content": answer}, "reference": inp["reference"]} for inp in inputs
            ]

        gen = MagicMock()
        gen.side_effect = side_effect
        return gen

    def test_appends_tool_call_step(self):
        engine = _mock_search_engine(_docs("Quantum computing uses qubits."))
        gen = self._gen_tool_then_answer(engine)
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        assert len(results) == 1
        assert len([step for step in results[0].steps if step.role == "tool_call"]) == 1

    def test_appends_tool_result_step(self):
        engine = _mock_search_engine(_docs("Quantum computing uses qubits."))
        gen = self._gen_tool_then_answer(engine)
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        assert len([step for step in results[0].steps if step.role == "tool_result"]) == 1

    def test_appends_assistant_step(self):
        engine = _mock_search_engine(_docs("Quantum computing uses qubits."))
        gen = self._gen_tool_then_answer(engine, answer="Quantum computing is powerful.")
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        assistant_steps = [step for step in results[0].steps if step.role == "assistant"]
        assert len(assistant_steps) == 1
        assert assistant_steps[0].content == "Quantum computing is powerful."

    def test_tool_call_contains_query(self):
        engine = _mock_search_engine(_docs("doc"))
        query_text = "specific query for testing"
        gen = self._gen_tool_then_answer(engine, query=query_text)
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        tool_call_step = [step for step in results[0].steps if step.role == "tool_call"][-1]
        assert tool_call_step.content["arguments"]["query"] == query_text

    def test_tool_call_uses_tool_name_from_registry(self):
        engine = _mock_search_engine(_docs("doc"))
        gen = self._gen_tool_then_answer(engine, tool_name="my_retriever", namespace="custom")
        stage = self._stage(gen, engine, tool_name="my_retriever", namespace="custom")

        results = stage([self._ctx_with_user_turn()])

        tool_call_step = [step for step in results[0].steps if step.role == "tool_call"][-1]
        assert tool_call_step.content["name"] == "custom::my_retriever"

    def test_retrieval_result_stored_in_tool_result_step(self):
        docs = _docs("Quantum computers use superposition.")
        engine = _mock_search_engine(docs)
        gen = self._gen_tool_then_answer(engine)
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        tool_result_step = [step for step in results[0].steps if step.role == "tool_result"][-1]
        stored = tool_result_step.content["result"]
        assert isinstance(stored, list)
        assert stored[0]["text"] == "Quantum computers use superposition."

    def test_empty_retrieval_result_still_produces_assistant_step(self):
        engine = _mock_search_engine_empty()
        gen = self._gen_tool_then_answer(engine, answer="I cannot find relevant information.")
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        assert len(results) == 1
        assert len([step for step in results[0].steps if step.role == "assistant"]) == 1
        assert len([step for step in results[0].steps if step.role == "tool_call"]) == 1
        assert len([step for step in results[0].steps if step.role == "tool_result"]) == 1

    def test_engine_called_with_conversation_id(self):
        engine = _mock_search_engine(_docs("doc"))
        gen = self._gen_tool_then_answer(engine)
        stage = self._stage(gen, engine)

        ctx = self._ctx_with_user_turn()
        stage([ctx])

        engine.execute.assert_called_once()
        call_args = engine.execute.call_args
        assert call_args[1].get("session_id") == ctx.conversation_id or (
            call_args[0] and call_args[0][0] == ctx.conversation_id
        )

    def test_content_only_response_produces_assistant_step_directly(self):
        """When model returns content without tool calls, AssistantStep is appended directly."""
        engine = _mock_search_engine(_docs("doc"))
        gen = _mock_generator("I need more context before I can help.")
        stage = self._stage(gen, engine)

        results = stage([self._ctx_with_user_turn()])

        assert len(results) == 1
        assert len([step for step in results[0].steps if step.role == "assistant"]) == 1
        assert len([step for step in results[0].steps if step.role == "tool_call"]) == 0
        assert len([step for step in results[0].steps if step.role == "tool_result"]) == 0

    def test_empty_inputs_returns_empty(self):
        engine = _mock_search_engine(_docs("doc"))
        gen = _mock_generator("q")
        stage = self._stage(gen, engine)
        assert stage([]) == []

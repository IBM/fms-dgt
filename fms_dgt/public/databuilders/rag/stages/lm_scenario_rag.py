# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import json
import random
import uuid

# Local
from fms_dgt.core.blocks.llm.llm import LMProvider
from fms_dgt.core.databuilders.conversation.data_objects import (
    ConversationDataPoint,
    ScenarioStep,
)
from fms_dgt.core.databuilders.conversation.registry import register_stage
from fms_dgt.core.databuilders.conversation.stages.base import Stage
from fms_dgt.core.databuilders.conversation.utils import (
    get_last_step_of_type,
    rank_by_tfidf,
)
from fms_dgt.core.tools.engines.base import ToolEngine
from fms_dgt.core.tools.engines.search.base import Document, SearchToolEngine
from fms_dgt.core.tools.engines.search.samplers.base import get_document_sampler
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep

# ===========================================================================
#                       CONSTANT
# ===========================================================================

_SCENARIO_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step analysis of the documents before writing the scenario.",
                },
                "scenario": {
                    "type": "string",
                    "description": "The background scenario describing who the user is and what information need they have.",
                },
            },
            "required": ["reasoning", "scenario"],
            "additionalProperties": False,
        },
    },
}


# ===========================================================================
#                       HELPER METHODS
# ===========================================================================
def _render_document_list(documents: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        parts.append(f"[Document {i}]\n{doc.text}")
    return "\n\n".join(parts) if parts else "(no documents)"


def _truncate_doc_text(text: str, max_words: int = 300) -> str:
    """Return text as-is if under max_words, otherwise splice first/middle/last thirds."""
    words = text.split()
    n = len(words)
    if n <= max_words:
        return text
    third = max_words // 3
    start = words[:third]
    mid_start = (n - third) // 2
    middle = words[mid_start : mid_start + third]
    end = words[n - third :]
    return " ".join(start) + " [...] " + " ".join(middle) + " [...] " + " ".join(end)


def _render_seed_scenarios(seed_data: List[ConversationDataPoint]) -> str:
    """Render ICL examples as document-grounded scenario pairs.

    Each example shows the truncated documents that were used alongside the
    scenario written from them, so the model understands the grounding
    relationship between document content and scenario framing.
    """
    examples = []
    for entry in seed_data:
        step = get_last_step_of_type(entry.steps, RAGScenarioStep) or get_last_step_of_type(
            entry.steps, ScenarioStep
        )
        if not step:
            continue

        parts = []

        # Include truncated document text when available (RAGScenarioStep only)
        if isinstance(step, RAGScenarioStep) and step.documents:
            doc_parts = []
            for i, doc in enumerate(step.documents, start=1):
                text = doc.get("text", "") if isinstance(doc, dict) else getattr(doc, "text", "")
                truncated = _truncate_doc_text(text, max_words=300)
                doc_parts.append(f"[Document {i}]\n{truncated}")
            parts.append("Documents:\n" + "\n\n".join(doc_parts))

        example: Dict[str, str] = {"scenario": step.content, "reasoning": ""}
        if isinstance(step, RAGScenarioStep) and step.reasoning:
            example["reasoning"] = step.reasoning
        parts.append("Output:\n" + json.dumps(example, ensure_ascii=False))

        examples.append("\n".join(parts))

    return "\n\n---\n\n".join(examples) if examples else ""


def _build_scenario_messages(icl_text: str, doc_text: str) -> list:
    content = (
        "You are creating a background scenario for a multi-turn conversation grounded in the provided documents.\n\n"
        "Follow these steps:\n"
        "1. Read all documents and identify the domain and key information they contain.\n"
        "2. Identify a realistic information need that a user could have, which requires consulting these documents.\n"
        "3. Check whether the documents together support comparisons, gaps, or edge cases that would make the conversation richer.\n"
        "4. Write a background scenario describing who the user is, their context, and what they need to find out. "
        "The scenario must be grounded only in what the documents actually contain. "
        "It should be detailed enough that an assistant would need to reference the documents across multiple turns to fully satisfy the user's need.\n\n"
        "Return a JSON object with:\n"
        '- "reasoning": your step-by-step analysis (steps 1-3 above)\n'
        '- "scenario": the background scenario text'
    )
    if icl_text:
        content += (
            f"\n\nExamples (documents used and the scenario written from them):\n\n{icl_text}"
        )
    content += f"\n\n---\n\nNow write a scenario for these documents:\n{doc_text}"
    return [{"role": "user", "content": content}]


def _parse_scenario_output(raw: str) -> tuple[str, str | None]:
    """Parse LM output into (scenario_text, reasoning).

    Tries JSON parse first to extract structured fields. Falls back to
    treating the entire string as the scenario text if parsing fails or
    required fields are missing.
    """
    try:
        parsed = json.loads(raw)
        scenario = parsed.get("scenario", "").strip()
        reasoning = parsed.get("reasoning", "").strip() or None
        if scenario:
            return scenario, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass
    return raw.strip(), None


def _build_samplers(
    document_samplers: List[Dict],
    engines: Dict[str, ToolEngine],
) -> List[tuple]:
    """Build (sampler, weight) pairs from document_samplers config.

    Each entry in document_samplers must have ``type``, ``engine`` (engine name
    string matching a key in ``engines``), and optionally ``weight``
    (defaults to 1.0) plus any sampler-specific kwargs (``k``, ``method``, etc.).

    Raises:
        ValueError: If engine name is not found in engines.
        ValueError: If weights do not sum to 1.0.
    """
    pairs = []
    for entry in document_samplers:
        cfg = dict(entry)
        sampler_type = cfg.pop("type")
        engine_name = cfg.pop("engine")
        weight = cfg.pop("weight", 1.0)

        if engine_name not in engines:
            raise ValueError(
                f"document_samplers entry references engine '{engine_name}' "
                f"which is not in engines. "
                f"Available: {sorted(engines)}"
            )
        search_engine = engines[engine_name]
        if not isinstance(search_engine, SearchToolEngine):
            raise TypeError(
                f"Engine '{engine_name}' must be a SearchToolEngine for RAG samplers, "
                f"got {type(search_engine).__name__}."
            )
        sampler = get_document_sampler(sampler_type, engine=search_engine, **cfg)
        pairs.append((sampler, weight))

    total = sum(w for _, w in pairs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"document_samplers weights must sum to 1.0, got {total:.6f}. "
            f"Adjust the weight values so they sum to exactly 1.0."
        )
    return pairs


# ===========================================================================
#                       MAIN CLASSES
# ===========================================================================
@register_stage("lm/scenario/rag")
class RAGScenarioStage(Stage):
    """Initialization stage for Pattern 1 (static context) RAG recipes.

    Samples documents from a DocumentSampler, generates a conversation
    scenario grounded in those documents, and appends a RAGScenarioStep
    carrying both the scenario text and the selected document set.

    Samplers are constructed at stage init time by resolving engine names
    against ``tool_engine.engines``. ``sample()`` is called once per data
    point during generation.

    For live retrieval recipes, use ``lm/scenario/rag/live`` instead — no
    document sampling is needed at initialization time.
    """

    def __init__(
        self,
        *,
        name: str,
        generator: LMProvider,
        component_tool_engines: Dict[str, ToolEngine],
        document_samplers: List[Dict],
        k: int = 5,
        num_icl_examples: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize RAGScenarioStage.

        Args:
            name: Stage registry name.
            generator: LM provider used to generate the scenario text.
            component_tool_engines: Dict mapping engine name to ToolEngine instance,
                built by ``Task.__init__`` from the ``tools.engines`` block. Used to
                resolve engine name strings in ``document_samplers``.
            document_samplers: List of sampler config dicts, each with ``type``,
                ``engine`` (name string), ``weight``, and sampler-specific kwargs.
            k: Number of documents to sample per scenario.
            num_icl_examples: Number of ICL examples to use for scenario generation.
        """
        super().__init__(name=name)
        self._generator = generator
        self._sampler_pairs = _build_samplers(document_samplers, component_tool_engines)
        self._k = k
        self._num_icl_examples = num_icl_examples

    def _select_sampler(self):
        samplers, weights = zip(*self._sampler_pairs)
        return random.choices(samplers, weights=weights, k=1)[0]

    def __call__(
        self,
        data_points: List[ConversationDataPoint],
        seed_data: List[ConversationDataPoint] | None = None,
        **kwargs: Any,
    ) -> List[ConversationDataPoint]:
        seed_data = seed_data or []

        # Pre-extract and join document texts from seed_data for TF-IDF ranking.
        seed_doc_texts: List[str] = []
        for entry in seed_data:
            step = get_last_step_of_type(entry.steps, RAGScenarioStep)
            if step and step.documents:
                seed_doc_texts.append(
                    " ".join(
                        d.get("text", "") if isinstance(d, dict) else getattr(d, "text", "")
                        for d in step.documents
                    )
                )
            else:
                seed_doc_texts.append("")

        generator_inputs = []
        doc_batches: List[List[Document]] = []
        for data_point in data_points:
            sampler = self._select_sampler()
            documents = sampler.sample(
                session_id=data_point.conversation_id,
                k=self._k,
            )
            doc_batches.append(documents)

            if any(seed_doc_texts):
                reference = " ".join(doc.text for doc in documents)
                ranked_indices = rank_by_tfidf(reference, seed_doc_texts)
                sample = [
                    seed_data[i]
                    for i in ranked_indices[: min(self._num_icl_examples, len(seed_data))]
                ]
            else:
                # No embedded documents in seeds; fall back to random sampling.
                sample = random.sample(seed_data, min(self._num_icl_examples, len(seed_data)))
            icl_text = _render_seed_scenarios(sample)
            doc_text = _render_document_list(documents)

            generator_inputs.append(
                {
                    "input": _build_scenario_messages(icl_text, doc_text),
                    "gen_kwargs": {
                        "max_new_tokens": 1024,
                        "response_format": _SCENARIO_RESPONSE_FORMAT,
                    },
                    "reference": data_point,
                    "task_name": data_point.task_name,
                }
            )

        outputs = self._generator(
            generator_inputs, method=LMProvider.CHAT_COMPLETION, disable_tqdm=True
        )

        results = []
        for out, documents in zip(outputs, doc_batches):
            result = out.get("result") or ""
            if isinstance(result, dict):
                result = result.get("content") or ""
            raw = result.strip()
            if not raw:
                continue
            scenario_text, reasoning = _parse_scenario_output(raw)
            if not scenario_text:
                continue
            data_point: ConversationDataPoint = out["reference"]
            data_point.steps.append(
                RAGScenarioStep(
                    content=scenario_text,
                    stage_name=self.name,
                    scenario_family_id=str(uuid.uuid4()),
                    documents=[doc.to_dict() for doc in documents],
                    reasoning=reasoning,
                )
            )
            results.append(data_point)
        return results

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List
import statistics

# Local
from fms_dgt.base.data_objects import DataPoint
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    FlowControllerStep,
    ToolCallStep,
    UserStep,
)
from fms_dgt.core.databuilders.conversation.task import ConversationTask
from fms_dgt.public.databuilders.rag.data_objects import RAGScenarioStep


class RAGConversationTask(ConversationTask):
    """ConversationTask subclass that reports RAG-specific metrics.

    Identical to ConversationTask in all respects except for
    record_task_results(), which computes corpus coverage, turn statistics,
    word-count statistics, flow pattern distributions, conversation completion
    rate, and retrieval tool call rate.
    """

    def record_task_results(self, intermediate_data: List[DataPoint]) -> Dict[str, Any]:
        conversations = intermediate_data
        if not conversations:
            return {}

        # ----------------------------------------------------------------
        # Per-conversation accumulators
        # ----------------------------------------------------------------
        turn_counts: List[int] = []
        user_word_counts: List[int] = []
        assistant_word_counts: List[int] = []
        per_conv_pattern_fractions: Dict[str, List[float]] = {}
        global_pattern_counts: Dict[str, int] = {}
        completed_count: int = 0
        p2_tool_call_counts: List[int] = []
        p2_assistant_counts: List[int] = []
        seen_doc_ids: set = set()

        for conv in conversations:
            steps = getattr(conv, "steps", [])

            # Turns = number of user steps (one per iteration cycle)
            user_steps = [step for step in steps if isinstance(step, UserStep)]
            assistant_steps = [step for step in steps if isinstance(step, AssistantStep)]
            fc_steps = [step for step in steps if isinstance(step, FlowControllerStep)]
            tool_call_steps = [step for step in steps if isinstance(step, ToolCallStep)]
            scenario_steps = [step for step in steps if isinstance(step, RAGScenarioStep)]

            n_turns = len(user_steps)
            turn_counts.append(n_turns)

            # Word counts per turn
            for step in user_steps:
                text = step.content if isinstance(step.content, str) else ""
                user_word_counts.append(len(text.split()))
            for step in assistant_steps:
                text = step.content if isinstance(step.content, str) else ""
                assistant_word_counts.append(len(text.split()))

            # Flow patterns
            conv_patterns: Dict[str, int] = {}
            for step in fc_steps:
                pattern = step.content if isinstance(step.content, str) else ""
                if pattern:
                    conv_patterns[pattern] = conv_patterns.get(pattern, 0) + 1
                    global_pattern_counts[pattern] = global_pattern_counts.get(pattern, 0) + 1

            total_fc = sum(conv_patterns.values())
            if total_fc > 0:
                for pattern, count in conv_patterns.items():
                    if pattern not in per_conv_pattern_fractions:
                        per_conv_pattern_fractions[pattern] = []
                    per_conv_pattern_fractions[pattern].append(count / total_fc)

            # Completion: last FC step has terminate=True
            if fc_steps and fc_steps[-1].terminate:
                completed_count += 1

            # Retrieval tool call rate (Pattern 2 only — no static documents)
            is_pattern2 = scenario_steps and not scenario_steps[-1].documents
            if is_pattern2:
                p2_tool_call_counts.append(len(tool_call_steps))
                p2_assistant_counts.append(len(assistant_steps))

            # Document coverage (Pattern 1)
            for step in scenario_steps:
                for doc in step.documents or []:
                    doc_id = doc.get("id")
                    if doc_id is not None:
                        seen_doc_ids.add(doc_id)

        # ----------------------------------------------------------------
        # Corpus size from bounded search engines
        # ----------------------------------------------------------------
        total_corpus_size = sum(
            e.corpus_size()
            for e in self.component_tool_engines.values()
            if hasattr(e, "corpus_size")
        )

        # ----------------------------------------------------------------
        # Aggregate
        # ----------------------------------------------------------------
        n = len(conversations)

        def _mean(vals: List[float]) -> float:
            return round(statistics.mean(vals), 2) if vals else 0.0

        def _std(vals: List[float]) -> float:
            return round(statistics.stdev(vals), 2) if len(vals) >= 2 else 0.0

        metrics: Dict[str, Any] = {}

        # Turn length
        metrics["avg_turns"] = _mean(turn_counts)
        metrics["std_turns"] = _std(turn_counts)
        metrics["min_turns"] = min(turn_counts) if turn_counts else 0
        metrics["max_turns"] = max(turn_counts) if turn_counts else 0

        # Word counts
        metrics["avg_user_turn_words"] = _mean(user_word_counts)
        metrics["std_user_turn_words"] = _std(user_word_counts)
        metrics["avg_assistant_turn_words"] = _mean(assistant_word_counts)
        metrics["std_assistant_turn_words"] = _std(assistant_word_counts)

        # Corpus coverage
        metrics["corpus_size"] = total_corpus_size
        metrics["unique_docs_used"] = len(seen_doc_ids)
        metrics["corpus_coverage_pct"] = (
            round(len(seen_doc_ids) / total_corpus_size * 100, 2) if total_corpus_size > 0 else 0.0
        )

        # Flow pattern counts and distributions
        metrics["flow_pattern_counts"] = dict(global_pattern_counts)
        avg_dist = {
            p: round(statistics.mean(fracs), 4) for p, fracs in per_conv_pattern_fractions.items()
        }
        std_dist = {
            p: round(statistics.stdev(fracs), 4) if len(fracs) >= 2 else 0.0
            for p, fracs in per_conv_pattern_fractions.items()
        }
        metrics["flow_pattern_distribution"] = avg_dist
        metrics["flow_pattern_distribution_std"] = std_dist

        # Completion rate
        metrics["completion_rate"] = round(completed_count / n, 4) if n > 0 else 0.0

        # Retrieval tool call rate (Pattern 2 only)
        total_p2_calls = sum(p2_tool_call_counts)
        total_p2_assistant = sum(p2_assistant_counts)
        metrics["retrieval_tool_call_rate"] = (
            round(total_p2_calls / total_p2_assistant, 4) if total_p2_assistant > 0 else None
        )

        return metrics

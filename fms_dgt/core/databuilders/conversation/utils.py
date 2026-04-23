# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Tuple, Type, TypeVar, Union, overload
import json
import math
import random
import re

# Local
from fms_dgt.core.databuilders.conversation.data_objects import (
    AssistantStep,
    Step,
    ToolCallStep,
    ToolResultStep,
    UserStep,
)

T = TypeVar("T", bound=Step)


def steps_to_messages(steps: List[Step]) -> List[Dict[str, Any]]:
    """Convert a list of Steps to an OpenAI-compatible messages list.

    Handles user, assistant, tool_call, and tool_result roles. Pipeline-internal
    steps (scenario, persona, flow_controller) are skipped.

    Tool call / tool result pairs are serialized as:
      - assistant message with ``tool_calls`` list (from tool_call step)
      - tool message with ``tool_call_id`` and ``content`` (from tool_result step)

    This format is compatible with the OpenAI chat completions API and can be
    passed directly as the ``messages`` argument to any LM provider.
    """
    messages: List[Dict[str, Any]] = []
    for step in steps:
        if isinstance(step, UserStep):
            messages.append({"role": "user", "content": step.content})
        elif isinstance(step, AssistantStep):
            messages.append({"role": "assistant", "content": step.content})
        elif isinstance(step, ToolCallStep):
            content = step.content
            if isinstance(content, dict):
                call_id = content.get("call_id", "")
                name = content.get("name", "")
                arguments = content.get("arguments", {})
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": (
                                        json.dumps(arguments)
                                        if not isinstance(arguments, str)
                                        else arguments
                                    ),
                                },
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": str(content)})
        elif isinstance(step, ToolResultStep):
            content = step.content
            if isinstance(content, dict):
                call_id = content.get("call_id", "")
                result = content.get("result", "")
                error = content.get("error", None)
                tool_content = (
                    error
                    if error
                    else (json.dumps(result) if not isinstance(result, str) else result)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_content,
                    }
                )
            else:
                messages.append({"role": "tool", "tool_call_id": "", "content": str(content)})
    return messages


def steps_to_text(steps: List[Step]) -> str:
    """Render conversation steps as a human-readable string for use in prompts.

    Produces the same turn coverage as ``steps_to_messages`` but as plain text
    suitable for embedding in a system or user prompt.
    """
    lines = []
    for msg in steps_to_messages(steps):
        role = msg["role"]
        if role == "user":
            lines.append(f"User: {msg['content']}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(f"Assistant (tool call): {fn.get('name')}({fn.get('arguments')})")
            else:
                lines.append(f"Assistant: {msg['content']}")
        elif role == "tool":
            lines.append(f"Tool result [{msg.get('tool_call_id', '')}]: {msg['content']}")
    return "\n".join(lines)


@overload
def get_first_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[False] = ...,
) -> T | None: ...


@overload
def get_first_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[True],
) -> Tuple[int, T] | Tuple[None, None]: ...


def get_first_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: bool = False,
) -> Union[T, Tuple[int, T], None, Tuple[None, None]]:
    """Return the first step that is an instance of ``tgt_class``.

    Args:
        steps: List of steps to search.
        tgt_class: Step subclass to match.
        return_index: If True, return ``(index, step)`` instead of just the step.

    Returns:
        The first matching step, or None if not found.
        If ``return_index`` is True, returns ``(index, step)`` or ``(None, None)``.
    """
    for idx, step in enumerate(steps):
        if isinstance(step, tgt_class):
            return (idx, step) if return_index else step
    return (None, None) if return_index else None


@overload
def get_last_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[False] = ...,
) -> T | None: ...


@overload
def get_last_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[True],
) -> Tuple[int, T] | Tuple[None, None]: ...


def get_last_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: bool = False,
) -> Union[T, Tuple[int, T], None, Tuple[None, None]]:
    """Return the last step that is an instance of ``tgt_class``.

    Args:
        steps: List of steps to search.
        tgt_class: Step subclass to match.
        return_index: If True, return ``(index, step)`` instead of just the step.

    Returns:
        The last matching step, or None if not found.
        If ``return_index`` is True, returns ``(index, step)`` or ``(None, None)``.
    """
    for idx, step in reversed(list(enumerate(steps))):
        if isinstance(step, tgt_class):
            return (idx, step) if return_index else step
    return (None, None) if return_index else None


@overload
def get_random_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[False] = ...,
) -> T | None: ...


@overload
def get_random_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: Literal[True],
) -> Tuple[int, T] | Tuple[None, None]: ...


def get_random_step_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: bool = False,
) -> Union[T, Tuple[int, T], None, Tuple[None, None]]:
    """Return a randomly selected step that is an instance of ``tgt_class``.

    Args:
        steps: List of steps to search.
        tgt_class: Step subclass to match.
        return_index: If True, return ``(index, step)`` instead of just the step.

    Returns:
        A randomly chosen matching step, or None if not found.
        If ``return_index`` is True, returns ``(index, step)`` or ``(None, None)``.
    """
    matches = [(idx, step) for idx, step in enumerate(steps) if isinstance(step, tgt_class)]
    if not matches:
        return (None, None) if return_index else None
    idx, step = random.choice(matches)
    return (idx, step) if return_index else step


def get_all_steps_of_type(
    steps: List[Step],
    tgt_class: Type[T],
    return_index: bool = False,
) -> List[Union[T, Tuple[int, T]]]:
    """Return all steps that are instances of ``tgt_class``.

    Args:
        steps: List of steps to search.
        tgt_class: Step subclass to match.
        return_index: If True, each entry is ``(index, step)`` instead of just the step.

    Returns:
        List of matching steps (or ``(index, step)`` pairs if ``return_index`` is True).
        Empty list if none found.
    """
    if return_index:
        return [(idx, step) for idx, step in enumerate(steps) if isinstance(step, tgt_class)]
    return [step for step in steps if isinstance(step, tgt_class)]


# ---------------------------------------------------------------------------
# TF-IDF similarity ranking
# ---------------------------------------------------------------------------

_TFIDF_STOPWORDS: frozenset = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "if",
        "not",
        "no",
        "so",
        "up",
        "out",
        "about",
        "into",
        "than",
        "then",
        "when",
        "which",
        "who",
        "what",
        "your",
        "you",
        "we",
        "our",
        "they",
        "their",
        "he",
        "she",
        "his",
        "her",
        "all",
        "also",
        "more",
        "can",
        "any",
        "each",
        "after",
        "before",
        "between",
        "through",
        "how",
    }
)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in tokens if t not in _TFIDF_STOPWORDS]


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_tfidf(
    reference: str,
    candidates: List[str],
) -> List[int]:
    """Rank candidates by TF-IDF cosine similarity to a reference string.

    IDF is computed over the full candidate pool so that terms common across all
    candidates are down-weighted relative to discriminative terms.

    Args:
        reference: Query text.
        candidates: List of candidate texts to rank.

    Returns:
        Indices into ``candidates`` sorted by descending similarity to ``reference``.
        Falls back to original order when all scores are zero (e.g., empty inputs).
    """
    if not candidates:
        return []

    candidate_tokens = [_tokenize(text) for text in candidates]
    ref_tokens = _tokenize(reference)

    # IDF over candidate pool only (reference is the query, not a document)
    df: Dict[str, int] = defaultdict(int)
    for tokens in candidate_tokens:
        for t in set(tokens):
            df[t] += 1
    n = len(candidate_tokens)
    idf = {t: math.log((n + 1) / (freq + 1)) + 1 for t, freq in df.items()}

    ref_vec = _tfidf_vector(ref_tokens, idf)
    scores = [_cosine(ref_vec, _tfidf_vector(tokens, idf)) for tokens in candidate_tokens]

    return sorted(range(len(candidates)), key=lambda i: -scores[i])

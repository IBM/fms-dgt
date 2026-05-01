# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, List, Set
import copy
import json
import math
import random

# Third Party
from scipy import stats

# Local
from fms_dgt.core.blocks.validators.tool_calling import ToolCallValidator
from fms_dgt.core.databuilders.conversation.constants import (
    ASSISTANT_ROLE,
    CONTENT_KEY,
    ROLE_KEY,
    USER_ROLE,
)
from fms_dgt.core.tools.constants import (
    TOOL_CALL_ARGS,
    TOOL_CALL_ID,
    TOOL_NAME,
    TOOL_PROPERTIES,
)
from fms_dgt.core.tools.data_objects import Tool
from fms_dgt.utils import dgt_logger

###
#
###


def validate_tool_calls(
    tool_calls: List[Dict],
    tools: List[Tool],
    conversation_history: str | None = None,
    require_nested: bool = False,
    allow_nested: bool = False,
    check_arg_question_overlap: bool = True,
    log_failures: bool = False,
) -> bool:
    """Validate tool calls against schema and conversation context.

    Args:
        tool_calls: Tool call dict or list of tool call dicts to validate.
        conversation_history: Conversation history for context validation.
        tools: List of available tool schemas.
        require_nested: Whether nested tool calls are required.
        allow_nested: Whether nested tool calls are allowed.
        check_arg_question_overlap: Whether to check if arguments appear in the conversation.
        log_failures: Whether to log validation failures.

    Returns:
        List of validated ToolCall objects (empty if validation fails).
    """
    # If empty tool calls, then return empty list
    if not tool_calls:
        return []

    # If not a list or dictionary, then return empty list
    if not (isinstance(tool_calls, list) or isinstance(tool_calls, dict)):
        return []

    # Cast to list, if necessary
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]

    # Initalize necessary variables
    is_valid = False

    # Initialize tool calling validator
    tool_call_validator = ToolCallValidator(name="tool_call_validator")

    # Validate
    if all(
        [isinstance(tc, dict) and tc.get(TOOL_NAME) and tc.get(TOOL_CALL_ARGS) for tc in tool_calls]
    ) and all(
        [
            (TOOL_CALL_ID not in tc)
            or (isinstance(tc.get(TOOL_CALL_ID), str) and tc.get(TOOL_CALL_ID, "").startswith("$"))
            for tc in tool_calls
        ]
    ):
        val_inp = [
            {
                "tools": [t.to_dict() for t in tools],
                "answer": json.dumps(
                    [
                        {
                            k: v
                            for k, v in tc.items()
                            if k in [TOOL_NAME, TOOL_CALL_ARGS, TOOL_CALL_ID]
                        }
                        for tc in tool_calls
                    ]
                ),
                "question": conversation_history,
                "check_arg_question_overlap": check_arg_question_overlap,
                "allow_subset": True,
                "allow_nested": allow_nested,
                "require_nested": require_nested,
                "validate_question": conversation_history is not None,
                "ignore_types": ["boolean", "date"],
            }
        ]
        result = tool_call_validator(val_inp)[0]
        is_valid, reason = result["is_valid"], (result["metadata"] or dict()).get("reason")

        if log_failures and not is_valid:
            dgt_logger.debug(reason)

    # Return
    return is_valid


def syntax_check(conversation: List[Dict]) -> bool:
    """
    Perform minimal syntax check on the conversation

    Checks:
    - Must be non-empty list of dictionaries
    - Each dictionary must contain 'role' and 'content' key
    - `role` field must have either "user" or "assistant" value
    - `content` field must be a string or dictionary
    - conversation must start with `role` == "user"
    - `content` field for all steps with `role` == "user" must be a dictionary
    - All steps with `role` != "assistant" must not have `name`, `arguments` and `content` field.

    Args:
        conversation (List[Dict]): conversation to check

    Returns:
        bool: True, if all check pass.
    """
    if (
        conversation
        and isinstance(conversation, list)
        and all(
            [
                isinstance(step, dict)
                and step.get(ROLE_KEY) in [USER_ROLE, ASSISTANT_ROLE]
                and isinstance(step.get(CONTENT_KEY), (str, dict))
                for step in conversation
            ]
        )
        and not any(
            [
                step.get(ROLE_KEY) == ASSISTANT_ROLE and not isinstance(step.get(CONTENT_KEY), dict)
                for step in conversation
            ]
        )
        and all(
            [
                step.get(ROLE_KEY) != USER_ROLE
                or (step.get(CONTENT_KEY) and isinstance(step.get(CONTENT_KEY), str))
                for step in conversation
            ]
        )
        and all(
            [
                step.get(ROLE_KEY) != ASSISTANT_ROLE
                or (
                    not set(step.get(CONTENT_KEY, dict()).keys()).symmetric_difference(
                        [TOOL_NAME, TOOL_CALL_ARGS, TOOL_CALL_ID]
                    )
                    and isinstance(step[CONTENT_KEY][TOOL_CALL_ID], str)
                    and isinstance(step[CONTENT_KEY][TOOL_CALL_ARGS], dict)
                )
                for step in conversation
            ]
        )
    ):
        return True
    return False


def extract_largest_connected_component(
    tool_calls: List[Dict],
    context: List[Dict] | None = None,
    min_component_size: int = 2,
) -> List[Dict]:
    """Extract the largest connected component of tool calls based on dependencies.

    Analyzes tool call dependencies (nested references) and extracts the largest
    connected group of tool calls that reference each other.

    Args:
        tool_calls: List of tool call dicts to analyze.
        context: Optional list of context tool calls (from previous steps).
        min_component_size: Minimum size for a valid component (excluding context).

    Returns:
        List of tool calls forming the largest connected component.
    """

    def _connected_components(graph: Dict[str, Set]):

        def _component(node):
            component = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                component.append(node)
                seen.add(node)
                nodes.update(graph[node].difference(seen))
            return component

        seen = set()
        components = []
        for node in graph:
            if node not in seen:
                components.append(_component(node))

        return components

    def _has_ref(tgt: str, d: Any):
        if not isinstance(tgt, str):
            return False
        elif isinstance(d, dict):
            return any([_has_ref(tgt, v) for v in d.values()])
        elif isinstance(d, list):
            return any([_has_ref(tgt, v) for v in d])
        elif isinstance(d, str):
            return d.startswith(str(tgt))
        else:
            return tgt == d

    if context is None:
        context = []
    context_ids = [tc.get(TOOL_CALL_ID) for tc in context]
    tool_calls = context + tool_calls

    # graph with keys = parents + values = children
    parent_to_args = {tc.get(TOOL_CALL_ID): set() for tc in tool_calls}
    arg_to_parents = {tc.get(TOOL_CALL_ID): set() for tc in tool_calls}
    for inp_tool_call in tool_calls:
        if isinstance(inp_tool_call.get(TOOL_CALL_ARGS), dict):
            for out_tool_call in tool_calls:
                if inp_tool_call == out_tool_call:
                    continue
                elif _has_ref(
                    out_tool_call.get(TOOL_CALL_ID) + ".", inp_tool_call.get(TOOL_CALL_ARGS)
                ):
                    parent_to_args[inp_tool_call.get(TOOL_CALL_ID)].add(
                        out_tool_call.get(TOOL_CALL_ID)
                    )
                    arg_to_parents[out_tool_call.get(TOOL_CALL_ID)].add(
                        inp_tool_call.get(TOOL_CALL_ID)
                    )

    graph = {tc.get(TOOL_CALL_ID): set() for tc in tool_calls}
    for par, args in parent_to_args.items():
        for arg in args:
            graph[par].add(arg)
            graph[arg].add(par)

    id_map = {tc.get(TOOL_CALL_ID): tc for tc in tool_calls}
    all_ids = [tc.get(TOOL_CALL_ID) for tc in tool_calls]

    subst_groups = []
    connected_components = _connected_components(graph)
    connected_components = [
        list(sorted(conn, key=lambda x: all_ids.index(x))) for conn in connected_components
    ]
    if min_component_size is not None:
        connected_components = [
            conn
            for conn in connected_components
            if len(set(conn).difference(context_ids)) >= min_component_size
        ]
        keep_ids = max(connected_components, key=len) if connected_components else []
    else:
        keep_ids = [tc_id for conn in connected_components for tc_id in conn]

    subst_groups = order_based_subst(
        [id_map[id] for id in keep_ids if id not in context_ids], context_ids
    )

    return subst_groups


def order_based_subst(func_calls: List[Dict], context_ids: List[str] | None = None) -> List[Dict]:

    def _subst_tokens(d: Any, mapping: Dict):
        if isinstance(d, dict):
            return {_subst_tokens(k, mapping): _subst_tokens(v, mapping) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [_subst_tokens(el, mapping) for el in d]
        elif isinstance(d, str):
            for k, v in mapping.items():
                if d.startswith(str(k) + ".") or d == k:
                    return d.replace(k, v)
            return d
        else:
            return d

    def _int_to_id(q_id: int):
        return f"${q_id}"

    def _id_to_int(q_id: str):
        return int(q_id.replace("$", ""))

    start_from = max([_id_to_int(id) for id in (context_ids or [])] + [0]) + 1
    mapping = dict()
    for func_call in func_calls:
        new_id = _int_to_id(len(mapping) + start_from)
        old_id = func_call[TOOL_CALL_ID]
        mapping[old_id] = new_id
    replaced_groups = _subst_tokens(func_calls, mapping)

    return replaced_groups


def get_leaves(plan: List[Dict]):
    """ """
    leaves = dict()
    for tc in plan:
        for arg in tc.get(TOOL_CALL_ARGS, dict()).values():
            if not ((isinstance(arg, str) and arg.startswith("$")) or isinstance(arg, bool)):
                leaves[str(arg)] = arg

    return list(leaves.values())


def shuffle_tools(tools: List[Dict]):

    def shuff(d):
        #
        if isinstance(d, dict) and isinstance(d.get(TOOL_PROPERTIES), dict):
            props = list(d[TOOL_PROPERTIES].items())
            random.shuffle(props)
            d[TOOL_PROPERTIES] = dict(props)
        #
        if isinstance(d, dict):
            return {k: shuff(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [shuff(v) for v in d]
        else:
            return d

    tools = copy.deepcopy(tools)
    shuff(tools)
    random.shuffle(tools)

    return tools


def correlation_check(
    plan: List[Dict],
    user_request: str,
    hidden_set: List | None = None,
    hidden_args: List | None = None,
    corr_thresh: float = 0.0,
    to_ignore: List | None = None,
):
    if to_ignore is None:
        to_ignore = ["true", "false"]

    hidden_args = set(map(normalize_str, hidden_args)) if hidden_args else set()

    args = []
    for tc_i, tc in enumerate(plan):
        for val in tc.get(TOOL_CALL_ARGS, dict()).values():
            if not (isinstance(val, str) and val.startswith("$")):
                if hidden_set or hidden_args:
                    args.append(
                        (
                            (
                                0
                                if (hidden_set and tc in hidden_set)
                                or normalize_str(val) in hidden_args
                                else 1
                            ),
                            val,
                        )
                    )
                else:
                    args.append((tc_i, val))

    for is_first in [True, False]:
        occ_args = [
            (index_of_occurrence(arg, user_request, first=is_first), arg) for _, arg in args
        ]
        matchable_args = [
            arg for ind, arg in occ_args if ind != -1 and str(arg).lower() not in to_ignore
        ]
        # corr, _ = stats.spearmanr(
        corr, _ = stats.kendalltau(
            [x[0] for x in args if x[1] in matchable_args],
            [x[0] for x in occ_args if x[1] in matchable_args],
        )
        if math.isnan(corr) or corr > corr_thresh:
            return False
    return True


def index_of_occurrence(arg_content: Any, question: str, first: bool = True):

    def _find(lst: List, srch: List):
        #
        if len(srch) > len(lst) or not lst or not srch:
            return -1

        match, fallback = -1, -1
        for i in range(len(lst) - len(srch) + 1):
            if lst[i : i + len(srch)] == srch:
                match = match if first and match != -1 else i
            if lst[i].startswith(srch[0]):
                fallback = fallback if first and fallback != -1 else i

        return match if match != -1 else fallback

    def _get_occ(value: Any):
        if isinstance(value, (list, tuple)):
            opts = [_get_occ(x) for x in value]
            return min(opts) if opts else -1
        elif isinstance(value, dict):
            # we'll skip checking keys
            opts = [_get_occ(v) for v in value.values()]
            return min(opts) if opts else -1
        else:
            return _find(question_words, [normalize_str(v) for v in str(value).split()])

    question_words = [normalize_str(wrd) for wrd in question.split()]

    return _get_occ(arg_content)


def normalize_str(value: Any):
    normed = "".join([c for c in str(value) if c.isalnum() or c == " "]).lower()
    if not normed:
        normed = str(value).lower()
    return normed

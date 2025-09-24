# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json

# Third Party
from dateutil import parser as date_parser
from openapi_schema_validator import validate
import jsonschema

# Local
from fms_dgt.base.block import ValidatorBlock
from fms_dgt.base.data_objects import ValidatorBlockData
from fms_dgt.base.registry import register_block
from fms_dgt.constants import TYPE_KEY
from fms_dgt.core.tools import (
    ARGS,
    CALL_ID,
    NAME,
    OUTPUT_PARAMETERS,
    PARAMETERS,
    PROPERTIES,
)
from fms_dgt.utils import dgt_logger, try_parse_json_string


@dataclass(kw_only=True)
class TCValidatorData(ValidatorBlockData):
    tools: List[Dict]
    answer: str
    question: str = None
    check_arg_question_overlap: bool = True
    require_nested: bool = False
    allow_nested: bool = False
    multi_output: bool = False
    validate_question: bool = True
    ignore_types: list = None

    def __post_init__(self):
        if self.question is None:
            self.question = ""
        if self.ignore_types is None:
            self.ignore_types = ["boolean"]


@register_block("tool_call_validator")
class ToolCallValidator(ValidatorBlock):
    """Class for Tool Call Sequence Prediction Validator"""

    DATA_TYPE: TCValidatorData = TCValidatorData

    def _validate(self, inp: TCValidatorData) -> bool | Tuple[bool, Dict | None]:
        # Parse tool calls to validate
        tool_calls_to_validate = try_parse_json_string(inp.answer)

        # Return false, if failed to parse tool calls to validate
        if tool_calls_to_validate is None:
            return (False, {"reason": "Failed to parse tool call[s]"})

        # Malformedness check
        if not isinstance(tool_calls_to_validate, list) or any(
            [not isinstance(tc, dict) for tc in tool_calls_to_validate]
        ):
            dgt_logger.debug(
                'Malformed tool calls "%s" for text "%s"',
                inp.answer,
                inp.question,
            )
            return (
                False,
                {
                    "reason": "Malformed tool call[s]",
                },
            )

        # Extract available tool names, tool call names and tool call IDs
        available_tool_names = set([str(tool.get(NAME)) for tool in inp.tools])
        tc_names = set([str(tc.get(NAME)) for tc in tool_calls_to_validate])
        tc_ids = set([str(tc.get(CALL_ID)) for tc in tool_calls_to_validate if tc.get(CALL_ID)])

        # Exact-match duplicates check
        if (len(set([str(x) for x in tool_calls_to_validate])) != len(tool_calls_to_validate)) or (
            len(set(tc_ids)) != len(tc_ids)
        ):
            dgt_logger.debug(
                'Duplicate tool calls in "%s" for text "%s"',
                inp.answer,
                inp.question,
            )
            return (
                False,
                {
                    "reason": "Duplicate tool call[s]",
                },
            )

        # check that question does not contain available tool names or available tool formats
        for available_tool_name in available_tool_names:
            if inp.validate_question and (
                _normalize_str(available_tool_name)
                in [_normalize_str(wrd) for wrd in inp.question.split()]
                or ("{" in inp.question and 'name": ' in inp.question)
                or ("{" in inp.question and available_tool_name + '": ' in inp.question)
            ):
                dgt_logger.debug(
                    'Text "%s" contains tool name or tool format',
                    inp.answer,
                )
                return (
                    False,
                    {
                        "reason": "Question contains tool call[s]",
                    },
                )

        # Check multiple unique tool calls, if `multi_output` == True
        if inp.multi_output and len(set([str(tc) for tc in tool_calls_to_validate])) <= 1:
            dgt_logger.debug(
                'Expected multiple tool calls for text "%s" but received "%s"',
                inp.answer,
                inp.question,
            )
            return (
                False,
                {
                    "reason": "Expected multiple tool calls",
                },
            )

        # Check only all or subset of available tools are used
        if not tc_names.issubset(available_tool_names):
            dgt_logger.debug(
                'Unrecognized/unavailable tool[s] "%s" for text "%s"',
                inp.answer,
                inp.question,
            )
            return (
                False,
                {
                    "reason": "Unrecognized/unavailable tool[s]",
                },
            )

        # Initialize necessary variables
        allow_nested = inp.require_nested or inp.allow_nested
        has_nested = False

        for idx, tool_call_to_validate in enumerate(tool_calls_to_validate):
            # Malformedness check
            if not set(tool_call_to_validate.keys()).issubset(set([NAME, ARGS, CALL_ID])):
                dgt_logger.debug(
                    'Malformed tool call "%s" with additional keys for text "%s"',
                    json.dumps(tool_call_to_validate),
                    inp.question,
                )
                return (
                    False,
                    {
                        "reason": "Malformed tool call with additional keys",
                        "tool_call": tool_call_to_validate,
                    },
                )

            # Mandatory fields check
            if NAME not in tool_call_to_validate or ARGS not in tool_call_to_validate:
                dgt_logger.debug(
                    'Missing mandatory field[s] ("name", "arguments") from tool call "%s" for text "%s"',
                    json.dumps(tool_call_to_validate),
                    inp.question,
                )
                return (
                    False,
                    {
                        "reason": 'Missing mandatory field[s] ("name", "arguments")',
                        "tool_call": tool_call_to_validate,
                    },
                )

            # Get first matching tool, since we've checked that each tool call to validate has an associated tool
            matching_tool = next(
                tool for tool in inp.tools if tool[NAME] == tool_call_to_validate[NAME]
            )
            matching_tool_params = matching_tool.get(PARAMETERS, dict())
            tc_args = tool_call_to_validate[ARGS]

            # Validate argument schema
            try:
                matching_tool and validate(tc_args, matching_tool_params)
            except (
                jsonschema.exceptions.ValidationError,
                jsonschema.exceptions.SchemaError,
                jsonschema.exceptions._WrappedReferencingError,
            ) as err:
                # if error is about a var label, e.g., $var1, then ignore error. Otherwise, raise error
                if not (
                    allow_nested and any([str(err).startswith(f"'{tc_id}") for tc_id in tc_ids])
                ):
                    dgt_logger.debug(
                        '"%s" tool call for text "%s" has following parameter validation error:\n%s',
                        json.dumps(tool_call_to_validate),
                        inp.question,
                        str(err),
                    )
                    return (
                        False,
                        {
                            "reason": "Parameter validation error",
                            "tool_call": tool_call_to_validate,
                        },
                    )

            # Check for argument's type
            if not isinstance(tc_args, dict):
                dgt_logger.debug(
                    '"%s" tool call for text "%s" has malformed arguments',
                    json.dumps(tool_call_to_validate),
                    inp.question,
                )
                return (
                    False,
                    {
                        "reason": "Malformed arguments",
                        "tool_call": tool_call_to_validate,
                    },
                )

            # Individual argument checking
            matching_tool_args = matching_tool_params.get(PROPERTIES, dict())
            for arg_name, arg_value in tc_args.items():

                # Check for argument name hallucinations
                if arg_name not in matching_tool_args:
                    dgt_logger.debug(
                        'Hallucinated argument name "%s" for tool call "%s" for text "%s"',
                        arg_name,
                        json.dumps(tool_call_to_validate),
                        inp.question,
                    )
                    return (
                        False,
                        {
                            "reason": "Hallucinated argument name",
                            "tool_call": tool_call_to_validate,
                            "argument_name": arg_name,
                        },
                    )

                # Check where nested calls reference the future
                is_future_nested = _is_nested_match(
                    arg_value, tool_calls_to_validate[idx:], inp.tools
                )
                if is_future_nested:
                    dgt_logger.debug(
                        'Future variable reference in "%s" for text "%s"',
                        inp.answer,
                        inp.question,
                    )
                    return (
                        False,
                        {
                            "reason": "Nested future tool call reference failure",
                        },
                    )

                # check for nested
                is_nested_call = _is_nested_match(
                    arg_value, tool_calls=tool_calls_to_validate[:idx], tools=inp.tools
                )
                has_nested = is_nested_call or has_nested

                # handle the case where slot values are not mentioned in the question
                if (
                    inp.check_arg_question_overlap
                    and not is_nested_call
                    and (matching_tool_args[arg_name].get(TYPE_KEY) not in inp.ignore_types)
                    and not ("date" in inp.ignore_types and _is_date_time(arg_value))
                    and not is_valid_tc_arg_overlap(arg_value, inp.question)
                ):
                    dgt_logger.debug(
                        'Hallucinated argument value "%s" for tool call "%s" for text "%s"',
                        arg_value,
                        json.dumps(tool_call_to_validate),
                        inp.question,
                    )
                    return (
                        False,
                        {
                            "reason": "Hallucinated argument value",
                            "tool_call": tool_call_to_validate,
                            "argument_value": arg_value,
                        },
                    )

                # check for internal var to string
                indirect_nested_call = (not is_nested_call) and _indirect_nested_call(
                    arg_value, tool_calls=tool_calls_to_validate, tools=inp.tools
                )
                if indirect_nested_call:
                    dgt_logger.debug(
                        'Indirect variable reference "%s" in "%s" for text "%s"',
                        arg_value,
                        inp.answer,
                        inp.question,
                    )
                    return (
                        False,
                        {"reason": f"Indirect variable reference {arg_value}"},
                    )

        # if has_nested, then check_nested must be true. Otherwise, require_nested must be false
        is_valid = (not has_nested and not inp.require_nested) or (has_nested and allow_nested)

        if not is_valid:
            dgt_logger.debug(
                'Nested tool call validation failed for "%s" for text "%s"',
                inp.answer,
                inp.question,
            )

        return is_valid, (
            {
                "reason": "Nested tool call validation failed",
            }
            if not is_valid
            else None
        )


def _is_date_time(string: str) -> bool:
    if isinstance(string, str) and len(string) < 50:
        try:
            date_parser.parse(string)
            return True
        except (date_parser.ParserError, OverflowError):
            return False
    return False


def _normalize_str(value: Any):
    normed = "".join([c for c in str(value) if c.isalnum() or c == " "]).lower()
    if not normed:
        normed = str(value).lower()
    return normed


def is_valid_tc_arg_overlap(arg_content: str, question: str):
    def _check_match(value: Any):
        if isinstance(value, (list, tuple)):
            return all([_check_match(x) for x in value])
        elif isinstance(value, dict):
            # we'll skip checking keys
            return all([_check_match(v) for v in value.values()])
        elif str(value) in question:
            # case where normalized string would become empty
            return True
        # SPECIAL CASE: Handles when value is " "
        elif isinstance(value, str) and value.strip():
            return _normalize_str(value) in normalized_question
        else:
            if value in number_map and str(number_map[value]) in normalized_question:
                return True
            return _normalize_str(value) in normalized_question

    number_map = dict(
        enumerate(
            [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ]
        )
    )

    normalized_question = _normalize_str(question)

    return str(arg_content) and _check_match(arg_content)


def _is_nested_match(
    arg_value: str,
    tool_calls: List[Dict],
    tools: List[Dict],
):

    def _nested_check(inp, id_to_check, out_param_name):
        if isinstance(inp, dict):
            return any(
                [
                    _nested_check(k, id_to_check, out_param_name)
                    or _nested_check(v, id_to_check, out_param_name)
                    for k, v in inp.items()
                ]
            )
        elif isinstance(inp, (list, tuple)):
            return any([_nested_check(v, id_to_check, out_param_name) for v in inp])
        elif isinstance(inp, str):
            return (
                inp.startswith(id_to_check)
                and " " not in inp
                # hmm, is this general purpose
                and inp.startswith(id_to_check + "." + out_param_name)
            )

    for tool_call in tool_calls:
        # Proceed only if "id" field is available in tool call
        if CALL_ID not in tool_call:
            continue

        # Find matching tool from list of tools
        matching_tool = next(tool for tool in tools if tool[NAME] == tool_call[NAME])

        for out_param_name in (
            matching_tool.get(OUTPUT_PARAMETERS, dict()).get(PROPERTIES, dict()).keys()
        ):
            if _nested_check(arg_value, str(tool_call[CALL_ID]), out_param_name):
                return True

    return False


def _indirect_nested_call(
    arg_value: str,
    tool_calls: List[Dict],
    tools: List[Dict],
):
    def _check(val):
        for tool_call in tool_calls:
            # Proceed only if "id" field is available in tool call
            if CALL_ID not in tool_call:
                continue

            # Find matching tool from list of tools
            matching_tool = next(tool for tool in tools if tool[NAME] == tool_call[NAME])

            if (
                (str(tool_call[CALL_ID]) + ".") in val
                or any(
                    [
                        ("." + out_param_name) in val
                        for out_param_name in (
                            matching_tool.get(OUTPUT_PARAMETERS, dict())
                            .get(PROPERTIES, dict())
                            .keys()
                        )
                    ]
                )
            ) and not any(
                [
                    val.startswith(str(tool_call[CALL_ID]) + "." + out_param_name)
                    and " " not in val
                    for out_param_name in (
                        matching_tool.get(OUTPUT_PARAMETERS, dict()).get(PROPERTIES, dict()).keys()
                    )
                ]
            ):
                return True

    def _nested_check(inp):
        if isinstance(inp, dict):
            return any([_nested_check(k) or _nested_check(v) for k, v in inp.items()])
        elif isinstance(inp, (list, tuple)):
            return any([_nested_check(v) for v in inp])
        elif isinstance(inp, str):
            return _check(inp)

    if _nested_check(arg_value):
        return True

    return False

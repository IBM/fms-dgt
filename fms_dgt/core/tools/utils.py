# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import re


def extract_first_tool_call(string_w_tool_calls: str) -> str | None:
    """ """
    try:
        bt, first_point, l_tok, r_tok = 0, None, None, None
        for i in range(len(string_w_tool_calls)):
            if string_w_tool_calls[i] == "[":
                continuation = re.sub(r"\s", "", string_w_tool_calls[i + 1 :])
                if continuation and any(continuation.startswith(lead) for lead in ["{"]):
                    first_point, l_tok, r_tok = i, "[", "]"
                    break
            elif string_w_tool_calls[i] == "{":
                continuation = re.sub(r"\s", "", string_w_tool_calls[i + 1 :])
                if continuation:
                    first_point, l_tok, r_tok = i, "{", "}"
                    break
        if first_point is None:
            return None
    except ValueError:
        return None

    stack = 0
    for i, c in enumerate(string_w_tool_calls[first_point:]):
        if c == l_tok:
            stack += 1
        if c == r_tok:
            stack -= 1
        if stack == 0:
            return string_w_tool_calls[(first_point - bt) : first_point + i + 1]

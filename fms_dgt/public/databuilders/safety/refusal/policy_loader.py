# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Iterable, List
import logging

# Local
from fms_dgt.constants import BASE_LOGGER_NAME
from fms_dgt.utils import read_yaml

_logger = logging.getLogger(BASE_LOGGER_NAME)


def load_policy_metadata(file_path: str, risk_name: str) -> dict:
    """Return a flat dict of policy metadata for a specific risk within a policy file.

    Args:
        file_path: Path to the policy YAML.
        risk_name: The ``risk`` field value to look up (e.g. ``"discrimination_at_work"``).

    Returns:
        Dict with keys: risk, risk_description, risk_group, risk_group_id, risk_id,
        reason_denial, short_reply_type, exception, policy_version,
        reply_cannot_contain, reply_may_contain.
    """
    policy = read_yaml(file_path=file_path)

    risk_group = policy.get("risk_group", "unknown")
    risk_group_id = policy.get("risk_group_id", -1)
    policy_version = policy.get("policy_version", "v0.1")

    risks = policy.get("risks", [])
    match = next((r for r in risks if r.get("risk") == risk_name), {})
    risk_policy = match.get("policy", {})

    return {
        "risk": risk_name,
        "risk_description": match.get("description", ""),
        "risk_group": risk_group,
        "risk_group_id": risk_group_id,
        "risk_id": match.get("risk_id", ""),
        "reason_denial": match.get("reason_denial", ""),
        "short_reply_type": match.get("short_reply_type", ""),
        "exception": match.get("exception", ""),
        "policy_version": policy_version,
        "reply_cannot_contain": risk_policy.get("reply_cannot_contain", []),
        "reply_may_contain": risk_policy.get("reply_may_contain", []),
    }


def load_risk_name(file_path: str) -> str:
    """Return the ``risk`` field of the first risk entry in the policy file.

    Used as a fallback when no risk name is specified in the task config.
    """
    policy = read_yaml(file_path=file_path)
    try:
        return policy["risks"][0]["risk"]
    except (KeyError, IndexError, TypeError):
        return policy.get("risk_group", "unknown_risk")


def _traverse(node: dict, path: List[str]) -> Iterable:
    try:
        key, *rest = path
    except ValueError:
        yield node
        return

    if key == "*":
        if not isinstance(node, list):
            return
        for item in node:
            yield from _traverse(item, rest)
    else:
        if key not in node:
            return
        yield from _traverse(node[key], rest)

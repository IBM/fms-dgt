# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Local
from fms_dgt.utils import from_dict, group_by, sanitize_path, to_dict


def test_group_by_basic():
    data = [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "a", "v": 3}]
    result = group_by(data, key=lambda x: x["k"])
    assert list(result.keys()) == ["a", "b"]
    assert result["a"] == [{"k": "a", "v": 1}, {"k": "a", "v": 3}]
    assert result["b"] == [{"k": "b", "v": 2}]


def test_group_by_empty():
    assert group_by([], key=lambda x: x) == {}


def test_group_by_single_group():
    data = [1, 2, 3]
    result = group_by(data, key=lambda x: "same")
    assert list(result.keys()) == ["same"]
    assert result["same"] == [1, 2, 3]


def test_group_by_all_distinct():
    data = ["a", "b", "c"]
    result = group_by(data, key=lambda x: x)
    assert result == {"a": ["a"], "b": ["b"], "c": ["c"]}


def test_group_by_preserves_insertion_order():
    data = [{"k": "c"}, {"k": "a"}, {"k": "b"}, {"k": "c"}]
    result = group_by(data, key=lambda x: x["k"])
    assert list(result.keys()) == ["c", "a", "b"]


def test_group_by_non_string_key():
    data = [1, 2, 3, 4, 5, 6]
    result = group_by(data, key=lambda x: x % 2)
    assert result[1] == [1, 3, 5]
    assert result[0] == [2, 4, 6]


def test_sanitize_path():
    assert sanitize_path("../test") == "test"
    assert sanitize_path("../../test") == "test"
    assert sanitize_path("../../abc/../test") == "test"
    assert sanitize_path("../../abc/../test/fixtures") == "test/fixtures"
    assert sanitize_path("../../abc/../.test/fixtures") == ".test/fixtures"
    assert sanitize_path("/test/foo") == "test/foo"
    assert sanitize_path("./test/bar") == "test/bar"
    assert sanitize_path(".test/baz") == ".test/baz"
    assert sanitize_path("qux") == "qux"


def test_from_dict():
    reference = {
        "key_1": {
            "key_2": {
                "key_3": [
                    {"list_key_1": "list_value_1"},
                    {
                        "list_key_2": {
                            "list_key_2_value_1": {"list_key_2_value_1_list_1": [1, 2, 3]}
                        }
                    },
                    {"list_key_3": "list_value_3"},
                ],
                "key_4": "key_4_value",
            }
        }
    }

    assert from_dict(dictionary=reference, key="key_1.key_2.key_4"), "key_4_value"
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3"), [
        {"list_key_1": "list_value_1"},
        {"list_key_2": {"list_key_2_value_1": {"list_key_2_value_1_list_1": [1, 2, 3]}}},
        {"list_key_3": "list_value_3"},
    ]
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3[1:]"), [
        {"list_key_2": {"list_key_2_value_1": {"list_key_2_value_1_list_1": [1, 2, 3]}}},
        {"list_key_3": "list_value_3"},
    ]
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3[:1]"), [
        {"list_key_1": "list_value_1"},
    ]
    assert from_dict(
        dictionary=reference, key="key_1.key_2.key_3[1].list_key_2.list_key_2_value_1"
    ), {"list_key_2_value_1_list_1": [1, 2, 3]}


def test_to_dict():
    reference = {}
    to_dict(dictionary=reference, key="key_1.key_2[0]", value="value_1")
    assert {"key_1": {"key_2": ["value_1"]}}, reference

    reference = {}
    to_dict(dictionary=reference, key="key_1.key_2[0].key_3", value="value_1")
    assert {"key_1": {"key_2": [{"key_3": "value_1"}]}}, reference

    reference = {"key_1": {"key_2": [{"key_3": "value_1"}]}}
    to_dict(dictionary=reference, key="key_1.key_2[0].key_3", value="value_N")
    assert {"key_1": {"key_2": [{"key_3": "value_N"}]}}, reference

    reference = {"key_1": {"key_2": [{"key_3": "value_1"}]}}
    to_dict(dictionary=reference, key="key_1.key_2", value="value_N")
    assert {"key_1": {"key_2": "value_N"}}, reference

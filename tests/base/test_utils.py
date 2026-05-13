# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import MISSING

# Third Party
import pytest

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

    assert from_dict(dictionary=reference, key="key_1.key_2.key_4") == "key_4_value"
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3") == [
        {"list_key_1": "list_value_1"},
        {"list_key_2": {"list_key_2_value_1": {"list_key_2_value_1_list_1": [1, 2, 3]}}},
        {"list_key_3": "list_value_3"},
    ]
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3[1:]") == [
        {"list_key_2": {"list_key_2_value_1": {"list_key_2_value_1_list_1": [1, 2, 3]}}},
        {"list_key_3": "list_value_3"},
    ]
    assert from_dict(dictionary=reference, key="key_1.key_2.key_3[:1]") == [
        {"list_key_1": "list_value_1"},
    ]
    assert from_dict(
        dictionary=reference, key="key_1.key_2.key_3[1].list_key_2.list_key_2_value_1"
    ) == {"list_key_2_value_1_list_1": [1, 2, 3]}


def test_from_dict_strict_raises_on_missing_terminal():
    """Default behavior raises KeyError when the terminal segment is absent."""
    reference = {"a": {"b": 1}}
    with pytest.raises(KeyError):
        from_dict(reference, "a.missing")
    with pytest.raises(KeyError):
        from_dict(reference, "missing")


def test_from_dict_non_strict_returns_missing_sentinel():
    """strict=False returns dataclasses.MISSING for absent terminal keys."""
    reference = {"a": {"b": 1}}
    assert from_dict(reference, "a.missing", strict=False) is MISSING
    assert from_dict(reference, "missing", strict=False) is MISSING


def test_from_dict_present_none_value_returns_none_even_when_non_strict():
    """A terminal that resolves to None must be distinguishable from absent."""
    reference = {"a": {"b": None}}
    # Present-and-None: returns None, not MISSING, in both modes.
    assert from_dict(reference, "a.b") is None
    assert from_dict(reference, "a.b", strict=False) is None


def test_from_dict_intermediate_missing_always_raises():
    """Intermediate segment absence raises regardless of strict flag."""
    reference = {"a": {"b": 1}}
    with pytest.raises(KeyError):
        from_dict(reference, "missing.b")
    with pytest.raises(KeyError):
        from_dict(reference, "missing.b", strict=False)


def test_from_dict_intermediate_none_raises_cleanly():
    """Intermediate value of None raises KeyError rather than crashing on NoneType."""
    reference = {"a": None}
    with pytest.raises(KeyError):
        from_dict(reference, "a.b")
    with pytest.raises(KeyError):
        from_dict(reference, "a.b", strict=False)


def test_from_dict_rejects_append_modifier():
    """[+] is a write-only DSL element and must be rejected on reads."""
    with pytest.raises(ValueError):
        from_dict({"a": [1]}, "a[+]")


def test_to_dict():
    reference = {}
    to_dict(dictionary=reference, key="key_1.key_2[0]", value="value_1")
    assert reference == {"key_1": {"key_2": ["value_1"]}}

    reference = {}
    to_dict(dictionary=reference, key="key_1.key_2[0].key_3", value="value_1")
    assert reference == {"key_1": {"key_2": [{"key_3": "value_1"}]}}

    reference = {"key_1": {"key_2": [{"key_3": "value_1"}]}}
    to_dict(dictionary=reference, key="key_1.key_2[0].key_3", value="value_N")
    assert reference == {"key_1": {"key_2": [{"key_3": "value_N"}]}}

    reference = {"key_1": {"key_2": [{"key_3": "value_1"}]}}
    to_dict(dictionary=reference, key="key_1.key_2", value="value_N")
    assert reference == {"key_1": {"key_2": "value_N"}}


def test_to_dict_append_terminal_creates_list():
    """Terminal [+] on a missing key creates an empty list and appends."""
    reference = {}
    to_dict(reference, "items[+]", "first")
    assert reference == {"items": ["first"]}
    to_dict(reference, "items[+]", "second")
    assert reference == {"items": ["first", "second"]}


def test_to_dict_append_terminal_existing_list():
    """Terminal [+] on an existing list appends without overwriting."""
    reference = {"items": ["a", "b"]}
    to_dict(reference, "items[+]", "c")
    assert reference == {"items": ["a", "b", "c"]}


def test_to_dict_append_intermediate_descends_into_new_dict():
    """Intermediate [+] appends a fresh dict and descends into it."""
    reference = {}
    to_dict(reference, "annotations[+].magpie.score", 0.9)
    assert reference == {"annotations": [{"magpie": {"score": 0.9}}]}
    to_dict(reference, "annotations[+].magpie.score", 0.7)
    assert reference == {
        "annotations": [
            {"magpie": {"score": 0.9}},
            {"magpie": {"score": 0.7}},
        ]
    }


def test_to_dict_append_rejects_non_list_target():
    """[+] on a key that already holds a non-list must raise TypeError."""
    reference = {"items": "not a list"}
    with pytest.raises(TypeError):
        to_dict(reference, "items[+]", "x")


def test_to_dict_rejects_slice_notation():
    """Writes never accept slice modifiers."""
    reference = {}
    with pytest.raises(ValueError):
        to_dict(reference, "items[:2]", "x")
    with pytest.raises(ValueError):
        to_dict(reference, "items[1:]", "x")


def test_to_dict_none_as_absent_intermediate():
    """An intermediate value of None is treated as absent (auto-creates)."""
    reference = {"magpie_tags": None}
    to_dict(reference, "magpie_tags.metadata.label", "gold")
    assert reference == {"magpie_tags": {"metadata": {"label": "gold"}}}


def test_to_dict_none_as_absent_for_list_intermediate():
    """Intermediate None with a bracket-next segment creates a fresh list."""
    reference = {"entries": None}
    to_dict(reference, "entries[+].name", "alice")
    assert reference == {"entries": [{"name": "alice"}]}

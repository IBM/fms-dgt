# Local
from fms_dgt.utils import from_dict, sanitize_path, to_dict


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

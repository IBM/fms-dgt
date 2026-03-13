# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any
import asyncio

# Local
from fms_dgt.core.blocks.utilities.field_map import FieldMapBlock


@dataclass
class SampleDataClass:
    field1: Any = None
    field2: Any = None
    field3: Any = None
    field4: Any = None


def test_field_map_single():
    block = FieldMapBlock(
        name="test_field_map",
        field_map={"field1": "field2", "field3": "field4", "field2": "field3"},
    )
    test_data = [SampleDataClass(field1=1, field2=2, field3=3, field4=4)]
    block(test_data)
    assert (
        test_data[0].field1 == 1
        and test_data[0].field2 == 1
        and test_data[0].field3 == 2
        and test_data[0].field4 == 3
    )


def test_field_map_multi():
    block = FieldMapBlock(name="test_field_map", field_map={"field1": "field2"})
    test_data = [SampleDataClass(field1=1, field2=2, field3=3, field4=4)]
    block(test_data)
    block = FieldMapBlock(name="test_field_map", field_map={"field2": "field3"})
    block(test_data)
    assert test_data[0].field1 == 1 and test_data[0].field2 == 1 and test_data[0].field3 == 1


def test_field_map_dict():
    block = FieldMapBlock(
        name="test_field_map",
        field_map={"field1": "field2", "field3": "field4", "field2": "field3"},
    )
    test_data = [{"field1": 1, "field2": 2, "field3": 3, "field4": 4}]
    block(test_data)
    assert (
        test_data[0]["field1"] == 1
        and test_data[0]["field2"] == 1
        and test_data[0]["field3"] == 2
        and test_data[0]["field4"] == 3
    )


# ===========================================================================
#                       acall / aexecute
# ===========================================================================


def test_acall_produces_same_result_as_call():
    """acall should produce the same output as __call__ for a sync block."""
    block = FieldMapBlock(
        name="test_async_field_map",
        field_map={"field1": "field2"},
    )
    test_data = [{"field1": 1, "field2": 2, "field3": 3}]
    expected = block([{"field1": 1, "field2": 2, "field3": 3}])

    # Reset: FieldMapBlock mutates in-place, so use a fresh copy
    test_data = [{"field1": 1, "field2": 2, "field3": 3}]
    result = asyncio.run(block.acall(test_data))

    assert list(result)[0]["field2"] == list(expected)[0]["field2"]


def test_acall_returns_list_when_input_is_list():
    block = FieldMapBlock(
        name="test_async_list",
        field_map={"field1": "field2"},
    )
    test_data = [{"field1": 10, "field2": 20}]
    result = asyncio.run(block.acall(test_data))
    assert isinstance(result, list)


def test_acall_multiple_items():
    block = FieldMapBlock(
        name="test_async_multi",
        field_map={"field1": "field2"},
    )
    test_data = [{"field1": i, "field2": 0, "field3": i * 2} for i in range(5)]
    result = asyncio.run(block.acall(test_data))
    outputs = list(result)
    assert len(outputs) == 5
    for i, out in enumerate(outputs):
        assert out["field2"] == i  # field2 now holds original field1 value


def test_aexecute_is_coroutine():
    # Standard
    import inspect

    block = FieldMapBlock(name="test_coro_check", field_map={"field1": "field2"})
    # aexecute returns a coroutine when called
    coro = block.aexecute([{"field1": 1, "field2": 2}])
    assert inspect.iscoroutine(coro)
    coro.close()  # clean up without running

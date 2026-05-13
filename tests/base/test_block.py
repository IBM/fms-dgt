# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`Block`'s ``transform_input`` / ``transform_output``.

Groups cases by route (dict ``src_data`` vs. dataclass ``src_data``) and by
direction (input vs. output). Each section exercises the nested DSL on both
sides of its ``*_map`` so the helpers and the branch logic are covered
together.
"""

# Standard
from dataclasses import MISSING, dataclass
from typing import Any, Dict, List, Optional
import dataclasses
import logging

# Third Party
import pytest

# Local
from fms_dgt.base.block import Block, ValidatorBlock
from fms_dgt.base.data_objects import BlockData, DataPoint, ValidatorBlockData
from fms_dgt.base.telemetry import _NoOpSpanWriter
from fms_dgt.utils import from_dataclass, to_dataclass


# ===========================================================================
#                       TEST FIXTURES
# ===========================================================================
class _NullBlock(Block):
    """A DATA_TYPE=None block used purely to exercise transform_output.

    Overriding ``__init__`` lets the tests construct it without going through
    the full registry / datastore machinery.
    """

    DATA_TYPE = None

    def __init__(self, output_map=None):
        self._name = "_null"
        self._block_type = "test"
        self._input_map = None
        self._output_map = output_map
        self._req_args, self._opt_args = [], []


@dataclass
class _SamplePoint(DataPoint):
    """A realistic user ``DataPoint`` for the dataclass-branch tests.

    Inherits ``task_name`` / ``is_seed`` from :class:`DataPoint` and declares
    the fields touched by the block's default ``output_map`` echo (``score``,
    ``labels``, ``tag``) plus extra fields that the tests aim nested writes at
    (``tags``, ``annotations``, ``metadata``). ``store_names`` is intentionally
    omitted: it is framework-reserved bookkeeping and must never appear in any
    default output_map.
    """

    question: str = ""
    score: Optional[float] = None
    labels: Optional[Dict[str, Any]] = None
    tag: Optional[str] = None
    tags: Optional[List[str]] = None
    annotations: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class _SampleBlockData(BlockData):
    """A ``BlockData`` subclass that mirrors the magpie-style shape."""

    score: float = 0.0
    labels: Optional[Dict[str, Any]] = None
    tag: Optional[str] = None


class _DataclassBlock(Block):
    """A block with a dataclass DATA_TYPE, for exercising the dataclass
    branch of ``transform_output`` without standing up a full databuilder.
    """

    DATA_TYPE = _SampleBlockData

    def __init__(self, output_map=None):
        self._name = "_dc"
        self._block_type = "test"
        self._input_map = None
        self._output_map = output_map
        self._req_args = [
            f.name
            for f in dataclasses.fields(_SampleBlockData)
            if f.default is dataclasses.MISSING and f.name != "SRC_DATA"
        ]
        self._opt_args = [
            f.name
            for f in dataclasses.fields(_SampleBlockData)
            if f.default is not dataclasses.MISSING
        ]


def _build_inp(src, **extras):
    """Mirror the real pipeline's ``inp`` construction for DATA_TYPE=None blocks.

    ``transform_input`` passes every key from ``src_data`` through into ``inp``
    alongside the block's own outputs, plus ``SRC_DATA``. Tests that call
    ``transform_output`` directly must preserve that invariant, otherwise the
    default-map's echo entries point at keys that don't exist on ``inp``.
    """
    return {**src, **extras, "SRC_DATA": src}


# ===========================================================================
#                       transform_output — dict src_data
# ===========================================================================
def test_flat_write_preserved():
    """Flat paths on both sides behave identically to the pre-DSL implementation."""
    block = _NullBlock(output_map={"tags": "result_tags"})
    src = {"question": "q"}
    inp = _build_inp(src, tags=["a", "b"])

    out = block.transform_output(inp, {"tags": "result_tags"})

    assert out is src
    assert out == {"question": "q", "result_tags": ["a", "b"]}


def test_nested_write_path_creates_intermediate_dicts():
    """DSL path on the ``v`` side auto-creates nested dicts on the source row."""
    block = _NullBlock()
    src = {"question": "q"}
    inp = _build_inp(src, score=0.9)

    block.transform_output(inp, {"score": "annotations.magpie.score"})

    assert src == {
        "question": "q",
        "annotations": {"magpie": {"score": 0.9}},
    }


def test_nested_read_path_from_block_output():
    """DSL path on the ``k`` side reads nested values from the block's output dict."""
    block = _NullBlock()
    src = {}
    inp = _build_inp(src, labels={"primary": "safe", "secondary": "low_risk"})

    block.transform_output(inp, {"labels.primary": "safety_label"})

    assert src == {"safety_label": "safe"}


def test_nested_read_and_write_combined():
    """DSL applies on both sides in a single entry."""
    block = _NullBlock()
    src = {}
    inp = _build_inp(src, result={"scores": [{"value": 0.7}, {"value": 0.3}]})

    block.transform_output(inp, {"result.scores[0].value": "metadata.top_score"})

    assert src == {"metadata": {"top_score": 0.7}}


def test_append_modifier_on_destination():
    """``[+]`` grows a list on the source row across multiple calls."""
    block = _NullBlock()
    src = {}

    block.transform_output(_build_inp(src, tag="a"), {"tag": "tags[+]"})
    block.transform_output(_build_inp(src, tag="b"), {"tag": "tags[+]"})

    assert src == {"tags": ["a", "b"]}


def test_append_intermediate_grows_list_of_annotations():
    """``[+]`` at an intermediate segment appends a fresh dict and descends."""
    block = _NullBlock()
    src = {}

    block.transform_output(_build_inp(src, score=0.9), {"score": "annotations[+].magpie.score"})
    block.transform_output(_build_inp(src, score=0.4), {"score": "annotations[+].magpie.score"})

    assert src == {
        "annotations": [
            {"magpie": {"score": 0.9}},
            {"magpie": {"score": 0.4}},
        ]
    }


def test_missing_k_raises_keyerror():
    """A typo in the ``k`` side surfaces as a KeyError rather than silent ``None``."""
    block = _NullBlock()
    src = {}
    inp = _build_inp(src, actual_field=1)

    with pytest.raises(KeyError):
        block.transform_output(inp, {"typo_field": "result"})


def test_none_intermediate_on_destination_treated_as_absent():
    """An ``Optional[Dict] = None`` style source field is materialized on write."""
    block = _NullBlock()
    src = {"magpie_tags": None}
    inp = _build_inp(src, label="gold")

    block.transform_output(inp, {"label": "magpie_tags.metadata.label"})

    assert src == {"magpie_tags": {"metadata": {"label": "gold"}}}


def test_present_none_value_flows_through():
    """A block output that is present and ``None`` must be written as ``None``, not treated as missing."""
    block = _NullBlock()
    src = {}
    inp = _build_inp(src, maybe=None)

    block.transform_output(inp, {"maybe": "annotations.maybe"})

    assert src == {"annotations": {"maybe": None}}


# ===========================================================================
#                       utils.from_dataclass / utils.to_dataclass
# ===========================================================================
def test_from_dataclass_reads_flat_field():
    obj = _SamplePoint(task_name="t", question="q", tags=["a"])
    assert from_dataclass(obj, "tags") == ["a"]


def test_from_dataclass_reads_through_dict():
    obj = _SamplePoint(task_name="t", question="q", annotations={"magpie": {"score": 0.9}})
    assert from_dataclass(obj, "annotations.magpie.score") == 0.9


def test_from_dataclass_reads_through_list():
    obj = _SamplePoint(task_name="t", question="q", tags=["a", "b", "c"])
    assert from_dataclass(obj, "tags[1]") == "b"


def test_from_dataclass_undeclared_field_raises():
    obj = _SamplePoint(task_name="t", question="q")
    with pytest.raises(AttributeError):
        from_dataclass(obj, "no_such_field")


def test_from_dataclass_undeclared_field_raises_even_when_not_strict():
    """Schema errors are never silenced, unlike missing dict keys."""
    obj = _SamplePoint(task_name="t", question="q")
    with pytest.raises(AttributeError):
        from_dataclass(obj, "no_such_field", strict=False)


def test_from_dataclass_missing_dict_terminal_strict_raises():
    obj = _SamplePoint(task_name="t", question="q", annotations={"magpie": {"score": 0.9}})
    with pytest.raises(KeyError):
        from_dataclass(obj, "annotations.missing")


def test_from_dataclass_missing_dict_terminal_non_strict_returns_missing():
    obj = _SamplePoint(task_name="t", question="q", annotations={"magpie": {"score": 0.9}})
    assert from_dataclass(obj, "annotations.missing", strict=False) is MISSING


def test_from_dataclass_none_intermediate_raises():
    """A declared field whose value is None is not traversable on reads."""
    obj = _SamplePoint(task_name="t", question="q", annotations=None)
    with pytest.raises(KeyError):
        from_dataclass(obj, "annotations.magpie")


def test_to_dataclass_sets_flat_field():
    obj = _SamplePoint(task_name="t", question="q")
    to_dataclass(obj, "tags", ["a", "b"])
    assert obj.tags == ["a", "b"]


def test_to_dataclass_materializes_none_optional_dict_field():
    """Intermediate None on a declared dataclass field is replaced on write."""
    obj = _SamplePoint(task_name="t", question="q", annotations=None)
    to_dataclass(obj, "annotations.magpie.score", 0.9)
    assert obj.annotations == {"magpie": {"score": 0.9}}


def test_to_dataclass_materializes_none_optional_list_field():
    """Intermediate None with a bracket modifier creates a fresh list."""
    obj = _SamplePoint(task_name="t", question="q", tags=None)
    to_dataclass(obj, "tags[+]", "alpha")
    to_dataclass(obj, "tags[+]", "beta")
    assert obj.tags == ["alpha", "beta"]


def test_to_dataclass_append_intermediate_grows_list_of_dicts():
    obj = _SamplePoint(task_name="t", question="q")
    to_dataclass(obj, "annotations.entries[+].name", "a")
    to_dataclass(obj, "annotations.entries[+].name", "b")
    assert obj.annotations == {"entries": [{"name": "a"}, {"name": "b"}]}


def test_to_dataclass_undeclared_field_raises_valueerror():
    """Typos on dataclass fields must fail — never silently create attributes."""
    obj = _SamplePoint(task_name="t", question="q")
    with pytest.raises(ValueError):
        to_dataclass(obj, "no_such_field", "x")


def test_to_dataclass_undeclared_nested_field_raises():
    """Undeclared fields raise at any depth, not only at the terminal."""

    @dataclass
    class _Outer:
        inner: Optional["_Inner"] = None

    @dataclass
    class _Inner:
        x: Optional[int] = None

    # declare before use: rebuild via assignment order
    outer = _Outer(inner=_Inner(x=1))
    with pytest.raises(ValueError):
        to_dataclass(outer, "inner.y", 2)


def test_to_dataclass_reserved_src_data_raises():
    obj = _SamplePoint(task_name="t", question="q")
    with pytest.raises(ValueError):
        to_dataclass(obj, "SRC_DATA", "oops")


def test_to_dataclass_bracket_on_non_list_field_raises():
    """``[+]`` targeting a declared field that isn't a list is an error."""
    obj = _SamplePoint(task_name="t", question="q", annotations={"magpie": {}})
    with pytest.raises(TypeError):
        to_dataclass(obj, "question[+]", "x")


def test_to_dataclass_slice_write_rejected():
    obj = _SamplePoint(task_name="t", question="q", tags=["a"])
    with pytest.raises(ValueError):
        to_dataclass(obj, "tags[:1]", "x")


# ===========================================================================
#                       transform_output — dataclass src_data
# ===========================================================================
def _dc_inp(src, **fields_to_set):
    """Build a ``_SampleBlockData`` instance matching what transform_input would produce.

    The real pipeline populates every declared field (required + optional,
    with defaults) before handing ``inp`` to ``execute`` / ``transform_output``,
    plus ``SRC_DATA``. Mirror that so the default-map echo has every key it expects.
    """
    return _SampleBlockData(SRC_DATA=src, **fields_to_set)


def test_transform_output_dataclass_flat_write():
    """Flat dataclass write works end-to-end (regression guard)."""
    block = _DataclassBlock(output_map={"score": "metadata"})
    src = _SamplePoint(task_name="t", question="q")
    inp = _dc_inp(src, score=0.75)

    out = block.transform_output(inp, {"score": "metadata"})

    assert out is src
    # ``_SamplePoint.metadata`` is declared ``Optional[Dict[str, Any]]``; setattr
    # assigns the raw value without leaf-type enforcement.
    assert out.metadata == 0.75


def test_transform_output_dataclass_nested_write_creates_dict():
    """Nested path on ``v`` materializes dict inside a ``None`` dataclass field."""
    block = _DataclassBlock()
    src = _SamplePoint(task_name="t", question="q", annotations=None)
    inp = _dc_inp(src, score=0.9)

    block.transform_output(inp, {"score": "annotations.magpie.score"})

    assert src.annotations == {"magpie": {"score": 0.9}}


def test_transform_output_dataclass_append_grows_list_field():
    """``[+]`` on a dataclass list field appends across calls."""
    block = _DataclassBlock()
    src = _SamplePoint(task_name="t", question="q", tags=None)

    block.transform_output(_dc_inp(src, tag="a"), {"tag": "tags[+]"})
    block.transform_output(_dc_inp(src, tag="b"), {"tag": "tags[+]"})

    assert src.tags == ["a", "b"]


def test_transform_output_dataclass_undeclared_destination_raises():
    """The old ``hasattr`` silent-skip is intentionally gone; typos must raise."""
    block = _DataclassBlock()
    src = _SamplePoint(task_name="t", question="q")
    inp = _dc_inp(src, score=0.5)

    with pytest.raises(ValueError):
        block.transform_output(inp, {"score": "typo_field"})


def test_transform_output_dataclass_nested_read_from_dict_inp():
    """``k`` side DSL works when the block produced a DATA_TYPE=None dict."""

    class _NestedOutputBlock(Block):
        DATA_TYPE = None

        def __init__(self):
            self._name = "_n"
            self._block_type = "test"
            self._input_map = None
            self._output_map = None
            self._req_args, self._opt_args = [], []

    block = _NestedOutputBlock()
    src = _SamplePoint(task_name="t", question="q")
    # DATA_TYPE=None + dataclass src_data: the default map echoes every
    # declared field of ``src_data`` via ``_get_default_map``. Build ``inp``
    # so each of those fields has a value to read, mirroring what the real
    # pipeline produces via ``transform_input``.
    inp = {f.name: getattr(src, f.name) for f in dataclasses.fields(src) if f.name != "SRC_DATA"}
    inp["SRC_DATA"] = src
    inp["result"] = {"scores": [{"value": 0.7}]}

    block.transform_output(inp, {"result.scores[0].value": "metadata.top_score"})

    assert src.metadata == {"top_score": 0.7}


# ===========================================================================
#       transform_output — framework-bookkeeping exclusion (regression)
# ===========================================================================
def test_default_map_excludes_store_names_on_dataclass_src():
    """Framework-reserved ``store_names`` must not appear in the default
    output_map, so a user ``DataPoint`` that omits the field is not required
    to declare it just to satisfy the echo. Regression for the
    direct-call-on-dataclass-list failure documented at
    ``.claude/discussions/transform-output-bookkeeping-fields-leak.md``.
    """

    @dataclass
    class _UserDP(DataPoint):
        payload: str = ""

    @dataclass(kw_only=True)
    class _ToyData(ValidatorBlockData):
        payload: str = ""

    class _ToyValidator(ValidatorBlock):
        DATA_TYPE = _ToyData

        def __init__(self):
            self._name = "toy"
            self._block_type = "test"
            self._input_map = None
            self._output_map = None
            self._req_args = []
            self._opt_args = [
                f.name for f in dataclasses.fields(_ToyData) if f.default is not dataclasses.MISSING
            ]
            self._filter_invalids = True
            # Bypass datastore / profiling machinery for this unit.
            self._datastores = None
            self.profiler_data = {"executions": []}
            self._builder_name = None
            self._span_writer = _NoOpSpanWriter()
            self._logger = logging.getLogger(f"fms_dgt.block.{self._name}")

        def _validate(self, instance):
            return instance.payload != "", None

    v = _ToyValidator()
    out = list(v([_UserDP(task_name="t", payload="hello"), _UserDP(task_name="t", payload="")]))

    assert len(out) == 1
    assert out[0].payload == "hello"


def test_default_map_silent_skip_for_unknown_is_valid_field():
    """Framework-synthesized default-map entries that target a field
    ``src_data`` does not declare are silently skipped. User-declared
    ``output_map`` entries remain strict (see
    ``test_transform_output_dataclass_undeclared_destination_raises``).
    """
    block = _DataclassBlock()  # DATA_TYPE=_SampleBlockData, no is_valid field
    src = _SamplePoint(task_name="t", question="q")
    inp = _dc_inp(src, score=0.5)

    # Default map echoes score/labels/tag. No user-declared map given, so the
    # absence of any field on _SamplePoint that the block might try to write
    # must not raise. (Happy-path existing tests already cover this shape, but
    # this pins the contract explicitly.)
    out = block.transform_output(inp, {})
    assert out is src
    assert out.score == 0.5


def test_user_declared_typo_still_raises_after_fix():
    """Fix must not re-introduce silent-skip for caller typos. Guards against
    over-broadening the dataclass-branch strict check.
    """
    block = _DataclassBlock()
    src = _SamplePoint(task_name="t", question="q")
    inp = _dc_inp(src, score=0.5)

    # User-declared path-side typo: strict-raise preserved.
    with pytest.raises(ValueError):
        block.transform_output(inp, {"score": "nonexistent_target"})


# ===========================================================================
#                       transform_input — dict src_data
# ===========================================================================
class _InputBlock(Block):
    """Block with a real dataclass DATA_TYPE for exercising ``transform_input``.

    Constructor bypasses the registry / datastore machinery; ``_req_args`` and
    ``_opt_args`` mirror what ``Block.__init__`` would compute from
    :class:`_SampleBlockData`.
    """

    DATA_TYPE = _SampleBlockData

    def __init__(self, input_map=None):
        self._name = "_in"
        self._block_type = "test"
        self._input_map = input_map
        self._output_map = None
        self._req_args = [
            f.name
            for f in dataclasses.fields(_SampleBlockData)
            if f.default is dataclasses.MISSING and f.name != "SRC_DATA"
        ]
        self._opt_args = [
            f.name
            for f in dataclasses.fields(_SampleBlockData)
            if f.default is not dataclasses.MISSING
        ]


def test_transform_input_flat_mapping():
    """A flat input_map routes row fields to declared dataclass args."""
    block = _InputBlock(input_map={"question_text": "labels"})
    row = {"question_text": {"primary": "safe"}}

    out = block.transform_input(row, block._input_map)

    assert out.labels == {"primary": "safe"}
    assert out.SRC_DATA is row


def test_transform_input_missing_optional_source_falls_back_to_default():
    """An optional source field that's absent on the row lets the dataclass default apply."""
    block = _InputBlock(input_map={"missing_key": "labels"})
    row = {"question_text": "q"}

    out = block.transform_input(row, block._input_map)

    # labels declared default is None; absence of "missing_key" must not override it.
    assert out.labels is None


def test_transform_input_present_none_flows_through():
    """A row key whose value is None passes None through, not default."""
    block = _InputBlock(input_map={"maybe": "labels"})
    row = {"maybe": None}

    out = block.transform_input(row, block._input_map)

    # None is a real value and must win over the dataclass default.
    assert out.labels is None  # matches default here, but see next test for distinction


def test_transform_input_nested_path_reads_terminal():
    """Nested DSL on the input_map reads through dicts and lists."""
    block = _InputBlock(input_map={"messages[0].content": "tag"})
    row = {"messages": [{"content": "hello"}, {"content": "world"}]}

    out = block.transform_input(row, block._input_map)

    assert out.tag == "hello"


def test_transform_input_nested_terminal_missing_falls_back_to_default():
    """Nested path whose terminal is absent behaves like flat absent: default applies."""
    block = _InputBlock(input_map={"messages[0].missing": "tag"})
    row = {"messages": [{"content": "hello"}]}

    out = block.transform_input(row, block._input_map)

    assert out.tag is None  # declared default on _SampleBlockData.tag


def test_transform_input_nested_intermediate_missing_raises():
    """Intermediate segment absent means the user asserted a shape the row lacks — loud error."""
    block = _InputBlock(input_map={"messages[0].content": "tag"})
    row = {"question_text": "q"}  # no 'messages' at all

    with pytest.raises(KeyError):
        block.transform_input(row, block._input_map)


def test_transform_input_missing_required_arg_raises_valueerror():
    """A required dataclass field with no resolvable source must raise the existing error."""

    @dataclass(kw_only=True)
    class _RequiredBlockData(BlockData):
        needed: str  # no default → becomes a required arg

    class _RequiredBlock(Block):
        DATA_TYPE = _RequiredBlockData

        def __init__(self):
            self._name = "_r"
            self._block_type = "test"
            self._input_map = None
            self._output_map = None
            self._req_args = ["needed"]
            self._opt_args = []

    block = _RequiredBlock()
    # Row has no mapping for "needed" and no input_map provided.
    row = {"unrelated": "value"}

    with pytest.raises(ValueError):
        block.transform_input(row, None)


# ===========================================================================
#       transform_input — framework-bookkeeping supply route (regression)
# ===========================================================================
def test_transform_input_default_map_carries_store_names_to_block_data():
    """Framework bookkeeping (``store_names``) injected on the input dict
    must reach the block-side ``BlockData`` instance via the default map.

    Regression for the asymmetric-direction fix: the output-side exclusion
    must not also strip ``store_names`` on the input side, otherwise
    ``Block.save_data`` finds ``store_names=None`` on every rejected
    instance and silently drops persistence (rejection log fires, but no
    file is written under the per-block datastore).

    See ``.claude/discussions/transform-output-bookkeeping-fields-leak.md``
    "Correction" section.
    """
    block = _InputBlock()
    row = {
        "question_text": "q",
        "score": 0.7,
        "labels": {"primary": "safe"},
        "tag": "x",
        "store_names": ["per_task_store"],
    }

    out = block.transform_input(row, block._input_map)

    # The whole point of the supply route: caller-injected store_names
    # lands on the constructed BlockData where save_data can read it.
    assert out.store_names == ["per_task_store"]
    # Other declared fields still flow as before.
    assert out.score == 0.7
    assert out.labels == {"primary": "safe"}
    assert out.tag == "x"


def test_transform_output_default_map_still_skips_store_names():
    """Direction asymmetry guard: the output-side default map must continue
    to exclude ``store_names`` even though the input side carries it. A
    user ``DataPoint`` that does not declare ``store_names`` must survive
    the echo without raising.
    """
    src = _SamplePoint(task_name="t", question="q")  # no store_names field
    block = _DataclassBlock()
    inp = _dc_inp(src, score=0.5)

    # Default output_map only — no user-declared keys. Must not raise on
    # the absent store_names field on _SamplePoint.
    out = block.transform_output(inp, {})

    assert out is src
    assert out.score == 0.5

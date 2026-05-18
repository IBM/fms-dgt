# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# Local
from fms_dgt.core.tools.engines import (
    LMToolEngine,
    ToolEngine,
    get_tool_engine,
    register_tool_engine,
)

# Local — shared helpers used across engine test modules
from tests.core.tools.engines.helpers import _make_registry

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    def test_get_lm_engine(self):
        with patch("fms_dgt.core.tools.engines.lm.get_block") as mock_get_block:
            mock_get_block.return_value = MagicMock()
            eng = get_tool_engine("lm", _make_registry(), lm_config={"type": "mock"})
        assert isinstance(eng, LMToolEngine)

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_tool_engine("nonexistent_engine_xyz", _make_registry())

    def test_register_custom_engine(self):
        @register_tool_engine("_test_custom_engine_xyz")
        class _CustomEngine(ToolEngine):
            def execute(self, session_id, tool_calls):
                return []

        eng = get_tool_engine("_test_custom_engine_xyz", _make_registry())
        assert isinstance(eng, _CustomEngine)

    def test_duplicate_registration_raises(self):
        with pytest.raises(AssertionError, match="conflicts"):

            @register_tool_engine("lm")
            class _Duplicate(ToolEngine):
                def execute(self, session_id, tool_calls):
                    return []

# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Third Party
import anthropic
import pytest

# Local
from fms_dgt.base.telemetry import _NoOpSpanWriter
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider
from fms_dgt.core.blocks.llm.anthropic import Anthropic
from tests.core.blocks.llm.test_llm import (
    LM_CFG,
    auto_chat_template_test,
    execute_chat_completion_flow,
    execute_completion_flow,
)

LM_ANTHROPIC_CFG = {
    **LM_CFG,
    "type": "anthropic",
    "model_id_or_path": "claude-3-haiku-20240307",
    "max_tokens": 25,
}


# ===========================================================================
#                       HELPERS (unit tests)
# ===========================================================================

_FAKE_KEY = "sk-ant-test-unit"


def _make_provider(**kwargs) -> Anthropic:
    """Create an Anthropic provider with a mocked client and no-op telemetry."""
    with patch("fms_dgt.core.blocks.llm.anthropic.get_resource") as mock_res:
        mock_res.return_value = MagicMock(key=_FAKE_KEY)
        provider = Anthropic(
            type="anthropic",
            name="test-anthropic",
            model_id_or_path="claude-3-haiku-20240307",
            **kwargs,
        )
    provider._span_writer = _NoOpSpanWriter()
    provider.async_client = MagicMock()
    return provider


def _text_response(text: str, input_tokens: int = 10, output_tokens: int = 5):
    """Anthropic message response with a single TextBlock."""
    content = [anthropic.types.TextBlock(type="text", text=text)]
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=content, usage=usage, completion=text)


def _tool_use_response(
    tool_name: str, tool_id: str, tool_input: dict, input_tokens: int = 8, output_tokens: int = 3
):
    """Anthropic message response with a single ToolUseBlock (no text content)."""
    content = [
        anthropic.types.ToolUseBlock(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
    ]
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=content, usage=usage)


def _mixed_response(
    text: str,
    tool_name: str,
    tool_id: str,
    tool_input: dict,
    input_tokens: int = 15,
    output_tokens: int = 8,
):
    """Anthropic message response with both TextBlock and ToolUseBlock."""
    content = [
        anthropic.types.TextBlock(type="text", text=text),
        anthropic.types.ToolUseBlock(type="tool_use", id=tool_id, name=tool_name, input=tool_input),
    ]
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=content, usage=usage)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_completion(model_cfg):
    with pytest.raises(RuntimeError) as exc_info:
        execute_completion_flow(model_cfg, prompts=["Question: x = 0 + 1\nAnswer: x ="])

    assert (
        exc_info.value.args[0]
        == 'Support for "completion" method for newer models has been deprecated as per "Anthropic" documentation: https://docs.anthropic.com/en/api/complete'
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_chat_completion(model_cfg):
    execute_chat_completion_flow(
        model_cfg,
        conversations=[
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello! How were you trained?"},
            ]
        ],
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic key is not available")
@pytest.mark.parametrize("model_cfg", [LM_ANTHROPIC_CFG])
def test_auto_chat_template(model_cfg):
    with pytest.raises(NotImplementedError) as exc_info:
        auto_chat_template_test(model_cfg)

    assert (
        exc_info.value.args[0]
        == 'Tokenization support is disabled for "Antropic" provider. Certain capabilites like "apply_chat_template", "truncation" will be unavailable.'
    )


# ===========================================================================
#                       UNIT TESTS (_extract_choice_content)
# ===========================================================================


class TestExtractChoiceContent:
    def test_text_only_response(self):
        provider = _make_provider(max_tokens=100)
        response = _text_response("the answer is 42")
        result = provider._extract_choice_content(response, method=LMProvider.CHAT_COMPLETION)
        assert result["role"] == "assistant"
        assert result["content"] == "the answer is 42"
        assert "tool_calls" not in result

    def test_tool_use_response_has_tool_calls(self):
        provider = _make_provider(max_tokens=100)
        response = _tool_use_response("search", "tc-001", {"query": "test"})
        result = provider._extract_choice_content(response, method=LMProvider.CHAT_COMPLETION)
        assert result["role"] == "assistant"
        assert "content" not in result
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "tc-001"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == {"query": "test"}

    def test_mixed_response_has_both_content_and_tool_calls(self):
        provider = _make_provider(max_tokens=100)
        response = _mixed_response("I will search", "search", "tc-002", {"query": "mixed"})
        result = provider._extract_choice_content(response, method=LMProvider.CHAT_COMPLETION)
        assert result["content"] == "I will search"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_multiple_text_blocks_joined(self):
        provider = _make_provider(max_tokens=100)
        content = [
            anthropic.types.TextBlock(type="text", text="hello"),
            anthropic.types.TextBlock(type="text", text="world"),
        ]
        response = SimpleNamespace(
            content=content, usage=SimpleNamespace(input_tokens=5, output_tokens=2)
        )
        result = provider._extract_choice_content(response, method=LMProvider.CHAT_COMPLETION)
        assert result["content"] == "hello world"


# ===========================================================================
#                       UNIT TESTS (chat_completion glue)
# ===========================================================================


class TestChatCompletionGlue:
    def _run_chat(self, provider: Anthropic, instance: LMBlockData, mock_response) -> None:
        provider.async_client.messages = MagicMock()
        provider.async_client.messages.create = AsyncMock(return_value=mock_response)
        provider.chat_completion([instance], disable_tqdm=True)

    def test_result_set_on_instance(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(SRC_DATA={}, input=[{"role": "user", "content": "say hello"}])
        self._run_chat(provider, instance, _text_response("hello there"))
        assert instance.result is not None

    def test_token_usage_in_addtl(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(SRC_DATA={}, input=[{"role": "user", "content": "say hello"}])
        self._run_chat(provider, instance, _text_response("hi", input_tokens=11, output_tokens=6))
        assert instance.addtl["prompt_tokens"] == 11
        assert instance.addtl["completion_tokens"] == 6

    def test_tool_call_result_contains_tool_calls(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(
            SRC_DATA={},
            input=[{"role": "user", "content": "search for something"}],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        )
        self._run_chat(
            provider, instance, _tool_use_response("search", "tc-003", {"query": "something"})
        )
        assert isinstance(instance.result, dict)
        assert instance.result["tool_calls"][0]["function"]["name"] == "search"

    def test_completion_method_raises(self):
        provider = _make_provider(max_tokens=100)
        with pytest.raises(RuntimeError, match="deprecated"):
            provider.completion([])

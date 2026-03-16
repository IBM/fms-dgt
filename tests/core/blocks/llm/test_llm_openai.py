# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpenAI provider glue code.

These tests mock the OpenAI HTTP client so no real API calls are made.
They cover:
- ``_extract_choice_content`` for text and tool-call responses
- ``_llm_span`` attribute filtering (null n / max_tokens omitted)
- Token usage propagated from response to ``LMBlockData.addtl``
- Result written to ``LMBlockData.result`` after a completion call
"""

# Standard
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Local
from fms_dgt.base.telemetry import _NoOpSpanWriter
from fms_dgt.core.blocks.llm import LMBlockData, LMProvider
from fms_dgt.core.blocks.llm.openai import OpenAI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_KEY = "sk-test-openai-unit"


def _make_provider(**kwargs) -> OpenAI:
    """Create an OpenAI provider with a mocked async client and no-op telemetry."""
    with patch("fms_dgt.core.blocks.llm.openai.get_resource") as mock_res:
        mock_res.return_value = MagicMock(key=_FAKE_KEY)
        provider = OpenAI(
            type="openai",
            name="test-openai",
            model_id_or_path="gpt-4o",
            **kwargs,
        )
    provider._span_writer = _NoOpSpanWriter()
    provider.async_client = MagicMock()
    return provider


def _text_choice(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a minimal OpenAI completion response with one text choice."""
    choice = SimpleNamespace(text=text, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _chat_message_mock(to_dict_value: dict):
    """Return a MagicMock that looks like a ChatCompletionMessage to isinstance."""
    # Third Party
    import openai.types.chat.chat_completion_message as _mod

    msg = MagicMock(spec=_mod.ChatCompletionMessage)
    msg.to_dict.return_value = to_dict_value
    return msg


def _chat_text_choice(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a minimal chat completion response with plain text content."""
    msg = _chat_message_mock({"role": "assistant", "content": content})
    choice = SimpleNamespace(message=msg, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _chat_tool_call_choice(
    tool_name: str, arguments: str, prompt_tokens: int = 8, completion_tokens: int = 3
):
    """Build a chat completion response that contains a tool call (content=None)."""
    msg = _chat_message_mock(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc-001",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ],
        }
    )
    choice = SimpleNamespace(message=msg, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# _extract_choice_content
# ---------------------------------------------------------------------------


class TestExtractChoiceContent:
    def test_completion_returns_text(self):
        provider = _make_provider()
        choice = SimpleNamespace(text="hello world", logprobs=None)
        result = provider._extract_choice_content(choice, method=LMProvider.COMPLETION)
        assert result == "hello world"

    def test_chat_text_returns_dict(self):
        provider = _make_provider()
        response = _chat_text_choice("the answer is 42")
        result = provider._extract_choice_content(
            response.choices[0], method=LMProvider.CHAT_COMPLETION
        )
        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert result["content"] == "the answer is 42"

    def test_chat_tool_call_returns_dict_with_tool_calls(self):
        provider = _make_provider()
        response = _chat_tool_call_choice("search", '{"query": "test"}')
        result = provider._extract_choice_content(
            response.choices[0], method=LMProvider.CHAT_COMPLETION
        )
        assert isinstance(result, dict)
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# _llm_span: null attribute filtering
# ---------------------------------------------------------------------------


class TestLlmSpanNullFiltering:
    """Verify that span attrs omit n / max_tokens when not set in params."""

    def _run_span(self, provider: OpenAI, params: dict) -> dict:
        """Run _llm_span and capture the attrs dict it yields."""
        captured = {}

        async def _inner():
            async with provider._llm_span("completion", batch_size=1, params=params) as attrs:
                captured.update(attrs)

        provider.run_async(_inner())
        return captured

    def test_no_null_n_when_not_set(self):
        provider = _make_provider()
        attrs = self._run_span(provider, {"temperature": 0.7})
        assert "n" not in attrs

    def test_no_null_max_tokens_when_not_set(self):
        provider = _make_provider()
        attrs = self._run_span(provider, {"temperature": 0.7})
        assert "max_tokens" not in attrs

    def test_n_present_when_set(self):
        provider = _make_provider()
        attrs = self._run_span(provider, {"n": 3})
        assert attrs["n"] == 3

    def test_max_tokens_resolved_from_max_completion_tokens(self):
        provider = _make_provider()
        attrs = self._run_span(provider, {"max_completion_tokens": 256})
        assert attrs["max_tokens"] == 256

    def test_temperature_present_when_set(self):
        provider = _make_provider()
        attrs = self._run_span(provider, {"temperature": 0.5})
        assert attrs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# chat_completion: result + token usage propagation
# ---------------------------------------------------------------------------


class TestChatCompletionGlue:
    """End-to-end through _process_item using a mocked async client."""

    def _run_chat(self, provider: OpenAI, instance: LMBlockData, mock_response) -> None:
        provider.async_client.chat = MagicMock()
        provider.async_client.chat.completions = MagicMock()
        provider.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider.chat_completion([instance], disable_tqdm=True)

    def test_result_set_on_instance(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(
            SRC_DATA={},
            input=[{"role": "user", "content": "say hello"}],
        )
        response = _chat_text_choice("hello there", prompt_tokens=10, completion_tokens=4)
        self._run_chat(provider, instance, response)
        assert instance.result is not None
        assert isinstance(instance.result, (str, dict))

    def test_token_usage_in_addtl(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(
            SRC_DATA={},
            input=[{"role": "user", "content": "say hello"}],
        )
        response = _chat_text_choice("hello", prompt_tokens=12, completion_tokens=7)
        self._run_chat(provider, instance, response)
        assert instance.addtl is not None
        assert instance.addtl["prompt_tokens"] == 12
        assert instance.addtl["completion_tokens"] == 7

    def test_tool_call_result_contains_tool_calls(self):
        provider = _make_provider(max_tokens=100)
        instance = LMBlockData(
            SRC_DATA={},
            input=[{"role": "user", "content": "search for something"}],
            tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        )
        response = _chat_tool_call_choice("search", '{"query": "something"}')
        self._run_chat(provider, instance, response)
        result = instance.result
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "search"

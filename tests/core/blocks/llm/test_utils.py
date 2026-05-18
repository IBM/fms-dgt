# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fms_dgt.core.blocks.llm.utils.

Covers:
- ``retry`` decorator: sync and async paths, exponential backoff, Retry-After
  header honoured, eventual success after transient failures.
- ``_retry_after_seconds``: header present, header absent, no .response attr.
"""

# Standard
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Local
from fms_dgt.core.blocks.llm.utils import _retry_after_seconds, retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeError(Exception):
    """Simulates an SDK error that may carry a Retry-After header."""

    def __init__(self, message="rate limited", retry_after: float | None = None):
        super().__init__(message)
        if retry_after is not None:
            self.response = MagicMock()
            self.response.headers = MagicMock()
            self.response.headers.get = lambda key, default=None: (
                str(retry_after) if key == "retry-after" else default
            )
        # no .response attribute when retry_after is None — matches exceptions
        # that do not carry an HTTP response (e.g. connection errors)


# ---------------------------------------------------------------------------
# _retry_after_seconds
# ---------------------------------------------------------------------------


def test_retry_after_seconds_uses_header_when_present():
    exc = _FakeError(retry_after=42.0)
    assert _retry_after_seconds(exc, backoff_time=10.0) == 42.0


def test_retry_after_seconds_falls_back_to_backoff_when_no_response():
    exc = _FakeError()  # no .response attribute
    assert _retry_after_seconds(exc, backoff_time=7.5) == 7.5


def test_retry_after_seconds_falls_back_when_header_missing():
    exc = _FakeError()
    exc.response = MagicMock()
    exc.response.headers = MagicMock()
    exc.response.headers.get = lambda key, default=None: default
    assert _retry_after_seconds(exc, backoff_time=5.0) == 5.0


# ---------------------------------------------------------------------------
# retry — sync path
# ---------------------------------------------------------------------------


def test_retry_sync_succeeds_immediately():
    call_count = 0

    @retry(on_exceptions=(_FakeError,), max_retries=3, backoff_time=0.0)
    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    with patch("fms_dgt.core.blocks.llm.utils.time.sleep"):
        result = fn()

    assert result == "ok"
    assert call_count == 1


def test_retry_sync_retries_on_exception_then_succeeds():
    attempts = []

    @retry(on_exceptions=(_FakeError,), max_retries=3, backoff_time=0.0)
    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise _FakeError()
        return "done"

    with patch("fms_dgt.core.blocks.llm.utils.time.sleep"):
        result = fn()

    assert result == "done"
    assert len(attempts) == 3


def test_retry_sync_exhausts_retries_silently():
    """After max_retries failures the wrapper returns None (no re-raise)."""

    @retry(on_exceptions=(_FakeError,), max_retries=2, backoff_time=0.0)
    def fn():
        raise _FakeError()

    with patch("fms_dgt.core.blocks.llm.utils.time.sleep"):
        result = fn()

    assert result is None


def test_retry_sync_uses_retry_after_header():
    """sync wrapper must pass the Retry-After value to time.sleep."""

    @retry(on_exceptions=(_FakeError,), max_retries=2, backoff_time=99.0)
    def fn():
        raise _FakeError(retry_after=1.5)

    with patch("fms_dgt.core.blocks.llm.utils.time.sleep") as mock_sleep:
        fn()

    for call in mock_sleep.call_args_list:
        assert call.args[0] == 1.5


# ---------------------------------------------------------------------------
# retry — async path
# ---------------------------------------------------------------------------


def test_retry_async_succeeds_immediately():
    call_count = 0

    @retry(on_exceptions=(_FakeError,), max_retries=3, backoff_time=0.0)
    async def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    with patch("fms_dgt.core.blocks.llm.utils.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(fn())

    assert result == "ok"
    assert call_count == 1


def test_retry_async_retries_on_exception_then_succeeds():
    attempts = []

    @retry(on_exceptions=(_FakeError,), max_retries=3, backoff_time=0.0)
    async def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise _FakeError()
        return "done"

    with patch("fms_dgt.core.blocks.llm.utils.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(fn())

    assert result == "done"
    assert len(attempts) == 3


def test_retry_async_uses_asyncio_sleep_not_time_sleep():
    """The async wrapper must call asyncio.sleep, never time.sleep."""

    @retry(on_exceptions=(_FakeError,), max_retries=2, backoff_time=0.0)
    async def fn():
        raise _FakeError()

    with (
        patch(
            "fms_dgt.core.blocks.llm.utils.asyncio.sleep", new_callable=AsyncMock
        ) as mock_async_sleep,
        patch("fms_dgt.core.blocks.llm.utils.time.sleep") as mock_time_sleep,
    ):
        asyncio.run(fn())

    assert mock_async_sleep.call_count > 0
    mock_time_sleep.assert_not_called()


def test_retry_async_uses_retry_after_header():
    """async wrapper must pass the Retry-After value to asyncio.sleep."""

    @retry(on_exceptions=(_FakeError,), max_retries=2, backoff_time=99.0)
    async def fn():
        raise _FakeError(retry_after=2.5)

    with patch("fms_dgt.core.blocks.llm.utils.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        asyncio.run(fn())

    for call in mock_sleep.call_args_list:
        assert call.args[0] == 2.5

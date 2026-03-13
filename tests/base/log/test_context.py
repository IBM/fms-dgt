# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import threading

# Third Party
import pytest

# Local
from fms_dgt.log.context import RunContextFilter, _run_ctx, run_context

# ===========================================================================
#                       run_context()
# ===========================================================================


def test_run_context_sets_and_clears():
    assert _run_ctx.get() is None

    with run_context("build-1", "run-1"):
        ctx = _run_ctx.get()
        assert ctx["build_id"] == "build-1"
        assert ctx["run_id"] == "run-1"

    assert _run_ctx.get() is None


def test_run_context_auto_run_id():
    with run_context("build-x") as _:
        ctx = _run_ctx.get()
        assert ctx["build_id"] == "build-x"
        # run_id should be a non-empty UUID-ish string
        assert ctx["run_id"]


def test_run_context_clears_on_exception():
    with pytest.raises(ValueError):
        with run_context("build-err"):
            raise ValueError("boom")

    assert _run_ctx.get() is None


def test_run_context_nesting():
    with run_context("outer", "run-outer"):
        with run_context("inner", "run-inner"):
            ctx = _run_ctx.get()
            assert ctx["build_id"] == "inner"
            assert ctx["run_id"] == "run-inner"
        # outer is restored
        ctx = _run_ctx.get()
        assert ctx["build_id"] == "outer"


def test_run_context_thread_isolation():
    """Mutations in a child thread do not leak into the parent."""
    results = {}

    def worker():
        with run_context("thread-build", "thread-run"):
            results["thread"] = _run_ctx.get()

    with run_context("main-build", "main-run"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # Parent context is unchanged
        assert _run_ctx.get()["build_id"] == "main-build"

    assert results["thread"]["build_id"] == "thread-build"


# ===========================================================================
#                       RunContextFilter
# ===========================================================================


def _make_record(msg="test") -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_run_context_filter_injects_ids():
    f = RunContextFilter()
    record = _make_record()
    with run_context("b1", "r1"):
        f.filter(record)
    assert record.build_id == "b1"
    assert record.run_id == "r1"


def test_run_context_filter_empty_outside_context():
    f = RunContextFilter()
    record = _make_record()
    f.filter(record)
    assert record.build_id == ""
    assert record.run_id == ""


def test_run_context_filter_always_returns_true():
    f = RunContextFilter()
    record = _make_record()
    assert f.filter(record) is True
    with run_context("b", "r"):
        assert f.filter(record) is True

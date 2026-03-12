# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock
import logging
import threading

# Local
from fms_dgt.log.context import run_context
from fms_dgt.log.handlers import FanOutHandler, LogDatastoreHandler

# ===========================================================================
#                       Helpers
# ===========================================================================


def _record(msg="hello", **extra) -> logging.LogRecord:
    r = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(r, k, v)
    return r


# ===========================================================================
#                       FanOutHandler
# ===========================================================================


def test_fanout_emit_to_all_handlers():
    h1, h2 = MagicMock(), MagicMock()
    fan = FanOutHandler()
    fan.register("task-a", h1)
    fan.register("task-b", h2)

    record = _record()
    fan.emit(record)

    h1.emit.assert_called_once_with(record)
    h2.emit.assert_called_once_with(record)


def test_fanout_unregister_stops_delivery():
    h1, h2 = MagicMock(), MagicMock()
    fan = FanOutHandler()
    fan.register("task-a", h1)
    fan.register("task-b", h2)
    fan.unregister("task-a")

    fan.emit(_record())

    h1.emit.assert_not_called()
    h2.emit.assert_called_once()


def test_fanout_register_is_idempotent():
    h = MagicMock()
    fan = FanOutHandler()
    fan.register("task-a", h)
    fan.register("task-a", h)  # duplicate

    fan.emit(_record())

    assert h.emit.call_count == 1


def test_fanout_no_handlers_is_safe():
    fan = FanOutHandler()
    fan.emit(_record())  # should not raise


def test_fanout_active_task_names():
    fan = FanOutHandler()
    fan.register("t1", MagicMock())
    fan.register("t2", MagicMock())
    assert set(fan.active_task_names) == {"t1", "t2"}
    fan.unregister("t1")
    assert fan.active_task_names == ["t2"]


def test_fanout_flush_calls_all():
    h1, h2 = MagicMock(), MagicMock()
    fan = FanOutHandler()
    fan.register("t1", h1)
    fan.register("t2", h2)
    fan.flush()
    h1.flush.assert_called_once()
    h2.flush.assert_called_once()


def test_fanout_thread_safe():
    """Concurrent register/emit should not raise or lose records."""
    received = []
    lock = threading.Lock()

    class CountHandler(logging.Handler):
        def emit(self, record):
            with lock:
                received.append(record)

    fan = FanOutHandler()
    ch = CountHandler()
    fan.register("t", ch)

    def emit_many():
        for _ in range(50):
            fan.emit(_record())

    threads = [threading.Thread(target=emit_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 200


# ===========================================================================
#                       LogDatastoreHandler
# ===========================================================================


def test_log_datastore_handler_writes_structured_record():
    ds = MagicMock()
    handler = LogDatastoreHandler(ds)
    handler.emit(_record("a message"))

    ds.save_data.assert_called_once()
    entry = ds.save_data.call_args[0][0][0]
    assert entry["message"] == "a message"
    assert entry["level"] == "INFO"
    assert "timestamp" in entry
    assert "logger" in entry


def test_log_datastore_handler_preserves_extra_fields():
    ds = MagicMock()
    handler = LogDatastoreHandler(ds)
    handler.emit(_record("msg", task_name="my_task", epoch=3))

    entry = ds.save_data.call_args[0][0][0]
    assert entry["task_name"] == "my_task"
    assert entry["epoch"] == 3


def test_log_datastore_handler_injects_run_context():
    ds = MagicMock()
    handler = LogDatastoreHandler(ds)
    with run_context("build-99", "run-99"):
        handler.emit(_record("msg"))

    entry = ds.save_data.call_args[0][0][0]
    assert entry["build_id"] == "build-99"
    assert entry["run_id"] == "run-99"


def test_log_datastore_handler_skips_run_context_when_already_present():
    ds = MagicMock()
    handler = LogDatastoreHandler(ds)
    r = _record("msg", build_id="explicit-build", run_id="explicit-run")
    with run_context("other-build", "other-run"):
        handler.emit(r)

    entry = ds.save_data.call_args[0][0][0]
    assert entry["build_id"] == "explicit-build"
    assert entry["run_id"] == "explicit-run"


def test_log_datastore_handler_close_closes_datastore():
    ds = MagicMock()
    handler = LogDatastoreHandler(ds)
    handler.close()
    ds.close.assert_called_once()


def test_log_datastore_handler_emit_error_does_not_raise():
    ds = MagicMock()
    ds.save_data.side_effect = RuntimeError("disk full")
    handler = LogDatastoreHandler(ds)
    handler.handleError = MagicMock()  # suppress default stderr output
    handler.emit(_record("msg"))  # should not raise
    handler.handleError.assert_called_once()

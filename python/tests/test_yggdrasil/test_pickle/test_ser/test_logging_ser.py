"""
test_logging_ser.py – Tests for yggdrasil.pickle.ser.logging

Covers:
- Wire-tag assignment and Tags registration
- Round-trip serialization via Serialized.from_python_object / as_python
- Round-trip via the public dumps / loads API
- Each concrete class: LogRecord, Formatter, Handler, Logger
- Handler subclasses (NullHandler, StreamHandler, MemoryHandler)
- Formatters with custom style / datefmt / defaults
- Loggers by name, root logger, level propagation
- Filters attached to handlers
- LogRecord with exception info
- LogRecord with stack info
- Tags.get_class round-trip (tag → class)
- Internal helper unit tests (_filter_names, _extract_handler_state, _handler_module_qualname)
"""
from __future__ import annotations

import logging
import logging.handlers as _lh
import sys
from io import StringIO

import pytest

from yggdrasil.pickle.ser import dumps, loads
from yggdrasil.pickle.ser.logging import (
    FormatterSerialized,
    HandlerSerialized,
    LoggingSerialized,
    LogRecordSerialized,
    LoggerSerialized,
    _extract_handler_state,
    _filter_names,
    _handler_module_qualname,
    _serialize_nested,
    _deserialize_nested,
    _opt_serialize,
    _opt_deserialize,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    name: str = "test",
    level: int = logging.INFO,
    msg: str = "hello",
    args: tuple = (),
    pathname: str = "/app/test.py",
    lineno: int = 10,
    func: str = "test_fn",
    exc_info=None,
    sinfo: str | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=args,
        exc_info=exc_info,
        func=func,
        sinfo=sinfo,
    )


def _roundtrip_obj(obj: object) -> object:
    """Serialize then deserialize via the public dumps/loads API."""
    return loads(dumps(obj))


def _roundtrip_serialized(obj: object) -> object:
    """Serialize then deserialize via the low-level Serialized API."""
    ser = Serialized.from_python_object(obj)
    assert ser is not None
    return ser.as_python()


# ===========================================================================
# Tags
# ===========================================================================

class TestTags:
    def test_logging_logger_tag(self) -> None:
        assert Tags.LOGGING_LOGGER == 216

    def test_logging_handler_tag(self) -> None:
        assert Tags.LOGGING_HANDLER == 217

    def test_logging_formatter_tag(self) -> None:
        assert Tags.LOGGING_FORMATTER == 218

    def test_logging_log_record_tag(self) -> None:
        assert Tags.LOGGING_LOG_RECORD == 219

    def test_all_logging_tags_are_system_range(self) -> None:
        for tag in (
            Tags.LOGGING_LOGGER,
            Tags.LOGGING_HANDLER,
            Tags.LOGGING_FORMATTER,
            Tags.LOGGING_LOG_RECORD,
        ):
            assert Tags.is_system(tag), f"Tag {tag} is not in system range"

    def test_tags_registered_in_classes(self) -> None:
        # Force registration by importing
        import yggdrasil.pickle.ser.logging  # noqa: F401

        assert Tags.LOGGING_LOGGER in Tags.CLASSES
        assert Tags.LOGGING_HANDLER in Tags.CLASSES
        assert Tags.LOGGING_FORMATTER in Tags.CLASSES
        assert Tags.LOGGING_LOG_RECORD in Tags.CLASSES

    def test_logger_type_registered(self) -> None:
        import yggdrasil.pickle.ser.logging  # noqa: F401
        assert Tags.TYPES.get(logging.Logger) is LoggerSerialized

    def test_handler_type_registered(self) -> None:
        import yggdrasil.pickle.ser.logging  # noqa: F401
        assert Tags.TYPES.get(logging.Handler) is HandlerSerialized

    def test_formatter_type_registered(self) -> None:
        import yggdrasil.pickle.ser.logging  # noqa: F401
        assert Tags.TYPES.get(logging.Formatter) is FormatterSerialized

    def test_logrecord_type_registered(self) -> None:
        import yggdrasil.pickle.ser.logging  # noqa: F401
        assert Tags.TYPES.get(logging.LogRecord) is LogRecordSerialized

    def test_get_class_by_tag_logger(self) -> None:
        cls = Tags.get_class(Tags.LOGGING_LOGGER)
        assert cls is LoggerSerialized

    def test_get_class_by_tag_handler(self) -> None:
        cls = Tags.get_class(Tags.LOGGING_HANDLER)
        assert cls is HandlerSerialized

    def test_get_class_by_tag_formatter(self) -> None:
        cls = Tags.get_class(Tags.LOGGING_FORMATTER)
        assert cls is FormatterSerialized

    def test_get_class_by_tag_logrecord(self) -> None:
        cls = Tags.get_class(Tags.LOGGING_LOG_RECORD)
        assert cls is LogRecordSerialized

    def test_tag_to_name_includes_logging(self) -> None:
        assert Tags.TAG_TO_NAME[216] == "LOGGING_LOGGER"
        assert Tags.TAG_TO_NAME[217] == "LOGGING_HANDLER"
        assert Tags.TAG_TO_NAME[218] == "LOGGING_FORMATTER"
        assert Tags.TAG_TO_NAME[219] == "LOGGING_LOG_RECORD"


# ===========================================================================
# Internal helpers
# ===========================================================================

class TestInternalHelpers:
    def test_filter_names_empty(self) -> None:
        h = logging.NullHandler()
        assert _filter_names(h) == []

    def test_filter_names_with_filters(self) -> None:
        h = logging.NullHandler()
        h.addFilter(logging.Filter("app.db"))
        h.addFilter(logging.Filter("app.web"))
        names = _filter_names(h)
        assert names == ["app.db", "app.web"]

    def test_filter_names_skips_non_filter_callables(self) -> None:
        h = logging.NullHandler()
        h.addFilter(lambda r: True)          # callable, not a Filter
        h.addFilter(logging.Filter("keep"))
        names = _filter_names(h)
        assert names == ["keep"]

    def test_extract_handler_state_null_handler(self) -> None:
        h = logging.NullHandler()
        state = _extract_handler_state(h)
        # NullHandler has no meaningful extra state
        assert state is None or isinstance(state, dict)

    def test_extract_handler_state_stream_handler(self) -> None:
        h = logging.StreamHandler(StringIO())
        state = _extract_handler_state(h)
        # stream itself is not in the public dict keys we capture,
        # but the return must be None or a dict
        assert state is None or isinstance(state, dict)

    def test_handler_module_qualname_null_handler(self) -> None:
        h = logging.NullHandler()
        mod, qual = _handler_module_qualname(h)
        assert mod == "logging"
        assert qual == "NullHandler"

    def test_handler_module_qualname_stream_handler(self) -> None:
        h = logging.StreamHandler()
        mod, qual = _handler_module_qualname(h)
        assert mod == "logging"
        assert qual == "StreamHandler"

    def test_serialize_deserialize_nested_primitive(self) -> None:
        blob = _serialize_nested(42)
        assert isinstance(blob, bytes)
        assert _deserialize_nested(blob) == 42

    def test_serialize_deserialize_nested_string(self) -> None:
        blob = _serialize_nested("hello")
        assert _deserialize_nested(blob) == "hello"

    def test_serialize_deserialize_nested_tuple(self) -> None:
        t = (1, "two", 3.0)
        blob = _serialize_nested(t)
        assert _deserialize_nested(blob) == t

    def test_opt_serialize_none(self) -> None:
        assert _opt_serialize(None) is None

    def test_opt_deserialize_none(self) -> None:
        assert _opt_deserialize(None) is None

    def test_opt_serialize_deserialize_roundtrip(self) -> None:
        blob = _opt_serialize({"key": "val"})
        assert blob is not None
        assert _opt_deserialize(blob) == {"key": "val"}


# ===========================================================================
# LogRecord
# ===========================================================================

class TestLogRecordSerialized:
    def test_tag(self) -> None:
        record = _make_record()
        ser = Serialized.from_python_object(record)
        assert ser.tag == Tags.LOGGING_LOG_RECORD

    def test_is_log_record_serialized(self) -> None:
        record = _make_record()
        ser = Serialized.from_python_object(record)
        assert isinstance(ser, LogRecordSerialized)

    def test_name_roundtrip(self) -> None:
        record = _make_record(name="myapp.core")
        out = _roundtrip_serialized(record)
        assert isinstance(out, logging.LogRecord)
        assert out.name == "myapp.core"

    def test_level_roundtrip(self) -> None:
        record = _make_record(level=logging.ERROR)
        out = _roundtrip_serialized(record)
        assert out.levelno == logging.ERROR

    def test_pathname_roundtrip(self) -> None:
        record = _make_record(pathname="/srv/app/main.py")
        out = _roundtrip_serialized(record)
        assert out.pathname == "/srv/app/main.py"

    def test_lineno_roundtrip(self) -> None:
        record = _make_record(lineno=99)
        out = _roundtrip_serialized(record)
        assert out.lineno == 99

    def test_msg_roundtrip(self) -> None:
        record = _make_record(msg="something happened %s")
        out = _roundtrip_serialized(record)
        assert out.msg == "something happened %s"

    def test_args_roundtrip(self) -> None:
        record = _make_record(msg="count=%d", args=(7,))
        out = _roundtrip_serialized(record)
        assert out.args == (7,)

    def test_func_name_roundtrip(self) -> None:
        record = _make_record(func="my_handler_fn")
        out = _roundtrip_serialized(record)
        assert out.funcName == "my_handler_fn"

    def test_stack_info_roundtrip(self) -> None:
        record = _make_record(sinfo="Stack (most recent call last):\n  ...")
        out = _roundtrip_serialized(record)
        assert out.stack_info == "Stack (most recent call last):\n  ..."

    def test_no_exc_info(self) -> None:
        record = _make_record(exc_info=None)
        out = _roundtrip_serialized(record)
        assert out.exc_info is None or out.exc_info[0] is None

    def test_exc_info_roundtrip(self) -> None:
        try:
            raise ValueError("test exception")
        except ValueError:
            ei = sys.exc_info()

        record = _make_record(exc_info=ei)
        out = _roundtrip_serialized(record)
        # Traceback objects are not serialisable; we store (type, value, None).
        # The type and a reconstructed exception value must survive.
        assert out.exc_info is not None
        assert out.exc_info[0] is ValueError
        assert isinstance(out.exc_info[1], ValueError)
        assert "test exception" in str(out.exc_info[1])

    def test_dumps_loads_roundtrip(self) -> None:
        record = _make_record(name="dumps.test", level=logging.WARNING, msg="warn %s", args=("!",))
        out = _roundtrip_obj(record)
        assert isinstance(out, logging.LogRecord)
        assert out.name == "dumps.test"
        assert out.levelno == logging.WARNING

    def test_from_value_class_method(self) -> None:
        record = _make_record()
        ser = LogRecordSerialized.from_value(record)
        assert isinstance(ser, LogRecordSerialized)
        out = ser.as_python()
        assert isinstance(out, logging.LogRecord)

    def test_multiple_records_are_independent(self) -> None:
        r1 = _make_record(name="a", level=logging.DEBUG)
        r2 = _make_record(name="b", level=logging.CRITICAL)
        o1 = _roundtrip_serialized(r1)
        o2 = _roundtrip_serialized(r2)
        assert o1.name == "a"
        assert o2.name == "b"
        assert o1.levelno == logging.DEBUG
        assert o2.levelno == logging.CRITICAL

    def test_all_log_levels(self) -> None:
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            r = _make_record(level=level)
            out = _roundtrip_serialized(r)
            assert out.levelno == level


# ===========================================================================
# Formatter
# ===========================================================================

class TestFormatterSerialized:
    def test_tag(self) -> None:
        fmt = logging.Formatter()
        ser = Serialized.from_python_object(fmt)
        assert ser.tag == Tags.LOGGING_FORMATTER

    def test_is_formatter_serialized(self) -> None:
        fmt = logging.Formatter()
        ser = Serialized.from_python_object(fmt)
        assert isinstance(ser, FormatterSerialized)

    def test_default_formatter_roundtrip(self) -> None:
        fmt = logging.Formatter()
        out = _roundtrip_serialized(fmt)
        assert isinstance(out, logging.Formatter)

    def test_fmt_string_roundtrip(self) -> None:
        fmt = logging.Formatter("%(levelname)s: %(message)s")
        out = _roundtrip_serialized(fmt)
        assert isinstance(out, logging.Formatter)
        assert out._fmt == "%(levelname)s: %(message)s"

    def test_datefmt_roundtrip(self) -> None:
        fmt = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S")
        out = _roundtrip_serialized(fmt)
        assert out.datefmt == "%Y-%m-%d %H:%M:%S"

    def test_percent_style_roundtrip(self) -> None:
        fmt = logging.Formatter("%(message)s", style="%")
        out = _roundtrip_serialized(fmt)
        assert isinstance(out._style, logging.PercentStyle)

    def test_brace_style_roundtrip(self) -> None:
        fmt = logging.Formatter("{message}", style="{", validate=False)
        out = _roundtrip_serialized(fmt)
        assert isinstance(out._style, logging.StrFormatStyle)

    def test_dollar_style_roundtrip(self) -> None:
        fmt = logging.Formatter("${message}", style="$", validate=False)
        out = _roundtrip_serialized(fmt)
        assert isinstance(out._style, logging.StringTemplateStyle)
    def test_validate_false_roundtrip(self) -> None:
        fmt = logging.Formatter("%(missing_key)s", validate=False)
        out = _roundtrip_serialized(fmt)
        assert isinstance(out, logging.Formatter)

    def test_dumps_loads_roundtrip(self) -> None:
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M")
        out = _roundtrip_obj(fmt)
        assert isinstance(out, logging.Formatter)
        assert out.datefmt == "%H:%M"

    def test_from_value_class_method(self) -> None:
        fmt = logging.Formatter("%(name)s %(message)s")
        ser = FormatterSerialized.from_value(fmt)
        assert isinstance(ser, FormatterSerialized)
        out = ser.as_python()
        assert isinstance(out, logging.Formatter)

    def test_formatter_formats_record(self) -> None:
        fmt = logging.Formatter("LEVEL=%(levelname)s MSG=%(message)s")
        out: logging.Formatter = _roundtrip_serialized(fmt)
        record = _make_record(level=logging.ERROR, msg="oops")
        text = out.format(record)
        assert "LEVEL=ERROR" in text
        assert "MSG=oops" in text


# ===========================================================================
# Handler
# ===========================================================================

class TestHandlerSerialized:
    def test_tag(self) -> None:
        h = logging.NullHandler()
        ser = Serialized.from_python_object(h)
        assert ser.tag == Tags.LOGGING_HANDLER

    def test_is_handler_serialized(self) -> None:
        h = logging.NullHandler()
        ser = Serialized.from_python_object(h)
        assert isinstance(ser, HandlerSerialized)

    def test_null_handler_roundtrip(self) -> None:
        h = logging.NullHandler()
        out = _roundtrip_serialized(h)
        assert isinstance(out, logging.Handler)

    def test_stream_handler_roundtrip(self) -> None:
        h = logging.StreamHandler()
        out = _roundtrip_serialized(h)
        assert isinstance(out, logging.Handler)

    def test_level_preserved(self) -> None:
        h = logging.NullHandler()
        h.setLevel(logging.WARNING)
        out = _roundtrip_serialized(h)
        assert out.level == logging.WARNING

    def test_all_levels_preserved(self) -> None:
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            h = logging.NullHandler()
            h.setLevel(level)
            out = _roundtrip_serialized(h)
            assert out.level == level

    def test_formatter_preserved(self) -> None:
        fmt = logging.Formatter("%(levelname)s %(message)s")
        h = logging.NullHandler()
        h.setFormatter(fmt)
        out = _roundtrip_serialized(h)
        assert out.formatter is not None
        assert isinstance(out.formatter, logging.Formatter)
        assert out.formatter._fmt == "%(levelname)s %(message)s"

    def test_no_formatter_preserved(self) -> None:
        h = logging.NullHandler()
        out = _roundtrip_serialized(h)
        assert out.formatter is None

    def test_filters_preserved(self) -> None:
        h = logging.NullHandler()
        h.addFilter(logging.Filter("app.db"))
        h.addFilter(logging.Filter("app.api"))
        out = _roundtrip_serialized(h)
        filter_names = [f.name for f in out.filters if isinstance(f, logging.Filter)]
        assert "app.db" in filter_names
        assert "app.api" in filter_names

    def test_dumps_loads_roundtrip(self) -> None:
        h = logging.NullHandler()
        h.setLevel(logging.ERROR)
        out = _roundtrip_obj(h)
        assert isinstance(out, logging.Handler)
        assert out.level == logging.ERROR

    def test_from_value_class_method(self) -> None:
        h = logging.NullHandler()
        ser = HandlerSerialized.from_value(h)
        assert isinstance(ser, HandlerSerialized)
        out = ser.as_python()
        assert isinstance(out, logging.Handler)

    def test_memory_handler_roundtrip(self) -> None:
        h = _lh.MemoryHandler(capacity=100)
        h.setLevel(logging.DEBUG)
        out = _roundtrip_serialized(h)
        assert isinstance(out, logging.Handler)
        assert out.level == logging.DEBUG

    def test_unknown_handler_class_falls_back_to_null_handler(self) -> None:
        """
        When the handler class cannot be resolved during deserialization,
        a NullHandler fallback is used.
        """
        from yggdrasil.pickle.ser.logging import _serialize_nested, _VERSION

        # Build a valid HandlerSerialized payload that references a non-existent class
        bad_payload = (
            _VERSION,
            "no_such_module_xyz",   # module
            "BogusHandler",          # qualname
            logging.WARNING,         # level
            None,                    # formatter_blob
            [],                      # filter_names
            None,                    # extra_state_blob
        )
        data = _serialize_nested(bad_payload)
        rebuilt = HandlerSerialized.build(tag=Tags.LOGGING_HANDLER, data=data)
        # Should not raise; falls back gracefully to NullHandler
        out = rebuilt.as_python()
        assert isinstance(out, logging.Handler)

    def test_handler_with_formatter_and_filter(self) -> None:
        fmt = logging.Formatter("%(name)s %(message)s")
        h = logging.NullHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        h.addFilter(logging.Filter("svc"))

        out = _roundtrip_serialized(h)

        assert out.level == logging.INFO
        assert out.formatter is not None
        assert out.formatter._fmt == "%(name)s %(message)s"
        filter_names = [f.name for f in out.filters if isinstance(f, logging.Filter)]
        assert "svc" in filter_names


# ===========================================================================
# Logger
# ===========================================================================

class TestLoggerSerialized:
    def test_tag(self) -> None:
        # Use from_value directly to guarantee LoggerSerialized regardless of
        # any custom Logger class installed by third-party packages (e.g. pip).
        logger = logging.getLogger("test.tag")
        ser = LoggerSerialized.from_value(logger)
        assert ser.tag == Tags.LOGGING_LOGGER

    def test_is_logger_serialized(self) -> None:
        logger = logging.getLogger("test.is_logger")
        ser = LoggerSerialized.from_value(logger)
        assert isinstance(ser, LoggerSerialized)

    def test_name_preserved(self) -> None:
        logger = logging.getLogger("test.name.preserved")
        out = _roundtrip_serialized(logger)
        assert isinstance(out, logging.Logger)
        assert out.name == "test.name.preserved"

    def test_level_set_when_unset(self) -> None:
        logger = logging.getLogger("test.level.unset")
        logger.setLevel(logging.DEBUG)
        out = _roundtrip_serialized(logger)
        assert out.level == logging.DEBUG

    def test_singleton_semantics(self) -> None:
        """Round-tripping a Logger returns the same singleton from getLogger."""
        name = "test.singleton"
        logger = logging.getLogger(name)
        out = _roundtrip_serialized(logger)
        assert out is logging.getLogger(name)

    def test_root_logger_roundtrip(self) -> None:
        root = logging.getLogger()
        ser = Serialized.from_python_object(root)
        assert isinstance(ser, LoggerSerialized)
        out = ser.as_python()
        assert isinstance(out, logging.Logger)
        assert out is logging.getLogger()

    def test_dumps_loads_roundtrip(self) -> None:
        logger = logging.getLogger("test.dumps")
        logger.setLevel(logging.WARNING)
        out = _roundtrip_obj(logger)
        assert isinstance(out, logging.Logger)
        assert out.name == "test.dumps"

    def test_from_value_class_method(self) -> None:
        logger = logging.getLogger("test.from_value")
        ser = LoggerSerialized.from_value(logger)
        assert isinstance(ser, LoggerSerialized)
        out = ser.as_python()
        assert isinstance(out, logging.Logger)
        assert out.name == "test.from_value"

    def test_different_loggers_produce_distinct_names(self) -> None:
        l1 = logging.getLogger("test.l1")
        l2 = logging.getLogger("test.l2")
        o1 = _roundtrip_serialized(l1)
        o2 = _roundtrip_serialized(l2)
        assert o1.name == "test.l1"
        assert o2.name == "test.l2"

    def test_root_logger_tag(self) -> None:
        root = logging.getLogger()
        ser = LoggerSerialized.from_value(root)
        assert ser.tag == Tags.LOGGING_LOGGER


# ===========================================================================
# LoggingSerialized base dispatch
# ===========================================================================

class TestLoggingSerializedDispatch:
    def test_dispatch_log_record(self) -> None:
        record = _make_record()
        ser = LoggingSerialized.from_python_object(record)
        assert isinstance(ser, LogRecordSerialized)

    def test_dispatch_formatter(self) -> None:
        fmt = logging.Formatter()
        ser = LoggingSerialized.from_python_object(fmt)
        assert isinstance(ser, FormatterSerialized)

    def test_dispatch_handler(self) -> None:
        h = logging.NullHandler()
        ser = LoggingSerialized.from_python_object(h)
        assert isinstance(ser, HandlerSerialized)

    def test_dispatch_logger(self) -> None:
        logger = logging.getLogger("test.dispatch")
        ser = LoggingSerialized.from_python_object(logger)
        assert isinstance(ser, LoggerSerialized)

    def test_dispatch_returns_none_for_unknown(self) -> None:
        ser = LoggingSerialized.from_python_object(object())
        assert ser is None

    def test_dispatch_formatter_before_handler(self) -> None:
        """Formatter is a subclass path; must be dispatched before Handler."""
        fmt = logging.Formatter("%(message)s")
        ser = LoggingSerialized.from_python_object(fmt)
        assert isinstance(ser, FormatterSerialized)


# ===========================================================================
# Cross-class / integration tests
# ===========================================================================

class TestIntegration:
    def test_logger_with_named_handlers_via_serialized(self) -> None:
        """Independently serialize each component of a logger setup."""
        fmt = logging.Formatter("%(levelname)s %(name)s %(message)s")
        handler = logging.NullHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)

        logger = logging.getLogger("integration.test.full")
        logger.setLevel(logging.DEBUG)

        # Serialize each independently
        fmt_out = _roundtrip_serialized(fmt)
        handler_out = _roundtrip_serialized(handler)
        logger_out = _roundtrip_serialized(logger)

        assert isinstance(fmt_out, logging.Formatter)
        assert isinstance(handler_out, logging.Handler)
        assert isinstance(logger_out, logging.Logger)
        assert logger_out.name == "integration.test.full"

    def test_logrecord_formats_correctly_after_roundtrip(self) -> None:
        fmt = logging.Formatter("%(levelname)s: %(message)s")
        record = _make_record(level=logging.WARNING, msg="check this")
        out_fmt: logging.Formatter = _roundtrip_serialized(fmt)
        out_rec: logging.LogRecord = _roundtrip_serialized(record)
        text = out_fmt.format(out_rec)
        assert "WARNING" in text
        assert "check this" in text

    def test_handler_chain_level_filtering(self) -> None:
        """Handler with level ERROR should accept ERROR but reject INFO."""
        h = logging.NullHandler()
        h.setLevel(logging.ERROR)
        out: logging.Handler = _roundtrip_serialized(h)
        record_info = _make_record(level=logging.INFO)
        record_error = _make_record(level=logging.ERROR)
        # logging internally gates on: record.levelno >= handler.level
        assert record_info.levelno < out.level   # INFO rejected
        assert record_error.levelno >= out.level  # ERROR accepted

    def test_serialized_api_preserves_magic(self) -> None:
        from yggdrasil.pickle.ser.constants import MAGIC
        record = _make_record()
        data = dumps(record)
        assert data.startswith(MAGIC)

    def test_all_objects_produce_bytes_via_dumps(self) -> None:
        objects = [
            _make_record(),
            logging.Formatter("%(message)s"),
            logging.NullHandler(),
            logging.getLogger("test.all"),
        ]
        for obj in objects:
            data = dumps(obj)
            assert isinstance(data, bytes)
            assert len(data) > 0

    def test_all_objects_roundtrip_via_dumps_loads(self) -> None:
        objects: list[tuple[object, type]] = [
            (_make_record(name="rr", level=logging.DEBUG), logging.LogRecord),
            (logging.Formatter("%(levelname)s %(message)s"), logging.Formatter),
            (logging.NullHandler(), logging.Handler),
            (logging.getLogger("test.all.loads"), logging.Logger),
        ]
        for obj, expected_type in objects:
            out = _roundtrip_obj(obj)
            assert isinstance(out, expected_type), (
                f"Expected {expected_type.__name__}, got {type(out).__name__}"
            )


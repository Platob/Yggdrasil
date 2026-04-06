"""
Unit tests for yggdrasil.pickle.ser.exceptions
(BaseExceptionSerialized, traceback capture, chained exceptions, backward compat).
"""

from __future__ import annotations

import sys
from typing import cast

import pytest

from yggdrasil.pickle.ser import Serialized, Tags
from yggdrasil.pickle.ser.exceptions import (
    BaseExceptionSerialized,
    TracebackFrame,
    _dump_exception_payload,
    _load_exception_payload,
    _extract_traceback_frames,
    _EXC_FORMAT_V2,
    _MAX_CHAIN_DEPTH,
)
from yggdrasil.pickle.ser.complexs import ComplexSerialized


# ---------------------------------------------------------------------------
# helper exception types
# ---------------------------------------------------------------------------


class HttpError(Exception):
    """Custom exception with extra attributes."""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(code, msg)
        self.code = code
        self.msg = msg


class DatabaseError(RuntimeError):
    """Another custom exception type."""

    def __init__(self, query: str, detail: str = "") -> None:
        super().__init__(query, detail)
        self.query = query
        self.detail = detail


class SlottedError(Exception):
    """Exception with __slots__ (no __dict__ by default)."""

    __slots__ = ("code",)

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _raise_and_catch(exc: BaseException) -> BaseException:
    """Raise *exc* so it gets a live traceback, then return it."""
    try:
        raise exc
    except type(exc) as caught:
        return caught


def _raise_chained_cause(inner: BaseException, outer: BaseException) -> BaseException:
    """Build ``raise outer from inner`` and catch ``outer``."""
    try:
        try:
            raise inner
        except type(inner):
            raise outer from inner
    except type(outer) as caught:
        return caught


def _raise_chained_context(inner: BaseException, outer: BaseException) -> BaseException:
    """Build implicit context chain and catch ``outer``."""
    try:
        try:
            raise inner
        except type(inner):
            raise outer
    except type(outer) as caught:
        return caught


# ===========================================================================
# TracebackFrame
# ===========================================================================


class TestTracebackFrame:
    def test_named_tuple_fields(self) -> None:
        frame = TracebackFrame(
            filename="/a/b/c.py",
            lineno=42,
            name="my_func",
            line="x = 1 + 1",
        )
        assert frame.filename == "/a/b/c.py"
        assert frame.lineno == 42
        assert frame.name == "my_func"
        assert frame.line == "x = 1 + 1"

    def test_none_line_allowed(self) -> None:
        frame = TracebackFrame(filename="x.py", lineno=1, name="f", line=None)
        assert frame.line is None

    def test_indexable(self) -> None:
        frame = TracebackFrame("f.py", 10, "g", "pass")
        assert frame[0] == "f.py"
        assert frame[1] == 10
        assert frame[2] == "g"
        assert frame[3] == "pass"

    def test_equality(self) -> None:
        a = TracebackFrame("f.py", 1, "f", "x")
        b = TracebackFrame("f.py", 1, "f", "x")
        assert a == b

    def test_inequality_on_lineno(self) -> None:
        a = TracebackFrame("f.py", 1, "f", "x")
        b = TracebackFrame("f.py", 2, "f", "x")
        assert a != b


# ===========================================================================
# _extract_traceback_frames
# ===========================================================================


class TestExtractTracebackFrames:
    def test_no_traceback_returns_none(self) -> None:
        exc = ValueError("no tb")
        assert exc.__traceback__ is None
        assert _extract_traceback_frames(exc) is None

    def test_raised_exception_has_frames(self) -> None:
        exc = _raise_and_catch(ValueError("live tb"))
        frames = _extract_traceback_frames(exc)

        assert frames is not None
        assert len(frames) >= 1
        # Each frame is a 4-tuple
        for f in frames:
            assert len(f) == 4
            filename, lineno, name, line = f
            assert isinstance(filename, str)
            assert isinstance(lineno, int)
            assert lineno > 0
            assert isinstance(name, str)
            # line may be None in some edge cases

    def test_frame_name_matches_function(self) -> None:
        exc = _raise_and_catch(RuntimeError("check name"))
        frames = _extract_traceback_frames(exc)
        assert frames is not None
        names = [f[2] for f in frames]
        assert "_raise_and_catch" in names


# ===========================================================================
# _dump_exception_payload / _load_exception_payload — version 2 (round-trip)
# ===========================================================================


class TestDumpLoadPayload:
    def test_simple_value_error(self) -> None:
        exc = ValueError("bad input")
        data = _dump_exception_payload(exc)
        got = _load_exception_payload(data)

        assert isinstance(got, ValueError)
        assert got.args == ("bad input",)

    def test_custom_exception_attributes_preserved(self) -> None:
        exc = HttpError(404, "missing")
        data = _dump_exception_payload(exc)
        got = _load_exception_payload(data)

        assert isinstance(got, HttpError)
        assert got.args == (404, "missing")
        assert got.code == 404
        assert got.msg == "missing"

    def test_no_traceback_yields_none(self) -> None:
        exc = RuntimeError("no tb")
        data = _dump_exception_payload(exc)
        got = _load_exception_payload(data)

        assert hasattr(got, "__serialized_traceback__")
        assert got.__serialized_traceback__ is None

    def test_raised_exc_yields_traceback_frames(self) -> None:
        exc = _raise_and_catch(ValueError("live"))
        data = _dump_exception_payload(exc)
        got = _load_exception_payload(data)

        tb = got.__serialized_traceback__
        assert tb is not None
        assert isinstance(tb, list)
        assert len(tb) >= 1
        assert all(isinstance(f, TracebackFrame) for f in tb)

    def test_traceback_frame_fields_are_correct_types(self) -> None:
        exc = _raise_and_catch(KeyError("k"))
        got = _load_exception_payload(_dump_exception_payload(exc))

        tb = got.__serialized_traceback__
        assert tb is not None
        for frame in tb:
            assert isinstance(frame.filename, str)
            assert isinstance(frame.lineno, int)
            assert frame.lineno > 0
            assert isinstance(frame.name, str)
            assert frame.line is None or isinstance(frame.line, str)

    def test_exc_state_extra_attributes(self) -> None:
        exc = RuntimeError("extra")
        exc.extra = {"key": 123}  # type: ignore[attr-defined]

        got = _load_exception_payload(_dump_exception_payload(exc))

        assert isinstance(got, RuntimeError)
        assert got.extra == {"key": 123}  # type: ignore[attr-defined]

    def test_slotted_error(self) -> None:
        exc = SlottedError(42)
        got = _load_exception_payload(_dump_exception_payload(exc))

        assert isinstance(got, SlottedError)
        assert got.code == 42


# ===========================================================================
# Chained exceptions (__cause__ and __context__)
# ===========================================================================


class TestChainedExceptions:
    def test_cause_is_serialized(self) -> None:
        inner = ValueError("inner")
        outer = _raise_chained_cause(inner, RuntimeError("outer"))

        got = _load_exception_payload(_dump_exception_payload(outer))

        assert isinstance(got, RuntimeError)
        assert got.args == ("outer",)
        assert got.__cause__ is not None
        assert isinstance(got.__cause__, ValueError)
        assert got.__cause__.args == ("inner",)

    def test_cause_sets_suppress_context_true(self) -> None:
        inner = ValueError("inner")
        outer = _raise_chained_cause(inner, RuntimeError("outer"))

        got = _load_exception_payload(_dump_exception_payload(outer))

        # When __cause__ is set via `raise X from Y`, suppress_context is True
        assert got.__suppress_context__ is True

    def test_implicit_context_is_serialized(self) -> None:
        inner = ValueError("ctx")
        outer = _raise_chained_context(inner, RuntimeError("outer"))

        got = _load_exception_payload(_dump_exception_payload(outer))

        assert isinstance(got, RuntimeError)
        assert got.__context__ is not None
        assert isinstance(got.__context__, ValueError)
        assert got.__context__.args == ("ctx",)
        # Implicit chaining does not set __cause__
        assert got.__cause__ is None

    def test_no_chain_when_no_cause_or_context(self) -> None:
        exc = ValueError("standalone")
        got = _load_exception_payload(_dump_exception_payload(exc))

        assert got.__cause__ is None
        assert got.__context__ is None

    def test_cause_same_as_context_not_double_serialized(self) -> None:
        """When __cause__ and __context__ are the same object, __context__ is
        deduplicated (not serialized twice) to avoid redundant data."""
        inner = ValueError("shared")
        outer = _raise_chained_cause(inner, RuntimeError("outer"))
        # Python sets __context__ = inner AND __cause__ = inner in this case
        assert outer.__context__ is inner or outer.__cause__ is inner

        data = _dump_exception_payload(outer)
        got = _load_exception_payload(data)

        # Cause is reconstructed
        assert got.__cause__ is not None
        assert isinstance(got.__cause__, ValueError)

    def test_chained_exc_traceback_preserved(self) -> None:
        inner = _raise_and_catch(ValueError("inner with tb"))
        outer = _raise_chained_cause(inner, RuntimeError("outer"))

        got = _load_exception_payload(_dump_exception_payload(outer))

        assert got.__cause__ is not None
        cause_tb = got.__cause__.__serialized_traceback__
        assert cause_tb is not None
        assert len(cause_tb) >= 1

    def test_chain_depth_limit_respected(self) -> None:
        """Chains deeper than _MAX_CHAIN_DEPTH are truncated gracefully."""
        exc: BaseException = ValueError("root")
        for i in range(_MAX_CHAIN_DEPTH + 5):
            next_exc = RuntimeError(f"level {i}")
            next_exc.__cause__ = exc
            exc = next_exc

        # Should not raise or recurse infinitely
        data = _dump_exception_payload(exc)
        got = _load_exception_payload(data)

        # Walk the chain; it must terminate
        depth = 0
        current: BaseException | None = got
        while current is not None:
            depth += 1
            assert depth <= _MAX_CHAIN_DEPTH + 2, "chain longer than expected"
            current = current.__cause__


# ===========================================================================
# Legacy v1 payloads are rejected
# ===========================================================================


class TestV1PayloadRejected:
    """v1 payloads (4-tuple with version=1) must be rejected with a clear error."""

    def _make_v1_payload(self, exc: BaseException) -> bytes:
        """Build a version-1 payload identical to the old libs.py output."""
        from yggdrasil.pickle.ser.libs import _dump_object_state, _serialize_nested

        return _serialize_nested(
            (
                1,           # _EXC_FORMAT_V1
                type(exc),
                exc.args,
                _dump_object_state(exc),
            )
        )

    def test_v1_payload_raises_value_error(self) -> None:
        data = self._make_v1_payload(ValueError("old format"))
        with pytest.raises(ValueError):
            _load_exception_payload(data)

    def test_v1_wrong_length_message(self) -> None:
        data = self._make_v1_payload(RuntimeError("v1"))
        with pytest.raises(ValueError, match="expected 8-tuple"):
            _load_exception_payload(data)

    def test_wrong_version_number_rejected(self) -> None:
        from yggdrasil.pickle.ser.libs import _dump_object_state, _serialize_nested

        data = _serialize_nested(
            (99, ValueError, ("x",), _dump_object_state(ValueError("x")),
             None, None, None, False)
        )
        with pytest.raises(ValueError, match="Unsupported exception payload version"):
            _load_exception_payload(data)


# ===========================================================================
# BaseExceptionSerialized — class-level API
# ===========================================================================


class TestBaseExceptionSerialized:
    def test_tag_is_base_exception(self) -> None:
        assert BaseExceptionSerialized.TAG == Tags.BASE_EXCEPTION

    def test_build_exception_returns_instance(self) -> None:
        exc = ValueError("test")
        ser = BaseExceptionSerialized.build_exception(exc)

        assert isinstance(ser, BaseExceptionSerialized)

    def test_value_property_roundtrip(self) -> None:
        exc = ValueError("val prop")
        ser = BaseExceptionSerialized.build_exception(exc)

        got = ser.value
        assert isinstance(got, ValueError)
        assert got.args == ("val prop",)

    def test_as_python_alias(self) -> None:
        exc = RuntimeError("as python")
        ser = BaseExceptionSerialized.build_exception(exc)

        assert isinstance(ser.as_python(), RuntimeError)

    def test_build_exception_with_codec(self) -> None:
        from yggdrasil.pickle.ser.constants import CODEC_ZSTD
        exc = KeyError("zstd")
        ser = BaseExceptionSerialized.build_exception(exc, codec=CODEC_ZSTD)

        got = ser.as_python()
        assert isinstance(got, KeyError)
        assert got.args == ("zstd",)

    def test_write_to_read_from_roundtrip(self) -> None:
        exc = HttpError(401, "unauthorized")
        original = BaseExceptionSerialized.build_exception(exc)

        buf = original.write_to()
        reread = Serialized.read_from(buf, pos=0)

        assert isinstance(reread, BaseExceptionSerialized)
        assert reread.tag == Tags.BASE_EXCEPTION
        got = reread.as_python()
        assert isinstance(got, HttpError)
        assert got.code == 401
        assert got.msg == "unauthorized"

    def test_traceback_preserved_after_wire_roundtrip(self) -> None:
        exc = _raise_and_catch(DatabaseError("SELECT 1", "timeout"))
        original = BaseExceptionSerialized.build_exception(exc)

        buf = original.write_to()
        reread = Serialized.read_from(buf, pos=0)
        assert isinstance(reread, BaseExceptionSerialized)

        got = reread.as_python()
        tb = got.__serialized_traceback__
        assert tb is not None
        assert len(tb) >= 1
        assert all(isinstance(f, TracebackFrame) for f in tb)

    def test_no_traceback_after_wire_roundtrip(self) -> None:
        exc = ValueError("no raise")
        original = BaseExceptionSerialized.build_exception(exc)

        buf = original.write_to()
        reread = Serialized.read_from(buf, pos=0)
        assert isinstance(reread, BaseExceptionSerialized)

        got = reread.as_python()
        assert got.__serialized_traceback__ is None

    def test_cause_preserved_after_wire_roundtrip(self) -> None:
        inner = ValueError("inner cause")
        outer = _raise_chained_cause(inner, RuntimeError("outer"))

        buf = BaseExceptionSerialized.build_exception(outer).write_to()
        got = Serialized.read_from(buf, pos=0).as_python()

        assert isinstance(got, RuntimeError)
        assert got.__cause__ is not None
        assert isinstance(got.__cause__, ValueError)
        assert got.__cause__.args == ("inner cause",)

    def test_context_preserved_after_wire_roundtrip(self) -> None:
        inner = KeyError("ctx")
        outer = _raise_chained_context(inner, RuntimeError("outer"))

        buf = BaseExceptionSerialized.build_exception(outer).write_to()
        got = Serialized.read_from(buf, pos=0).as_python()

        assert isinstance(got, RuntimeError)
        assert got.__context__ is not None
        assert isinstance(got.__context__, KeyError)

    def test_subclasses_registered_with_tags(self) -> None:
        """BaseExceptionSerialized should be registered in Tags (via complexs.py)."""
        import yggdrasil.pickle.ser.complexs  # noqa: F401 — triggers registration
        cls = Tags.get_class(Tags.BASE_EXCEPTION)
        assert cls is BaseExceptionSerialized


# ===========================================================================
# Dispatching via ComplexSerialized.from_python_object
# ===========================================================================


class TestDispatch:
    def test_dispatches_exception_to_base_exception_serialized(self) -> None:
        exc = RuntimeError("dispatch")
        ser = Serialized.from_python_object(exc)

        assert isinstance(ser, BaseExceptionSerialized)

    def test_dispatches_base_exception_subclass(self) -> None:
        exc = HttpError(403, "forbidden")
        ser = Serialized.from_python_object(exc)

        assert isinstance(ser, BaseExceptionSerialized)
        got = ser.as_python()
        assert isinstance(got, HttpError)
        assert got.code == 403

    def test_dispatches_keyboard_interrupt(self) -> None:
        exc = KeyboardInterrupt()
        ser = Serialized.from_python_object(exc)

        assert isinstance(ser, BaseExceptionSerialized)
        got = ser.as_python()
        assert isinstance(got, KeyboardInterrupt)

    def test_dispatches_system_exit(self) -> None:
        exc = SystemExit(1)
        ser = Serialized.from_python_object(exc)

        assert isinstance(ser, BaseExceptionSerialized)
        got = ser.as_python()
        assert isinstance(got, SystemExit)
        assert got.code == 1

    def test_exception_type_not_dispatched_as_exception(self) -> None:
        """The *type* ValueError (not an instance) should go to ClassSerialized."""
        from yggdrasil.pickle.ser.complexs import ClassSerialized

        ser = Serialized.from_python_object(ValueError)
        assert isinstance(ser, ClassSerialized)

    def test_full_roundtrip_via_from_python_object(self) -> None:
        exc = _raise_and_catch(DatabaseError("INSERT", "deadlock"))
        ser = Serialized.from_python_object(exc)
        assert isinstance(ser, BaseExceptionSerialized)

        buf = ser.write_to()
        got = Serialized.read_from(buf, pos=0).as_python()

        assert isinstance(got, DatabaseError)
        assert got.query == "INSERT"
        assert got.detail == "deadlock"
        tb = got.__serialized_traceback__
        assert tb is not None and len(tb) >= 1


# ===========================================================================
# Backward-compat import paths
# ===========================================================================


class TestImportPaths:
    def test_import_from_complexs(self) -> None:
        from yggdrasil.pickle.ser.complexs import BaseExceptionSerialized as BES
        assert BES is BaseExceptionSerialized

    def test_import_from_exceptions_direct(self) -> None:
        from yggdrasil.pickle.ser.exceptions import BaseExceptionSerialized as BES
        assert BES is BaseExceptionSerialized

    def test_traceback_frame_importable_from_exceptions(self) -> None:
        from yggdrasil.pickle.ser.exceptions import TracebackFrame as TF
        assert TF is TracebackFrame


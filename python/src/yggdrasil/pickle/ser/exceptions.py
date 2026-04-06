"""
Exception serialization with full traceback support.

Wire format (version 2)
-----------------------
    (2, exc_class, args, state, tb_frames, cause_data, context_data, suppress_context)

    tb_frames        – ``None``  |  list of ``(filename, lineno, funcname, text)`` tuples
    cause_data       – ``None``  |  bytes  (nested exception payload for ``__cause__``)
    context_data     – ``None``  |  bytes  (nested exception payload for ``__context__``,

                                            omitted when identical to ``__cause__``)
    suppress_context – bool      (mirrors ``exc.__suppress_context__``)

On deserialization the exception's ``__serialized_traceback__`` attribute is set
to a list of :class:`TracebackFrame` named-tuples (or ``None`` when no traceback
was captured).  ``__cause__`` and ``__context__`` are reconstructed recursively up
to :data:`_MAX_CHAIN_DEPTH` levels.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

from yggdrasil.pickle.ser.libs import (
    ComplexSerialized,
    _deserialize_nested,
    _dump_object_state,
    _require_tuple,
    _require_tuple_len,
    _restore_object_state,
    _serialize_nested,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "BaseExceptionSerialized",
    "TracebackFrame",
    "_dump_exception_payload",
    "_load_exception_payload",
]

# ---------------------------------------------------------------------------
# wire-format version constant
# ---------------------------------------------------------------------------

_EXC_FORMAT_V2 = 2

# v2 payload indices
_V2_CLASS = 1
_V2_ARGS = 2
_V2_STATE = 3
_V2_TRACEBACK = 4
_V2_CAUSE = 5
_V2_CONTEXT = 6
_V2_SUPPRESS = 7
_V2_LENGTH = 8

# maximum chain depth for __cause__ / __context__ serialization
_MAX_CHAIN_DEPTH = 10


# ---------------------------------------------------------------------------
# TracebackFrame — structured single frame
# ---------------------------------------------------------------------------


class TracebackFrame(NamedTuple):
    """One frame from a serialized exception traceback.

    Mirrors the attributes of :class:`traceback.FrameSummary`.
    """

    filename: str
    """Absolute (or as-reported) source file path."""

    lineno: int
    """1-based line number within the source file."""

    name: str
    """Name of the function / method that owns the frame."""

    line: str | None
    """The source text of the line, or ``None`` when unavailable."""


# ---------------------------------------------------------------------------
# internal traceback helpers
# ---------------------------------------------------------------------------


def _extract_traceback_frames(exc: BaseException) -> list[tuple[str, int, str, str | None]] | None:
    """Return serialisable frame data from *exc*'s live ``__traceback__``.

    Returns ``None`` when ``exc.__traceback__`` is ``None`` (i.e. the exception
    was never raised, or was constructed without one).
    """
    tb = exc.__traceback__
    if tb is None:
        return None

    summaries = traceback.extract_tb(tb)
    return [
        (fs.filename, fs.lineno, fs.name, fs.line)
        for fs in summaries
    ]


def _attach_serialized_traceback(
    exc: BaseException,
    frames_raw: object,
) -> None:
    """Parse *frames_raw* and attach it to *exc* as ``__serialized_traceback__``."""
    if frames_raw is None:
        value: list[TracebackFrame] | None = None
    else:
        value = []
        for raw_frame in frames_raw:  # type: ignore[union-attr]
            filename = str(raw_frame[0])
            lineno = int(raw_frame[1])
            name = str(raw_frame[2])
            line = str(raw_frame[3]) if raw_frame[3] is not None else None
            value.append(TracebackFrame(filename=filename, lineno=lineno, name=name, line=line))

    # Best-effort attribute assignment — some exception subclasses may not
    # have a __dict__ (e.g. slotted exceptions without __serialized_traceback__
    # in their __slots__).
    try:
        object.__setattr__(exc, "__serialized_traceback__", value)
    except (AttributeError, TypeError):
        try:
            exc.__dict__["__serialized_traceback__"] = value  # type: ignore[index]
        except Exception:
            pass  # silently ignore — traceback info is supplementary


# ---------------------------------------------------------------------------
# chained exception helpers
# ---------------------------------------------------------------------------


def _dump_chained_exception(
    exc: BaseException | None,
    *,
    _depth: int,
) -> bytes | None:
    """Recursively serialize a chained exception up to *_MAX_CHAIN_DEPTH* levels."""
    if exc is None or _depth >= _MAX_CHAIN_DEPTH:
        return None
    return _dump_exception_payload(exc, _depth=_depth + 1)


def _load_chained_exception(data: object) -> BaseException | None:
    """Deserialize a chained exception byte payload (or ``None``)."""
    if data is None:
        return None
    if not isinstance(data, (bytes, bytearray)):
        return None
    return _load_exception_payload(bytes(data))


# ---------------------------------------------------------------------------
# exception payload dump / load
# ---------------------------------------------------------------------------


def _dump_exception_payload(exc: BaseException, *, _depth: int = 0) -> bytes:
    """Serialize *exc* to wire bytes (version 2 format).

    The payload includes:
    - exception class, args, and custom ``__dict__`` / slot state;
    - structured traceback frames (if the exception has a live ``__traceback__``);
    - recursively serialized ``__cause__`` and ``__context__`` up to
      :data:`_MAX_CHAIN_DEPTH` levels deep.
    """
    tb_frames = _extract_traceback_frames(exc)

    cause_data = _dump_chained_exception(exc.__cause__, _depth=_depth)

    # Avoid double-serializing __context__ when it is the same object as __cause__.
    raw_context = exc.__context__
    context_to_dump = None if raw_context is exc.__cause__ else raw_context
    context_data = _dump_chained_exception(context_to_dump, _depth=_depth)

    return _serialize_nested(
        (
            _EXC_FORMAT_V2,
            type(exc),
            exc.args,
            _dump_object_state(exc),
            tb_frames,
            cause_data,
            context_data,
            exc.__suppress_context__,
        )
    )


def _load_exception_payload(data: bytes) -> BaseException:
    """Deserialize a version 2 exception payload."""
    raw = _deserialize_nested(data)

    if not isinstance(raw, tuple) or len(raw) != _V2_LENGTH:
        raise ValueError(
            f"Invalid exception payload: expected {_V2_LENGTH}-tuple, "
            f"got {type(raw).__name__} of length {len(raw) if isinstance(raw, tuple) else 'N/A'}"
        )

    if raw[0] != _EXC_FORMAT_V2:
        raise ValueError(f"Unsupported exception payload version: {raw[0]!r}")

    exc_cls = raw[_V2_CLASS]
    if not isinstance(exc_cls, type) or not issubclass(exc_cls, BaseException):
        raise TypeError("Decoded exception class is not a BaseException subclass")

    args_obj = _require_tuple(raw[_V2_ARGS], name="Exception args")
    state_payload = raw[_V2_STATE]
    tb_frames_raw = raw[_V2_TRACEBACK]
    cause_data = raw[_V2_CAUSE]
    context_data = raw[_V2_CONTEXT]
    suppress_context = bool(raw[_V2_SUPPRESS])

    try:
        exc = exc_cls(*args_obj)
    except Exception:
        exc = BaseException.__new__(exc_cls)
        exc.args = args_obj

    _restore_object_state(exc, state_payload)

    # Attach structured traceback info (supplementary — never raises)
    _attach_serialized_traceback(exc, tb_frames_raw)

    # Restore chained exceptions
    exc.__cause__ = _load_chained_exception(cause_data)
    exc.__context__ = _load_chained_exception(context_data)
    exc.__suppress_context__ = suppress_context

    return exc


# ---------------------------------------------------------------------------
# BaseExceptionSerialized
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaseExceptionSerialized(ComplexSerialized[BaseException]):
    """Serializer for any :class:`BaseException` subclass.

    Wire representation: :attr:`Tags.BASE_EXCEPTION` (tag 203) followed by a
    version 2 exception payload (see module docstring for the format).

    The deserialized exception carries an additional ``__serialized_traceback__``
    attribute — a list of :class:`TracebackFrame` named-tuples when the original
    exception had a live traceback, or ``None`` otherwise.
    """

    TAG: ClassVar[int] = Tags.BASE_EXCEPTION

    @property
    def value(self) -> BaseException:
        return _load_exception_payload(self.decode())

    @classmethod
    def build_exception(
        cls,
        exc: BaseException,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_exception_payload(exc),
            codec=codec,
        )


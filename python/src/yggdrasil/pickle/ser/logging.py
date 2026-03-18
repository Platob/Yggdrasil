"""
logging.py – Serializers for Python :mod:`logging` objects.

Wire tags (all in the system range 200–299):

    LOGGING_LOGGER      (216) – :class:`logging.Logger` / :class:`logging.RootLogger`
    LOGGING_HANDLER     (217) – :class:`logging.Handler` and concrete subclasses
    LOGGING_FORMATTER   (218) – :class:`logging.Formatter`
    LOGGING_LOG_RECORD  (219) – :class:`logging.LogRecord`

Design notes
------------
- Loggers are reference-serialised by *name*: they are singletons managed by
  :func:`logging.getLogger`, so we only store the name + effective level and
  re-bind on load.
- Handlers are reconstructed from their class reference plus the subset of
  state that is safe and portable (level, formatter, filters by name).
  Handler-specific init args (filename, stream, host/port, …) are stored via
  ``__getstate__`` when available; otherwise we fall back to a dict extracted
  from ``__dict__``.
- Formatters store their format strings (fmt, datefmt, style, defaults).
- LogRecords store all their public fields as a plain tuple payload.

The payload format uses :func:`yggdrasil.pickle.ser.serialized.Serialized`
round-tripped msgpack-like nested bytes so that each sub-object goes through
the normal serializer stack.
"""
from __future__ import annotations

import logging
import logging.handlers as _logging_handlers
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "LoggingSerialized",
    "LoggerSerialized",
    "HandlerSerialized",
    "FormatterSerialized",
    "LogRecordSerialized",
]

# ---------------------------------------------------------------------------
# Payload version sentinels
# ---------------------------------------------------------------------------
_VERSION = 1

# ---------------------------------------------------------------------------
# Compact tuple-index helpers
# ---------------------------------------------------------------------------

# Logger payload = (version, name, level)
_LG_VERSION = 0
_LG_NAME = 1
_LG_LEVEL = 2

# Handler payload = (version, module, qualname, level, formatter_bytes_or_none, filter_names, extra_state_or_none)
_HD_VERSION = 0
_HD_MODULE = 1
_HD_QUALNAME = 2
_HD_LEVEL = 3
_HD_FORMATTER = 4
_HD_FILTER_NAMES = 5
_HD_EXTRA_STATE = 6

# Formatter payload = (version, fmt, datefmt, style, validate, defaults_or_none)
_FM_VERSION = 0
_FM_FMT = 1
_FM_DATEFMT = 2
_FM_STYLE = 3
_FM_VALIDATE = 4
_FM_DEFAULTS = 5

# LogRecord payload = (version, name, level, pathname, lineno, msg, args, exc_info_or_none, func_name, stack_info_or_none, task_name_or_none)
_LR_VERSION = 0
_LR_NAME = 1
_LR_LEVEL = 2
_LR_PATHNAME = 3
_LR_LINENO = 4
_LR_MSG = 5
_LR_ARGS = 6
_LR_EXC_INFO = 7
_LR_FUNC_NAME = 8
_LR_STACK_INFO = 9
_LR_TASK_NAME = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialize_nested(obj: object) -> bytes:
    """Serialize *obj* through the Serialized stack and return raw bytes."""
    return Serialized.from_python_object(obj).write_to().to_bytes()


def _deserialize_nested(blob: bytes) -> object:
    """Deserialize bytes previously produced by :func:`_serialize_nested`."""
    from yggdrasil.io import BytesIO
    return Serialized.read_from(BytesIO(blob), pos=0).as_python()


def _opt_serialize(obj: object | None) -> bytes | None:
    """Serialize *obj* if it is not ``None``."""
    if obj is None:
        return None
    return _serialize_nested(obj)


def _opt_deserialize(blob: bytes | None) -> object | None:
    """Deserialize *blob* if it is not ``None``."""
    if blob is None:
        return None
    return _deserialize_nested(blob)


def _handler_module_qualname(handler: logging.Handler) -> tuple[str, str]:
    cls = type(handler)
    mod = getattr(cls, "__module__", None) or "logging"
    qual = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", "Handler")
    return mod, qual


def _resolve_handler_class(module: str, qualname: str) -> type[logging.Handler]:
    """Import and resolve the handler class by module + qualname."""
    import importlib

    mod = importlib.import_module(module)
    obj: object = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)

    if not (isinstance(obj, type) and issubclass(obj, logging.Handler)):
        raise TypeError(
            f"Resolved {module}.{qualname!r} is not a logging.Handler subclass"
        )
    return obj  # type: ignore[return-value]


def _extract_handler_state(handler: logging.Handler) -> object | None:
    """
    Extract handler-specific state.

    Prefer ``__getstate__`` if defined beyond the base class; otherwise
    return a filtered copy of ``__dict__`` (excluding internal bookkeeping
    fields shared by all handlers).
    """
    _skip = frozenset({
        "level", "formatter", "filters", "_name", "lock",
        "_closed", "createLock",
    })

    base_getstate = getattr(logging.Handler, "__getstate__", None)
    custom_getstate = getattr(type(handler), "__getstate__", None)

    if custom_getstate is not None and custom_getstate is not base_getstate:
        try:
            return handler.__getstate__()  # type: ignore[attr-defined]
        except Exception:
            pass

    state: dict[str, object] = {}
    for k, v in getattr(handler, "__dict__", {}).items():
        if k.startswith("_") or k in _skip:
            continue
        state[k] = v
    return state if state else None


def _restore_handler_state(handler: logging.Handler, state: object) -> None:
    """Apply *state* produced by :func:`_extract_handler_state`."""
    if state is None:
        return

    base_setstate = getattr(logging.Handler, "__setstate__", None)
    custom_setstate = getattr(type(handler), "__setstate__", None)

    if custom_setstate is not None and custom_setstate is not base_setstate:
        try:
            handler.__setstate__(state)  # type: ignore[attr-defined]
            return
        except Exception:
            pass

    if isinstance(state, dict):
        for k, v in state.items():
            try:
                setattr(handler, k, v)
            except Exception:
                handler.__dict__[k] = v


def _filter_names(handler: logging.Handler) -> list[str]:
    """
    Return the names of all :class:`logging.Filter` objects attached to *handler*.

    Non-``Filter`` callables are silently skipped – they are not portable.
    """
    names: list[str] = []
    for f in getattr(handler, "filters", []):
        if isinstance(f, logging.Filter):
            names.append(f.name)
    return names


def _handler_try_reconstruct(
    cls: type[logging.Handler],
    extra_state: object | None,
) -> logging.Handler | None:
    """
    Try to reconstruct a handler from its class and extra state.

    Strategy:
    1. ``__new__`` + ``__setstate__``
    2. Zero-arg ``__init__``
    3. ``NullHandler`` fallback
    """
    # Try no-arg construction (covers NullHandler, MemoryHandler-ish, etc.)
    try:
        instance = cls.__new__(cls)
        logging.Handler.__init__(instance)  # always call base to set lock/filters
        _restore_handler_state(instance, extra_state)
        return instance
    except Exception:
        pass

    try:
        instance = cls()
        _restore_handler_state(instance, extra_state)
        return instance
    except Exception:
        pass

    return None


# ============================================================================
# Base class
# ============================================================================

@dataclass(frozen=True, slots=True)
class LoggingSerialized(Serialized[object]):
    """
    Abstract base for all logging-related serializers.

    Concrete subclasses override :meth:`as_python` and :meth:`from_python_object`.
    """

    TAG: ClassVar[int] = -1  # sentinel; overridden by subclasses

    def as_python(self) -> object:
        raise NotImplementedError

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, logging.LogRecord):
            return LogRecordSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, logging.Formatter):
            return FormatterSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, logging.Handler):
            return HandlerSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, logging.Logger):
            return LoggerSerialized.from_value(obj, metadata=metadata, codec=codec)

        return None


# ============================================================================
# Logger
# ============================================================================

@dataclass(frozen=True, slots=True)
class LoggerSerialized(LoggingSerialized):
    """
    Serializer for :class:`logging.Logger`.

    Loggers are singletons: we persist the *name* and *effective level* only.
    On deserialisation we call :func:`logging.getLogger` to obtain the
    canonical instance, and (optionally) set the level when the logger has not
    been explicitly configured yet.
    """

    TAG: ClassVar[int] = Tags.LOGGING_LOGGER

    @property
    def value(self) -> logging.Logger:
        payload: tuple[Any, ...] = _deserialize_nested(self.decode())  # type: ignore[assignment]
        name: str = payload[_LG_NAME]
        level: int = payload[_LG_LEVEL]
        logger = logging.getLogger(name) if name != "root" else logging.getLogger()
        # Only override level when the logger level is NOTSET (unset / inherited)
        if logger.level == logging.NOTSET and level != logging.NOTSET:
            logger.setLevel(level)
        return logger

    def as_python(self) -> logging.Logger:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: logging.Logger,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "LoggerSerialized":
        name = obj.name if obj.name != "root" else "root"
        payload: tuple[Any, ...] = (
            _VERSION,
            name,
            obj.level,
        )
        data = _serialize_nested(payload)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=data,
            metadata=metadata,
            codec=codec,
        )


# ============================================================================
# Handler
# ============================================================================

@dataclass(frozen=True, slots=True)
class HandlerSerialized(LoggingSerialized):
    """
    Serializer for :class:`logging.Handler` and its subclasses.

    Only portable state is captured:

    - Handler class (module + qualname)
    - Level
    - Formatter (if set)
    - Filter names
    - Handler-specific extra state (``__getstate__`` or ``__dict__``  subset)

    The handler is reconstructed via :func:`_handler_try_reconstruct`; if
    reconstruction fails, a plain :class:`logging.NullHandler` is returned
    instead.
    """

    TAG: ClassVar[int] = Tags.LOGGING_HANDLER

    @property
    def value(self) -> logging.Handler:
        payload: tuple[Any, ...] = _deserialize_nested(self.decode())  # type: ignore[assignment]

        module: str = payload[_HD_MODULE]
        qualname: str = payload[_HD_QUALNAME]
        level: int = payload[_HD_LEVEL]
        formatter_blob: bytes | None = payload[_HD_FORMATTER]
        filter_names: list[str] = payload[_HD_FILTER_NAMES]
        extra_state_blob: bytes | None = payload[_HD_EXTRA_STATE]

        # Resolve class
        try:
            cls_resolved = _resolve_handler_class(module, qualname)
        except Exception:
            cls_resolved = logging.NullHandler

        # Reconstruct extra state
        extra_state = _opt_deserialize(extra_state_blob)

        handler = _handler_try_reconstruct(cls_resolved, extra_state)
        if handler is None:
            handler = logging.NullHandler()

        handler.setLevel(level)

        # Restore formatter
        if formatter_blob is not None:
            fmt = _deserialize_nested(formatter_blob)
            if isinstance(fmt, logging.Formatter):
                handler.setFormatter(fmt)

        # Restore filters
        for fname in filter_names:
            handler.addFilter(logging.Filter(fname))

        return handler

    def as_python(self) -> logging.Handler:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: logging.Handler,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "HandlerSerialized":
        module, qualname = _handler_module_qualname(obj)
        level: int = obj.level

        formatter_blob: bytes | None = None
        if obj.formatter is not None:
            try:
                formatter_blob = _serialize_nested(obj.formatter)
            except Exception:
                formatter_blob = None

        filter_names = _filter_names(obj)
        extra_state = _extract_handler_state(obj)
        extra_state_blob: bytes | None = None
        if extra_state is not None:
            try:
                extra_state_blob = _serialize_nested(extra_state)
            except Exception:
                extra_state_blob = None

        payload: tuple[Any, ...] = (
            _VERSION,
            module,
            qualname,
            level,
            formatter_blob,
            filter_names,
            extra_state_blob,
        )
        data = _serialize_nested(payload)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=data,
            metadata=metadata,
            codec=codec,
        )


# ============================================================================
# Formatter
# ============================================================================

@dataclass(frozen=True, slots=True)
class FormatterSerialized(LoggingSerialized):
    """
    Serializer for :class:`logging.Formatter`.

    Stores format string, date format, style, validation flag, and any extra
    ``defaults`` mapping.  Custom :class:`logging.Formatter` subclasses that
    do not require special construction args will round-trip faithfully; more
    exotic subclasses are reconstructed as plain :class:`logging.Formatter`.
    """

    TAG: ClassVar[int] = Tags.LOGGING_FORMATTER

    @property
    def value(self) -> logging.Formatter:
        payload: tuple[Any, ...] = _deserialize_nested(self.decode())  # type: ignore[assignment]

        fmt: str | None = payload[_FM_FMT]
        datefmt: str | None = payload[_FM_DATEFMT]
        style: str = payload[_FM_STYLE]
        defaults: dict[str, Any] | None = payload[_FM_DEFAULTS]

        # Always reconstruct with validate=False so non-% format strings don't
        # fail PercentStyle validation before the correct style is applied.
        try:
            return logging.Formatter(
                fmt=fmt,
                datefmt=datefmt,
                style=style,
                validate=False,
                defaults=defaults,
            )
        except TypeError:
            # Older Python versions don't support all kwargs
            return logging.Formatter(fmt=fmt, datefmt=datefmt, style=style)

    def as_python(self) -> logging.Formatter:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: logging.Formatter,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "FormatterSerialized":
        datefmt: str | None = getattr(obj, "datefmt", None)
        style_obj = getattr(obj, "_style", None)
        defaults: dict[str, Any] | None = getattr(obj, "defaults", None)

        # Recover style char and the actual format string.
        # IMPORTANT: StrFormatStyle and StringTemplateStyle are subclasses of
        # PercentStyle in Python 3.12, so check them FIRST.
        style_char: str = "%"
        fmt: str | None = getattr(obj, "_fmt", None) or getattr(obj, "fmt", None)

        if style_obj is not None:
            if isinstance(style_obj, logging.StrFormatStyle):
                style_char = "{"
                fmt = getattr(style_obj, "_fmt", fmt)
            elif isinstance(style_obj, logging.StringTemplateStyle):
                style_char = "$"
                # StringTemplateStyle stores fmt in _fmt (set by PercentStyle.__init__)
                # but the Template object is in style_obj.template; read _fmt directly.
                fmt = getattr(style_obj, "_fmt", fmt)
            else:
                # Plain PercentStyle
                style_char = "%"
                fmt = getattr(style_obj, "_fmt", fmt)

        payload: tuple[Any, ...] = (
            _VERSION,
            fmt,
            datefmt,
            style_char,
            True,   # validate slot – kept for wire compat; always reconstruct with validate=False
            defaults,
        )
        data = _serialize_nested(payload)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=data,
            metadata=metadata,
            codec=codec,
        )


# ============================================================================
# LogRecord
# ============================================================================

@dataclass(frozen=True, slots=True)
class LogRecordSerialized(LoggingSerialized):
    """
    Serializer for :class:`logging.LogRecord`.

    All public fields are persisted.  Exception info (``exc_info``) is
    serialised as a pickled bytes blob when the exception object is available;
    otherwise it is dropped.
    """

    TAG: ClassVar[int] = Tags.LOGGING_LOG_RECORD

    @property
    def value(self) -> logging.LogRecord:
        payload: tuple[Any, ...] = _deserialize_nested(self.decode())  # type: ignore[assignment]

        name: str = payload[_LR_NAME]
        level: int = payload[_LR_LEVEL]
        pathname: str = payload[_LR_PATHNAME]
        lineno: int = payload[_LR_LINENO]
        msg: object = payload[_LR_MSG]
        args: tuple[Any, ...] | dict[str, Any] | None = payload[_LR_ARGS]
        exc_info_blob: tuple[str, str, str] | None = payload[_LR_EXC_INFO]
        func_name: str | None = payload[_LR_FUNC_NAME]
        stack_info: str | None = payload[_LR_STACK_INFO]
        task_name: str | None = payload[_LR_TASK_NAME] if len(payload) > _LR_TASK_NAME else None

        # Restore exc_info
        exc_info: tuple[Any, Any, Any] | None = None
        if exc_info_blob is not None:
            try:
                # exc_info_blob is (exc_module, exc_qualname, exc_str_repr)
                exc_mod, exc_qual, exc_str = exc_info_blob  # type: ignore[misc]
                import importlib
                mod = importlib.import_module(exc_mod)
                exc_cls = mod
                for part in exc_qual.split("."):
                    exc_cls = getattr(exc_cls, part)
                exc_val = exc_cls(exc_str)
                exc_info = (exc_cls, exc_val, None)
            except Exception:
                exc_info = None

        record = logging.LogRecord(
            name=name,
            level=level,
            pathname=pathname,
            lineno=lineno,
            msg=msg,
            args=args or (),
            exc_info=exc_info,
            func=func_name or "",
            sinfo=stack_info,
        )

        if task_name is not None:
            try:
                record.taskName = task_name  # Python 3.12+
            except Exception:
                pass

        return record

    def as_python(self) -> logging.LogRecord:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: logging.LogRecord,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "LogRecordSerialized":
        name: str = obj.name
        level: int = obj.levelno
        pathname: str = obj.pathname
        lineno: int = obj.lineno
        msg: object = obj.msg
        args: Any = obj.args
        func_name: str | None = obj.funcName
        stack_info: str | None = getattr(obj, "stack_info", None)
        task_name: str | None = getattr(obj, "taskName", None)

        # Serialize exc_info using pickle as a compact binary blob
        exc_info_blob: bytes | None = None
        exc_info = getattr(obj, "exc_info", None)
        if exc_info is not None and exc_info[0] is not None:
            try:
                exc_cls = exc_info[0]
                exc_val = exc_info[1]
                exc_info_blob = (
                    getattr(exc_cls, "__module__", "builtins"),
                    getattr(exc_cls, "__qualname__", exc_cls.__name__),
                    str(exc_val) if exc_val is not None else "",
                )
            except Exception:
                exc_info_blob = None

        payload: tuple[Any, ...] = (
            _VERSION,
            name,
            level,
            pathname,
            lineno,
            msg,
            args,
            exc_info_blob,
            func_name,
            stack_info,
            task_name,
        )
        data = _serialize_nested(payload)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=data,
            metadata=metadata,
            codec=codec,
        )


# ============================================================================
# Registration
# ============================================================================

for _pytype, _cls in (
    (logging.Logger, LoggerSerialized),
    (logging.RootLogger, LoggerSerialized),
    (logging.Handler, HandlerSerialized),
    (logging.NullHandler, HandlerSerialized),
    (logging.StreamHandler, HandlerSerialized),
    (logging.FileHandler, HandlerSerialized),
    (_logging_handlers.MemoryHandler, HandlerSerialized),
    (_logging_handlers.RotatingFileHandler, HandlerSerialized),
    (_logging_handlers.TimedRotatingFileHandler, HandlerSerialized),
    (_logging_handlers.SocketHandler, HandlerSerialized),
    (_logging_handlers.DatagramHandler, HandlerSerialized),
    (_logging_handlers.SysLogHandler, HandlerSerialized),
    (_logging_handlers.HTTPHandler, HandlerSerialized),
    (_logging_handlers.QueueHandler, HandlerSerialized),
    (logging.Formatter, FormatterSerialized),
    (logging.LogRecord, LogRecordSerialized),
):
    Tags.register_class(_cls, pytype=_pytype)


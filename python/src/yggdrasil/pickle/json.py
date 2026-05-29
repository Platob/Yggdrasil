# yggdrasil/pickle/json.py
from __future__ import annotations

import base64
import json as _json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, IO, Mapping, overload
from uuid import UUID

import orjson

__all__ = ["load", "loads", "dump", "dumps"]

# orjson is a hard dependency: 3-10x faster than stdlib for both
# encode and decode, emits UTF-8 bytes natively, and serializes
# datetime / date / time / UUID / dataclass without a default hook.
# We fall back to stdlib only when the caller picks an option orjson
# can't express (ensure_ascii=True, indent != 2, custom separators).

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def _is_namedtuple_instance(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_asdict")


def _decode_bytes_best_effort(data: bytes, *, errors: str = "strict") -> str:
    try:
        return data.decode("utf-8", errors=errors)
    except UnicodeDecodeError:
        return base64.b64encode(data).decode("ascii")


def _default_safe(obj: Any) -> Any:
    """Conservative leaf-only conversions.

    Used with json.dumps(default=...). This does not pre-normalize nested
    structures or mapping keys.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping | MappingProxyType):
        return dict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if _is_namedtuple_instance(obj):
        return obj._asdict()
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            raise TypeError(f"Cannot encode bytes as JSON: {obj!r}")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _default_safe_orjson(obj: Any) -> Any:
    """Same as :func:`_default_safe` but skips types orjson handles natively.

    orjson already serializes :class:`datetime`, :class:`date`,
    :class:`time`, :class:`UUID`, and dataclasses without a default
    hook — so the hook only fires for the rest of the type set.
    """
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping | MappingProxyType):
        return dict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if _is_namedtuple_instance(obj):
        return obj._asdict()
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            raise TypeError(f"Cannot encode bytes as JSON: {obj!r}")
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _orjson_can_handle(
    *,
    ensure_ascii: bool,
    indent: int | None,
    separators: tuple[str, str] | None,
) -> bool:
    """Whether orjson can satisfy the requested formatting options.

    orjson always emits UTF-8 bytes (no ``ensure_ascii``), supports
    only 2-space indent (``OPT_INDENT_2``), and writes compact
    separators (``,`` + ``:``) — or with indent the JSON-canonical
    ``,`` + ``: ``. Anything else falls back to stdlib json.
    """
    if ensure_ascii:
        return False
    if indent not in (None, 2):
        return False
    if separators is not None:
        if indent == 2:
            return separators == (",", ": ")
        return separators == (",", ":")
    return True


def _orjson_options(*, sort_keys: bool, indent: int | None) -> int:
    opt = 0
    if sort_keys:
        opt |= orjson.OPT_SORT_KEYS
    if indent == 2:
        opt |= orjson.OPT_INDENT_2
    return opt


def _normalize_key_broad(key: Any, *, errors: str) -> str:
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, (int, float)):
        return str(key)
    if isinstance(key, (bytes, bytearray, memoryview)):
        return _decode_bytes_best_effort(bytes(key), errors=errors)
    if isinstance(key, (datetime, date, time)):
        return key.isoformat()
    if isinstance(key, timedelta):
        return str(key.total_seconds())
    if isinstance(key, Decimal):
        return str(key)
    if isinstance(key, UUID):
        return str(key)
    if isinstance(key, Enum):
        return _normalize_key_broad(key.value, errors=errors)
    if isinstance(key, Path):
        return str(key)
    return str(key)


def _normalize_obj_broad(obj: Any, *, errors: str) -> Any:
    """Aggressive recursive normalization for broader compatibility."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if is_dataclass(obj):
        return _normalize_obj_broad(asdict(obj), errors=errors)

    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()

    if isinstance(obj, timedelta):
        return obj.total_seconds()

    if isinstance(obj, Decimal):
        return str(obj)

    if isinstance(obj, UUID):
        return str(obj)

    if isinstance(obj, Enum):
        return _normalize_obj_broad(obj.value, errors=errors)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return _decode_bytes_best_effort(bytes(obj), errors=errors)

    if isinstance(obj, Mapping | MappingProxyType):
        return {
            _normalize_key_broad(k, errors=errors): _normalize_obj_broad(v, errors=errors)
            for k, v in obj.items()
        }

    if _is_namedtuple_instance(obj):
        return _normalize_obj_broad(obj._asdict(), errors=errors)

    if isinstance(obj, (list, tuple)):
        return [_normalize_obj_broad(v, errors=errors) for v in obj]

    if isinstance(obj, (set, frozenset)):
        return [_normalize_obj_broad(v, errors=errors) for v in obj]

    if hasattr(obj, "model_dump"):
        try:
            return _normalize_obj_broad(obj.model_dump(mode="json"), errors=errors)
        except TypeError:
            return _normalize_obj_broad(obj.model_dump(), errors=errors)

    if hasattr(obj, "dict"):
        try:
            return _normalize_obj_broad(obj.dict(), errors=errors)
        except TypeError:
            pass

    if hasattr(obj, "_asdict"):
        try:
            return _normalize_obj_broad(obj._asdict(), errors=errors)
        except TypeError:
            pass

    if hasattr(obj, "__json__"):
        try:
            return _normalize_obj_broad(obj.__json__(), errors=errors)
        except TypeError:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return _normalize_obj_broad(vars(obj), errors=errors)
        except TypeError:
            pass

    return str(obj)


def _try_parse_string(value: str) -> Any:
    """Best-effort rich type parsing for unsafe loads.

    Order matters. Datetime before date/time. Keep this conservative.
    """
    if not value:
        return value

    # datetime
    try:
        # Handles offsets too, e.g. 2026-04-14T12:34:56+00:00
        return datetime.fromisoformat(value)
    except ValueError:
        pass

    # date
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass

    # time
    try:
        return time.fromisoformat(value)
    except ValueError:
        pass

    # uuid
    if _UUID_RE.match(value):
        try:
            return UUID(value)
        except ValueError:
            pass

    return value


def _restore_obj_broad(obj: Any) -> Any:
    """Recursively post-process parsed JSON into richer Python objects."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        return _try_parse_string(obj)

    if isinstance(obj, list):
        return [_restore_obj_broad(v) for v in obj]

    if isinstance(obj, dict):
        return {k: _restore_obj_broad(v) for k, v in obj.items()}

    return obj


@overload
def loads(
    s: str | bytes | bytearray | memoryview,
    *,
    encoding: str = ...,
    errors: str = ...,
    safe: bool = ...,
) -> Any: ...


def loads(
    s: str | bytes | bytearray | memoryview,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    safe: bool = True,
) -> Any:
    """Parse JSON from *s*.

    safe=True:
        Parse JSON only. No post-processing.

    safe=False:
        Parse JSON, then recursively try to restore richer Python types from
        string values such as datetime/date/time/UUID.
    """
    if isinstance(s, str):
        try:
            obj = orjson.loads(s)
        except orjson.JSONDecodeError:
            obj = _json.loads(s)
    else:
        try:
            # orjson parses bytes / bytearray / memoryview natively — pass
            # a memoryview straight through (zero-copy) instead of
            # materializing ``bytes(s)``, so callers can hand us a view
            # over an existing buffer (e.g. an IO holder's ``read_mv``).
            obj = orjson.loads(s)
        except orjson.JSONDecodeError:
            # orjson rejects malformed UTF-8 outright; fall back to
            # stdlib so caller-controlled ``errors=`` still applies.
            text = bytes(s).decode(encoding, errors=errors)
            obj = _json.loads(text)

    if safe:
        return obj
    return _restore_obj_broad(obj)


@overload
def dumps(
    obj: Any,
    *,
    encoding: str | None = ...,
    errors: str = ...,
    ensure_ascii: bool = ...,
    sort_keys: bool = ...,
    indent: int | None = ...,
    separators: tuple[str, str] | None = ...,
    safe: bool = ...,
    to_bytes: bool = True,
) -> bytes: ...


@overload
def dumps(
    obj: Any,
    *,
    encoding: str | None = ...,
    errors: str = ...,
    ensure_ascii: bool = ...,
    sort_keys: bool = ...,
    indent: int | None = ...,
    separators: tuple[str, str] | None = ...,
    safe: bool = ...,
    to_bytes: bool = False,
) -> str: ...


def dumps(
    obj: Any,
    *,
    encoding: str | None = "utf-8",
    errors: str = "strict",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    safe: bool = True,
    to_bytes: bool = True,
) -> bytes | str:
    """Serialize *obj* to JSON.

    safe=True:
        Conservative mode. No recursive pre-normalization. Uses stdlib JSON
        semantics and a narrow default hook for leaf values only.

    safe=False:
        Broad compatibility mode. Recursively normalizes nested mappings,
        keys, sequences, bytes-like objects, and miscellaneous Python objects.
    """
    if separators is None and indent is None:
        separators = (",", ":")

    use_orjson = _orjson_can_handle(
        ensure_ascii=ensure_ascii, indent=indent, separators=separators,
    )

    if use_orjson:
        opt = _orjson_options(sort_keys=sort_keys, indent=indent)
        if safe:
            data = orjson.dumps(obj, default=_default_safe_orjson, option=opt)
        else:
            data = orjson.dumps(
                _normalize_obj_broad(obj, errors=errors), option=opt,
            )
        # orjson always emits UTF-8 bytes; re-encode when the caller
        # asked for a different encoding (or wanted a str).
        target_enc = (encoding or "utf-8").lower()
        if not to_bytes:
            return data.decode("utf-8", errors=errors)
        if target_enc in ("utf-8", "utf8"):
            return data
        return data.decode("utf-8", errors=errors).encode(
            encoding, errors=errors,
        )

    # Stdlib fallback for option combinations orjson can't express.
    if safe:
        text = _json.dumps(
            obj,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=_default_safe,
        )
    else:
        text = _json.dumps(
            _normalize_obj_broad(obj, errors=errors),
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
        )

    if to_bytes:
        return text.encode(encoding or "utf-8", errors=errors)
    return text


def load(
    fp: IO[Any],
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    safe: bool = True,
) -> Any:
    """Read and parse JSON from a file-like object."""
    data = fp.read()
    return loads(data, encoding=encoding, errors=errors, safe=safe)


def dump(
    obj: Any,
    fp: IO[Any],
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    safe: bool = True,
) -> None:
    """Serialize *obj* to JSON and write it to a file-like object."""
    value = dumps(
        obj,
        encoding=encoding,
        errors=errors,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        safe=safe,
        to_bytes=True,
    )
    try:
        fp.write(value)
    except TypeError:
        fp.write(value.decode(encoding, errors=errors))  # type: ignore[arg-type]
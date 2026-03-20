# yggdrasil/pickle/json.py
from __future__ import annotations

import json as _json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from typing import Any, IO, overload

__all__ = ["load", "loads", "dump", "dumps"]


def _default_encoder(obj: Any) -> Any:
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable "
                f"(bytes not valid UTF-8)"
            )
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@overload
def loads(s: str | bytes | bytearray | memoryview, *, encoding: str = ..., errors: str = ...) -> Any: ...


def loads(
    s: str | bytes | bytearray | memoryview,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> Any:
    """Parse JSON from *s* (str, bytes, bytearray, or memoryview)."""
    if isinstance(s, (bytes, bytearray, memoryview)):
        s = bytes(s).decode(encoding, errors=errors)
    return _json.loads(s)


def dumps(
    obj: Any,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    default: Any = _default_encoder,
) -> bytes:
    """Serialise *obj* to compact UTF-8 JSON bytes."""
    if separators is None and indent is None:
        separators = (",", ":")

    text = _json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=default,
    )
    return text.encode(encoding, errors=errors)


def load(fp: IO[Any], *, encoding: str = "utf-8", errors: str = "strict") -> Any:
    """Read and parse JSON from a file-like object."""
    data = fp.read()
    return loads(data, encoding=encoding, errors=errors)


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
    default: Any = _default_encoder,
) -> None:
    """Serialise *obj* to JSON and write it to a file-like object."""
    b = dumps(
        obj,
        encoding=encoding,
        errors=errors,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        default=default,
    )
    try:
        fp.write(b)
    except TypeError:
        fp.write(b.decode(encoding, errors=errors))  # type: ignore[arg-type]


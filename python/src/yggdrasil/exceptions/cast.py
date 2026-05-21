"""Cast-related exception types.

Lives under :mod:`yggdrasil.exceptions` so any module can import these
without pulling :mod:`yggdrasil.data` or :mod:`yggdrasil.arrow` first —
exceptions are reachable from the hot path without ordering surprises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .base import YGGException

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field


__all__ = ["CastError"]


class CastError(YGGException, pa.ArrowInvalid):
    """Raised when casting a value / array / table to a target field fails.

    Carries the source and target :class:`Field` so the message can name
    which column was being cast — without these, debugging a multi-column
    write meant guessing which child raised. Subclassing
    :class:`pyarrow.ArrowInvalid` keeps every existing
    ``except pa.ArrowInvalid`` handler in the wider codebase catching
    these unchanged; subclassing :class:`YGGException` lets a generic
    ``except YGGException`` catch every error this library raises.

    Message shape (single line so logs stay readable):

    ::

        cast payload: string -> payload: list<struct<a:int64, b:string>> failed:
        Invalid JSON at row 0: ... Value: 'pypsa'.

    ``original`` preserves the underlying exception for ``raise ... from``
    chains and for callers that want the raw pyarrow / json failure
    object.
    """

    def __init__(
        self,
        reason: str,
        *,
        source: "Field | None" = None,
        target: "Field | None" = None,
        original: BaseException | None = None,
    ) -> None:
        self.reason = reason
        self.source = source
        self.target = target
        self.original = original
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        src_desc = _describe_field(self.source) or "?"
        tgt_desc = _describe_field(self.target) or "?"
        return f"cast {src_desc} -> {tgt_desc} failed: {self.reason}"


def _describe_field(field: "Field | None") -> str | None:
    """Render a :class:`Field` as ``name: dtype`` on a single line.

    Prefers the Arrow type representation when available — ``DataType``'s
    own ``__str__`` formats nested types multi-line, which turns log
    lines into a mess. Falls back through ``arrow_type`` → ``type`` →
    ``dtype`` → ``repr(field)`` so this stays useful for non-yggdrasil
    field-shaped objects too. Only touches attributes every Field
    implementation exposes, so this module stays importable without
    dragging :mod:`yggdrasil.data` in.
    """
    if field is None:
        return None
    name = _safe_attr(field, "name")
    # ``arrow_type`` is a cached property on Field — for a self-referential
    # struct (a CastError target that recurses on itself), accessing it
    # triggers ``StructType.to_arrow`` → ``Field.to_arrow_field`` → ``arrow_type``
    # forever and the error formatter never returns. Guard the chain so the
    # original CastError surfaces with a degraded but finite description.
    dtype: Any = (
        _safe_attr(field, "arrow_type")
        or _safe_attr(field, "type")
        or _safe_attr(field, "dtype")
    )
    if name and dtype is not None:
        return f"{name}: {dtype}"
    if dtype is not None:
        return _safe_str(dtype)
    if name:
        return str(name)
    return _safe_str(field, default=object.__repr__(field))


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _safe_str(value: Any, default: str = "?") -> str:
    try:
        return str(value)
    except Exception:
        return default

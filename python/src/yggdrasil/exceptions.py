"""Yggdrasil-wide exception types.

Subclasses live here so any module can import them without pulling in
``data`` or ``arrow`` first — exceptions need to be importable from the
hot path without ordering surprises.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field


__all__ = [
    "YGGException",
    "CastError",
]


class YGGException(Exception):
    pass


class CastError(YGGException, pa.ArrowInvalid):
    """Raised when casting a value / array / table to a target field fails.

    Carries the source and target :class:`Field` so the message can name
    which column was being cast — without these, debugging a multi-column
    write meant guessing which child raised. Subclassing
    :class:`pyarrow.ArrowInvalid` keeps every existing
    ``except pa.ArrowInvalid`` handler in the wider codebase catching
    these unchanged.

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
        source_field: "Field | None" = None,
        target_field: "Field | None" = None,
        original: BaseException | None = None,
    ) -> None:
        self.reason = reason
        self.source_field = source_field
        self.target_field = target_field
        self.original = original
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        src_desc = _describe_field(self.source_field) or "?"
        tgt_desc = _describe_field(self.target_field) or "?"
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
    name = getattr(field, "name", None)
    dtype: Any = (
        getattr(field, "arrow_type", None)
        or getattr(field, "type", None)
        or getattr(field, "dtype", None)
    )
    if name and dtype is not None:
        return f"{name}: {dtype}"
    if dtype is not None:
        return str(dtype)
    if name:
        return str(name)
    return repr(field)

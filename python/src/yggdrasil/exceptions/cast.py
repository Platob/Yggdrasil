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
    """Render a :class:`Field` via its own pretty format.

    :class:`Field` defines ``__str__`` / ``__repr__`` as
    :meth:`Field.pretty_format`, so ``str(field)`` already produces
    the canonical representation used everywhere else in the library
    (single-line ``field: 'name' <dtype> {markers}`` for primitives,
    multi-line tree for nested struct / list / map). Going through
    the field's own format keeps the error message in lock-step with
    the rest of the library — no second projection of names /
    nullability / markers / comments — and avoids the
    self-referential-struct recursion the old ``arrow_type``
    fallback could hit (``Field.to_arrow_field`` → ``arrow_type`` →
    ``Field.to_arrow_field``…).

    :func:`_safe_str` keeps the formatter finite for non-yggdrasil
    field-shaped objects (or a half-built field whose ``pretty_format``
    raises) so the original :class:`CastError` always surfaces, even
    when its own description fails to render.
    """
    if field is None:
        return None
    return _safe_str(field, default=object.__repr__(field))


def _safe_str(value: Any, default: str = "?") -> str:
    try:
        return str(value)
    except Exception:
        return default
